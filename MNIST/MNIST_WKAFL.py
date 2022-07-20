import math
import numpy as np
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import rawDatasetsLoader
from datetime import datetime
import torchvision
import copy
import time
import random
import heapq

# 日志设置
logging.basicConfig(level=logging.INFO
                    , filename="./logs/MNIST_WKAFL.logs"
                    , filemode="w"
                    , format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                    , datefmt="%Y-%m-%d %H:%M:%S"
                    )
logging.info("This is  INFO !!")

hook = sy.TorchHook(torch)
date = datetime.now().strftime('%Y-%m-%d %H:%M')


class Argument:
    def __init__(self):
        self.user_num = 100  # number of total clients P
        self.K = 10  # number of participant clients K
        self.update_num = 15  # 每轮更新的客户端数目
        self.CB1 = 100  # clip parameter in both stages
        self.CB2 = 10  # clip parameter B at stage two
        self.lr = 0.005  # learning rate of global model
        self.batch_size = 4  # batch size of each client for local training
        self.itr_test = 10  # number of iterations for the two neighbour tests on test datasets
        self.itr_train = 100  # number of iterations for the two neighbour tests on training datasets
        self.test_batch_size = 128  # batch size for test datasets
        self.total_iterations = 1500  # total number of iterations
        self.threshold = 0.3  # threshold to judge whether gradients are consistent
        self.alpha = 0.1  # parameter for momentum to alleviate the effect of non-IID data
        self.seed = 1  # parameter for the server to initialize the model
        self.classes = 1  # number of data classes on each client, which can determine the level of non-IID data
        self.cuda_use = True
        self.train_data_size = 100000
        self.test_data_size = 10000
        self.time_maker_ai_min = 0.2
        self.time_maker_ai_max = 0.8
        self.time_maker_tau = 5

args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
device_cpu = torch.device("cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


##################################获取模型层数和各层的形状#############
def GetModelLayers(model):
    Layers_shape = []
    Layers_nodes = []
    for idx, params in enumerate(model.parameters()):
        Layers_num = idx
        Layers_shape.append(params.shape)
        Layers_nodes.append(params.numel())
    return Layers_num + 1, Layers_shape, Layers_nodes


##################################设置各层的梯度为0#####################
def ZerosGradients(Layers_shape, device):
    ZeroGradient = []
    for i in range(len(Layers_shape)):
        ZeroGradient.append(torch.zeros(Layers_shape[i], device=device))
    return ZeroGradient

################################调整学习率###############################
def lr_adjust(args, tau):
    tau = 0.1 * tau + 1
    lr = args.lr / tau
    return lr


#################################计算范数################################
def L_norm(Tensor, device):
    norm_Tensor = torch.tensor([0.], device=device)
    for i in range(len(Tensor)):
        norm_Tensor += Tensor[i].float().norm() ** 2
    return norm_Tensor.sqrt()


################################# 计算角相似度 ############################
def similarity(user_Gradients, yun_Gradients, device):
    sim = torch.tensor([0.], device=device)
    for i in range(len(user_Gradients)):
        sim = sim + torch.sum(user_Gradients[i] * yun_Gradients[i])
    if L_norm(user_Gradients, device) == 0:
        print('梯度为0.')
        sim = torch.tensor([1.], device=device)
        return sim
    sim = sim / (L_norm(user_Gradients, device) * L_norm(yun_Gradients, device))
    return sim


#################################聚合####################################
def aggregation(Collect_Gradients, K_Gradients, weight, Layers_shape, args, device, Clip=False):
    sim = torch.zeros([args.K], device=device)
    Gradients_Total = torch.zeros([args.K + 1], device=device)

    for i in range(args.K):
        Gradients_Total[i] = L_norm(K_Gradients[i], device)
    Gradients_Total[args.K] = L_norm(Collect_Gradients, device)
    # print('Gradients_norm', Gradients_Total)

    for i in range(args.K):
        sim[i] = similarity(K_Gradients[i], Collect_Gradients, device)
    index = (sim > args.threshold)
    # print('sim:', sim)
    if sum(index) == 0:
        print("相似度均较低")
        return Collect_Gradients

    Collect_Gradients = ZerosGradients(Layers_shape, device)

    totalSim = []
    Sel_Gradients = []
    for i in range(args.K):
        if sim[i] > args.threshold:
            totalSim.append((torch.exp(sim[i] * 10) * weight[i]).tolist())
            Sel_Gradients.append(K_Gradients[i])
    totalSim = torch.tensor(totalSim, device=device)
    totalSim = totalSim / torch.sum(totalSim)

    for i in range(len(totalSim)):
        Gradients_Sample = Sel_Gradients[i]
        if Clip:
            standNorm = L_norm(Collect_Gradients, device)
            Gradients_Sample = TensorClip(Gradients_Sample, args.CB2 * standNorm, device)
        for j in range(len(K_Gradients[i])):
            Collect_Gradients[j] += Gradients_Sample[j] * totalSim[i]
    return Collect_Gradients


################################ 定义剪裁 #################################
def TensorClip(Tensor, ClipBound, device):
    norm_Tensor = L_norm(Tensor, device)
    if ClipBound < norm_Tensor:
        for i in range(Layers_num):
            Tensor[i] = 1. * Tensor[i] * ClipBound / norm_Tensor
    return Tensor


############################定义测试函数################################
def test(model, test_loader, device):
    model.eval()
    test_loader_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    test_acc = 0
    # 测试数据不追踪梯度
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data = data.unsqueeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            cur_loss = F.nll_loss(output, target.long(), reduction='sum').item()  # sum up batch loss

            test_loss += cur_loss
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_loader_len

    test_acc = correct / test_loader_len

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_loader_len, 100. * test_acc))
    return test_loss, test_acc


##########################定义训练过程，返回梯度########################
def train(learning_rate, train_model, train_data, train_target, device, gradient=True):
    train_model.train()
    train_model.zero_grad()

    train_data = train_data.unsqueeze(1)
    output = train_model(train_data.float())
    loss_func = F.nll_loss
    loss = loss_func(output, train_target.long())
    loss.backward()

    Gradients_Tensor = []
    if gradient == False:
        for params in train_model.parameters():
            Gradients_Tensor.append(-learning_rate * params.grad.data)  # 返回-lr*grad
    if gradient == True:
        for params in train_model.parameters():
            Gradients_Tensor.append(params.grad.data)  # 把各层的梯度添加到张量Gradients_Tensor
    return Gradients_Tensor, loss


class TimeMaker:
    def __init__(self, ai, taui, di):
        self.ai = ai
        self.mui = 1 / ai
        self.taui = taui
        self.di = di

    def make(self):
        p = random.random()
        t = math.log(1-p) * self.taui * self.di / (- self.mui) + self.ai * self.taui * self.di
        return t


#  下发全局模型
def refresh_workers_model(model_server, workers_refresh):
    for client in workers_refresh:
        model_client = models[client]
        for idx, (params_server, params_client) in enumerate(zip(model_server.parameters(), model_client.parameters())):
            params_client.data = copy.deepcopy(params_server.data)
        models[client] = model_client

#  更新客户端模型轮次
def refresh_workers_itr(itr, workers_refresh):
    for client in workers_refresh:
        itrs[client] = itr

#  选取客户端下发全局模型并更新客户端模型轮次
def refresh_workers(update_num):
    users_selectable = set(users)
    users_unselectable = set(aggregating_dict)
    users_selectable.discard(users_unselectable)

    if update_num <= len(list(users_selectable)):
        workers_refresh = random.sample(list(users_selectable), args.update_num)
        refresh_workers_model(model, workers_refresh)
        refresh_workers_itr(itr, workers_refresh)
        for worker in workers_refresh:
            aggregating_dict[worker] = time_makers[worker].make()
    else:
        workers_refresh = list(users_selectable)
        refresh_workers_model(model, workers_refresh)
        refresh_workers_itr(itr, workers_refresh)
        for worker in workers_refresh:
            aggregating_dict[worker] = time_makers[worker].make()

#################
#  服务端和客户端  #
#################
print('start to gen model and workers')

model = Net()  # 全局模型
model.cuda()
workers = []
users = []
models = {}  # 模型
itrs = {}   # 各个客户端本地模型所处轮次
time_makers = {}  # 客户端运行时间生成器
aggregating_dict = {}  # 训练中或训练结束待聚合的客户端


for i in range(1, args.user_num + 1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i, i))
    exec('models["user{}"] = copy.deepcopy(model)'.format(i))
    exec('workers.append(user{})'.format(i))
    exec('users.append("user{}")'.format(i))
    exec('itrs["user{}"] = {}'.format(i, 1))
    exec('time_makers["user{}"] = TimeMaker(random.uniform(args.time_maker_ai_min, args.time_maker_ai_max), '
         'args.time_maker_tau, args.batch_size)'.format(i))

#################
#     数据载入   #
#################
print('start to load data')

dataType = 'mnist'  # 可选bymerge, byclass, digits, mnist, letters, balanced
datasets = rawDatasetsLoader.loadDatesets(
    trainDataSize=args.train_data_size,
    testDataSize=args.test_data_size,
    dataType=dataType
)

# 训练集，测试集
federated_data, datasNum = rawDatasetsLoader.dataset_federate_noniid(datasets, workers, args)
test_data = rawDatasetsLoader.testImages(datasets)
del datasets

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.test_batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

# 定义记录字典
logs = {'aggregation': [], 'itr': [], 'train_loss': [], 'test_loss': [], 'test_acc': [], 'staleness': [], 'time': []}

#################
# 开始联邦学习过程 #
#################
print('start to run fl')

# 获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
e = torch.exp(torch.tensor(1., device=device))
Collect_Gradients = ZerosGradients(Layers_shape, device)

fl_time = 0
for itr in range(1, args.total_iterations + 1):
    # step 1
    # 随机选取客户端下发最新模型进行训练, 没有训练完的不能被选择
    refresh_workers(args.update_num)
    while len(aggregating_dict) < args.K:
        refresh_workers(args.update_num)

    # step 2
    # 从客户端中选取模型进行聚合
    # 最小的K个训练时间，视为已经训练完
    aggregating_workers = heapq.nsmallest(args.K, aggregating_dict.keys())
    aggregating_workers_time = heapq.nsmallest(args.K, aggregating_dict.values())
    aggregating_workers_time_max = max(aggregating_workers_time)

    # 其余客户端减去本轮时间
    for key in aggregating_workers:
        aggregating_dict.pop(key)
    for key, value in aggregating_dict.items():
        if value < aggregating_workers_time_max:
            aggregating_dict[key] = 0
        else:
            aggregating_dict[key] -= aggregating_workers_time_max

    # step 3
    # 聚合时执行实际的train 未selected_workers则按worker_num随机选择
    federated_train_loader = sy.FederatedDataLoader(
        federated_data,
        batch_size=args.batch_size,
        shuffle=True,
        worker_num=args.K,
        batch_num=1,
        selected_workers=aggregating_workers
    )

    workers_list = federated_train_loader.workers  # 当前回合抽取的用户列表

    K_Gradients = []
    Loss_train = torch.tensor(0., device=device)
    weight = []
    K_tau = []
    tau_avg = 0

    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        tau = itr - itrs[train_data.location.id] + 1
        K_tau.append(tau)  # 添加延时信息
        tau_avg += tau
        model_round = models[train_data.location.id]

        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()

        Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, device, gradient=True)

        if itr > 1:
            for j in range(Layers_num):
                Gradients_Sample[j] = Gradients_Sample[j] + args.alpha * Collect_Gradients[j]
        K_Gradients.append(TensorClip(Gradients_Sample, args.CB1, device))
        Loss_train += loss

    Collect_Gradients = ZerosGradients(Layers_shape, device)
    K_tau = torch.tensor(K_tau, device=device) * 1.
    K_tau = K_tau - min(K_tau)  # 防止延时过大，导致后面计算的权重全部为0
    _, index = torch.sort(K_tau)
    normStandard = L_norm(K_Gradients[index[0]], device)
    weight = (e / 2) ** (-K_tau)

    if torch.sum(weight) == 0:
        print("延时过大。")
        for i in range(Layers_num):
            weight[index[0]] = 1.
            Collect_Gradients = K_Gradients[index[0]]
    else:
        weight = weight / torch.sum(weight)
        for i in range(args.K):
            Gradients_Sample = K_Gradients[i]
            Gradients_Sample = TensorClip(Gradients_Sample, normStandard * args.CB1, device)
            for j in range(Layers_num):
                Collect_Gradients[j] += Gradients_Sample[j] * weight[i]

    # print('weight:', weight, 'tau', K_tau)
    if itr < 800:
        Collect_Gradients = aggregation(Collect_Gradients, K_Gradients, weight, Layers_shape, args, device)
    elif itr > 100:
        Collect_Gradients = aggregation(Collect_Gradients, K_Gradients, weight, Layers_shape, args, device, Clip=True)

    lr = lr_adjust(args, torch.min(K_tau))
    for grad_idx, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-lr, Collect_Gradients[grad_idx])

    # 平均训练损失
    Loss_train /= (idx_outer + 1)
    tau_avg /= (idx_outer + 1)
    fl_time += aggregating_workers_time_max

    if itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        test_loss, test_acc = test(model, test_loader, device)
        logs['itr'].extend(workers_list)
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        logs['train_loss'].append(Loss_train.item())
        logs['staleness'].append(tau_avg)
        logs['time'].append(fl_time)
        logs['aggregation'].append(workers_list)


# 保存结果
log_title = '\n' + date + 'WKAFL Results (user_num is {}, K is {}, class_num is {}, batch_size is {}, LR is {}, itr_test is {}, total itr is {})\n'.\
    format(args.user_num, args.K, args.classes, args.batch_size, args.lr, args.itr_test, args.total_iterations)

with open('./results/MNIST_WKAFL.txt', 'a+') as fl:
    fl.write(log_title)
    fl.write('itr\n')
    fl.write(str(logs['itr']))
    fl.write('\ntime\n')
    fl.write(str(logs['time']))
    fl.write('\ntest_acc\n')
    fl.write(str(logs['test_acc']))
    fl.write('\ntrain_loss\n')
    fl.write(str(logs['train_loss']))
    fl.write('\ntest_loss\n')
    fl.write(str(logs['test_loss']))
    fl.write('\nstaleness\n')
    fl.write(str(logs['staleness']))
    fl.write('\n')
