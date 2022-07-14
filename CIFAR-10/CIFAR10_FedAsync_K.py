import math
import numpy as np
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms  # 实现图片变换处理的包
import torchvision as tv  # 里面含有许多数据集
import copy
import time
import random
import heapq
import cifar10_dataloader

# 日志设置
logging.basicConfig(level=logging.INFO
                    , filename="./logs/CIFAR10_FedAsync_K.logs"
                    , filemode="w"
                    , format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                    , datefmt="%Y-%m-%d %H:%M:%S"
                    )
hook = sy.TorchHook(torch)
date = datetime.now().strftime('%Y-%m-%d %H:%M')


class Argument:
    def __init__(self):
        self.user_num = 500  # number of total clients P
        self.K = 20  # number of participant clients K
        self.update_num = 30
        self.lr = 0.001  # learning rate of global model
        self.batch_size = 8  # batch size of each client for local training
        self.itr_test = 100  # number of iterations for the two neighbour tests on test datasets
        self.itr_train = 100  # number of iterations for the two neighbour tests on training datasets
        self.test_batch_size = 128  # batch size for test datasets
        self.total_iterations = 10000  # total number of iterations
        self.seed = 1  # parameter for the server to initialize the model
        self.classNum = 1  # number of data classes on each client, which can determine the level of non-IID data
        self.cuda_use = True
        self.train_data_size = 50000
        self.test_data_size = 10000

        self.rou = 0.05
        self.is_proxy = True


args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
device_cpu = torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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


############################定义测试函数################################
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for imagedata, labels in test_loader:
            imagedata, labels = imagedata.to(device), labels.to(device)

            outputs = model(Variable(imagedata))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_acc = 100. * correct / total
    test_loss /= len(test_loader.dataset)
    print('10000张测试集: testacc is {:.4f}%, testloss is {}.'.format(test_acc, test_loss))
    return test_loss, test_acc


##########################定义训练过程，返回梯度########################
def train(learning_rate, train_model, train_data, train_target, device, gradient=True):
    train_model.train()
    train_model.zero_grad()
    train_targets = Variable(train_target.long())
    outputs = train_model(train_data)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, train_targets)

    if args.is_proxy:
        proximal_term = 0.0
        for w, w_t in zip(train_model.parameters(), model.parameters()):
            proximal_term += (w - w_t).norm(2)
        loss += (args.rou / 2) * proximal_term

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
    def __init__(self, avg, delta):
        self.avg = avg
        self.delta = delta

    def make(self):
        return self.avg + random.uniform(-self.delta * self.avg, self.delta * self.avg)


#  下发全局模型
def refresh_workers_model(model_server, workers_refresh):
    for client in workers_refresh:
        # model_client = models[client]
        models[client] = copy.deepcopy(model_server)
        # for idx, (params_server, params_client) in enumerate(zip(model_server.parameters(), model_client.parameters())):
        #     params_client.data = copy.deepcopy(params_server.data)
        # models[client] = model_client

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
itrs = {}   # 各个客户端当前所处轮次
time_makers = {}  # 客户端运行时间生成器
aggregating_dict = {}  # 训练中或训练结束待聚合的客户端


for i in range(1, args.user_num + 1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i, i))
    exec('models["user{}"] = copy.deepcopy(model)'.format(i))
    exec('workers.append(user{})'.format(i))
    exec('users.append("user{}")'.format(i))
    exec('itrs["user{}"] = {}'.format(i, 1))
    exec('time_makers["user{}"] = TimeMaker(random.uniform(0.1, 0.3), random.uniform(0.1, 0.2))'.format(i))


#################
#     数据载入   #
#################
print('start to load data')

# 使用torchvision加载并预处理CIFAR10数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
# 把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
trainset = tv.datasets.CIFAR10(root='data2/', train=True, download=False, transform=transform)
federated_data, dataNum = cifar10_dataloader.dataset_federate_noniid(
    trainset,
    workers,
    transform,
    args.classNum,
    args.train_data_size
)

testset = tv.datasets.CIFAR10(root='data2/', train=False, download=False, transform=transform)
testset = cifar10_dataloader.testLoader(testset, args.test_data_size)
test_loader = torch.utils.data.DataLoader(
    testset,
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

    Loss_train = torch.tensor(0., device=device)
    Collect_Gradients = ZerosGradients(Layers_shape, device)
    tau_avg = 0

    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        model_round = models[train_data.location.id]
        tau = itr - itrs[train_data.location.id] + 1

        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()

        Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, device, gradient=True)
        Loss_train += loss
        tau_avg += tau

        c = math.floor(args.user_num / args.K)
        for j in range(Layers_num):
            Collect_Gradients[j] += Gradients_Sample[j] * args.lr / (tau * c)

    # step 4
    # 对全局模型进行更新
    for grad_idx, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-1., Collect_Gradients[grad_idx])

    # 平均训练损失
    Loss_train /= (idx_outer + 1)
    tau_avg /= (idx_outer + 1)
    fl_time += aggregating_workers_time_max

    if itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        test_loss, test_acc = test(model, test_loader, device)
        logs['itr'].extend(workers_list)
        logs['test_acc'].append(test_acc.item())
        logs['test_loss'].append(test_loss)
        logs['train_loss'].append(Loss_train.item())
        logs['staleness'].append(tau_avg)
        logs['time'].append(fl_time)
        logs['aggregation'].append(workers_list)


# 保存结果
log_title = '\n' + date + 'FedAsync_K Results (user_num is {}, K is {}, class_num is {}, batch_size is {}, LR is {}, itr_test is {}, total itr is {})\n'.\
    format(args.user_num, args.K, args.classNum, args.batch_size, args.lr, args.itr_test, args.total_iterations)

with open('./results/CIFAR10_FedAsync_K.txt', 'a+') as fl:
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