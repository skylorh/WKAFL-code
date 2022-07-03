import numpy as np
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import rawDatasetsLoader
from datetime import datetime
import copy
import time

hook = sy.TorchHook(torch)
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')


class Argument():
    def __init__(self):
        self.user_num = 100  # number of total clients P
        self.K = 10  # number of participant clients K
        self.CB1 = 100  # clip parameter in both stages
        self.CB2 = 10  # clip parameter B at stage two
        self.lr = 0.005  # learning rate of global model
        self.itr_test = 10  # number of iterations for the two neighbour tests on test datasets
        self.batch_size = 4  # batch size of each client for local training
        self.test_batch_size = 128  # batch size for test datasets
        self.total_iterations = 1000  # total number of iterations
        self.threshold = 0.3  # threshold to judge whether gradients are consistent
        self.alpha = 0.1  # parameter for momentum to alleviate the effect of non-IID data
        self.classes = 1  # number of data classes on each client, which can determine the level of non-IID data
        self.seed = 1  # parameter for the server to initialize the model
        self.cuda_use = True
        self.train_data_size = 100000
        self.test_data_size = 10000


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


############################ 定义测试函数 ################################
def test(model, test_loader, device):
    model.eval()
    test_loader_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data = data.unsqueeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_loader_len
    test_acc = correct / test_loader_len

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_loader_len,
        100. * test_acc))
    return test_loss, test_acc


##########################定义训练过程，返回梯度########################
def train(learning_rate, model, train_data, train_target, device, optimizer, gradient=True):
    model.train()
    model.zero_grad()
    train_data = train_data.unsqueeze(1)
    output = model(train_data.float())
    loss = F.nll_loss(output, train_target.long())
    loss.backward()
    Gradients_Tensor = []
    if gradient == False:
        for params in model.parameters():
            Gradients_Tensor.append(-learning_rate * params.grad.data)  # 返回-lr*grad
    if gradient == True:
        for params in model.parameters():
            Gradients_Tensor.append(params.grad.data)  # 把各层的梯度添加到张量Gradients_Tensor
    return Gradients_Tensor, loss


###################################################################################
print('start to gen model and workers')
##################################模型和用户生成###################################
model = Net()
model.cuda()
workers = []
models = {}
optims = {}
taus = {}
for i in range(1, args.user_num + 1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i, i))
    exec('models["user{}"] = copy.deepcopy(model)'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=copy.deepcopy(args.lr))'.format(i, i))
    exec('workers.append(user{})'.format(i))  # 列表形式存储用户
    exec('taus["user{}"] = {}'.format(i, 1))  # 列表形式存储用户
    # exec('workers["user{}"] = user{}'.format(i,i))    #字典形式存储用户

optim_sever = optim.SGD(params=model.parameters(), lr=args.lr)  # 定义服务器优化器

###################################################################################
print('start to load data')
###############################数据载入############################################
dataType = 'mnist'  # 可选bymerge, byclass, digits, mnist, letters, balanced
datasets = rawDatasetsLoader.loadDatesets(
    trainDataSize=args.train_data_size,
    testDataSize=args.test_data_size,
    dataType=dataType
)

# 训练集，测试集, datasNum为列表，datasNum[i]表示第i个学习者的信息，为字典，['3']=45表示图片三有45张
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
logs = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'staleness': [], 'time': []}

###################################################################################
print('start to run fl')
#################################联邦学习过程######################################
# 获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
e = torch.exp(torch.tensor(1., device=device))
# 定义训练/测试过程

fl_time = 0
for itr in range(1, args.total_iterations + 1):
    itr_start_time = time.time()
    # 按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
    # 为了对每个样本上的梯度进行裁剪，令batch_size=1，batch_num=args.batch_size*args.batchs_round，将样本逐个计算梯度
    federated_train_loader = sy.FederatedDataLoader(
        federated_data,
        batch_size=args.batch_size,
        shuffle=True,
        worker_num=args.K,
        batch_num=1
    )

    workers_list = federated_train_loader.workers  # 当前回合抽取的用户列表

    # 生成与模型梯度结构相同的元素=0的列表
    K_Gradients = []
    Loss_train = torch.tensor(0., device=device)
    tau_avg = 0
    weight = []
    K_tau = []

    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        K_tau.append(taus[train_data.location.id])  # 添加延时信息
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        tau_avg += taus[train_data.location.id] - 1

        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()

        # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
        Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, device, optimizer,
                                       gradient=True)
        if itr > 1:
            for j in range(Layers_num):
                Gradients_Sample[j] = Gradients_Sample[j] + args.alpha * Collect_Gradients[j]
        K_Gradients.append(TensorClip(Gradients_Sample, args.CB1, device))
        Loss_train += loss

    Collect_Gradients = ZerosGradients(Layers_shape, device)
    K_tau = torch.tensor(K_tau, device=device) * 1.
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

    # 升级延时信息
    for tau in taus:
        taus[tau] = taus[tau] + 1
    for worker in workers_list:
        taus[worker] = 1

    lr = lr_adjust(args, torch.min(K_tau))
    for grad_idx, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-lr, Collect_Gradients[grad_idx])

    # 同步更新不需要下面代码；异步更新需要下段代码
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for idx, (params_server, params_client) in enumerate(zip(model.parameters(), worker_model.parameters())):
            params_client.data = params_server.data
        models[workers_list[worker_idx]] = worker_model  # 添加把更新后的模型返回给用户

    # 平均训练损失
    Loss_train /= (idx_outer + 1)
    tau_avg /= (idx_outer + 1)
    itr_end_time = time.time()
    fl_time += itr_end_time - itr_start_time

    if itr == 1 or itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        logs['train_loss'].append(Loss_train.item())
        logs['staleness'].append(tau_avg)
        logs['time'].append(fl_time)

with open('./results/MNIST_WKAFL_testacc.txt', 'a+') as fl:
    fl.write(
        '\n' + date + 'WKAFL test_acc Results (user_num is {}, K is {}, CB1 is {}, CB2 is {}, sim_threshold is {},'.
        format(args.user_num, args.K, args.CB1, args.CB2, args.threshold))
    fl.write(' BZ is {}, LR is {}, itr_test is {}, total itr is {}， classesis{})\n'.
             format(args.batch_size, args.lr, args.itr_test, args.total_iterations, args.classes))
    fl.write(str(logs['test_acc']))

with open('./results/MNIST_WKAFL_trainloss.txt', 'a+') as fl:
    fl.write(
        '\n' + date + 'WKAFL train_loss Results (user_num is {}, K is {}, CB1 is {}, CB2 is {}, sim_threshold is {},'.
        format(args.user_num, args.K, args.CB1, args.CB2, args.threshold))
    fl.write(' BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format(args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write(str(logs['train_loss']))

with open('./results/MNIST_WKAFL_testloss.txt', 'a+') as fl:
    fl.write(
        '\n' + date + 'WKAFL test_loss Results (user_num is {}, K is {}, CB1 is {}, CB2 is {}, sim_threshold is {},'.
        format(args.user_num, args.K, args.CB1, args.CB2, args.threshold))
    fl.write(' BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format(args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write(str(logs['test_loss']))

with open('./results/MNIST_WKAFL_staleness.txt', 'a+') as fl:
    fl.write(
        '\n' + date + 'WKAFL staleness Results (user_num is {}, K is {}, CB1 is {}, CB2 is {}, sim_threshold is {},'.
        format(args.user_num, args.K, args.CB1, args.CB2, args.threshold))
    fl.write(' BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format(args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write(str(logs['staleness']))

with open('./results/MNIST_WKAFL_time.txt', 'a+') as fl:
    fl.write('\n' + date + 'WKAFL time Results (user_num is {}, K is {}, CB1 is {}, CB2 is {}, sim_threshold is {},'.
             format(args.user_num, args.K, args.CB1, args.CB2, args.threshold))
    fl.write(' BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format(args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write(str(logs['time']))
