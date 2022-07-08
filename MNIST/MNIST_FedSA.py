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
                    , filename="./logs/MNIST_FedSA.logs"
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
        self.K = 5  # number of participant clients K
        self.update_num = 10  # 每轮更新的客户端数目
        self.lr = 0.005  # learning rate of global model
        self.batch_size = 4  # batch size of each client for local training
        self.itr_test = 10  # number of iterations for the two neighbour tests on test datasets
        self.itr_train = 100  # number of iterations for the two neighbour tests on training datasets
        self.test_batch_size = 128  # batch size for test datasets
        self.total_iterations = 1000  # total number of iterations
        self.seed = 1  # parameter for the server to initialize the model
        self.classes = 1  # number of data classes on each client, which can determine the level of non-IID data
        self.cuda_use = True
        self.train_data_size = 100000
        self.test_data_size = 10000

        self.tau0 = 3
        self.K_star = 280
        self.phi = 0.05
        self.B = 582026 * 1200  # 通信预算


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
    optimizer = optim.SGD(train_model.parameters(), lr=learning_rate)
    train_model.train()
    optimizer.zero_grad()

    train_data = train_data.unsqueeze(1)
    output = train_model(train_data.float())
    loss_func = F.nll_loss
    loss = loss_func(output, train_target.long())
    loss.backward()
    optimizer.step()

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
itrs = {}   # 各个客户端当前所处轮次
time_makers = {}  # 客户端运行时间生成器
aggregating_dict = {}  # 训练中或训练结束待聚合的客户端

lrs = {}
pi = {}
_pi = {}
Qi = {}

counti = {}      # 记录各个客户端的参与次数
countsum = 0     # 记录总的参与次数

# from thop import profile
# input = torch.randn(1, 1, 28, 28, device=device)
# _, model_params_size = profile(model, inputs=(input, ))
model_params_size = 582026.0
# print('model_params_size: ', model_params_size)

for i in range(1, args.user_num + 1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i, i))
    exec('models["user{}"] = copy.deepcopy(model)'.format(i))
    exec('workers.append(user{})'.format(i))
    exec('users.append("user{}")'.format(i))
    exec('itrs["user{}"] = {}'.format(i, 1))
    exec('lrs["user{}"] = {}'.format(i, args.lr))  # 列表形式存储用户学习率
    exec('pi["user{}"] = {}'.format(i, 0.))
    exec('_pi["user{}"] = {}'.format(i, 0.))
    exec('Qi["user{}"] = []'.format(i))
    exec('counti["user{}"] = {}'.format(i, 0))
    exec('time_makers["user{}"] = TimeMaker(random.uniform(0.1, 5), random.uniform(0.1, 0.2))'.format(i))

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

# 统计各个客户端的数据数目
di_list = {}
D = 0
for i in range(args.user_num):
    useri = "user{}".format(i+1)
    di = len(federated_data.datasets[useri])
    di_list[useri] = di
    D += di
print('di_list: {}'.format(di_list))

# 定义记录字典
logs = {'aggregation': [], 'itr': [], 'train_loss': [], 'test_loss': [], 'test_acc': [], 'staleness': [], 'time': []}


def Algorithm2():
    Tmin = float('inf')
    _M = 1
    for m in range(1, args.user_num + 1):
        # print('m: ', m)
        ri = {}
        taui = {}
        ci = {}
        fi = {}
        taumax = 0
        pi_list = list(pi.values())
        ri_list = sorted(pi_list)
        _Vm = D
        ni = []

        for vi in range(1, args.user_num + 1):
            useri = "user{}".format(vi)
            ri[useri] = pi[useri]
            taui[useri] = 0
            ci[useri] = 0

        for ki in range(1, args.K_star + 1):
            tk = ri_list[m-1]
            _t = sum(ri_list[0:ki]) / ki
            Vk = []
            Di_sum = 0
            for vi in range(1, args.user_num + 1):
                useri = "user{}".format(vi)
                if ri[useri] <= tk:
                    Vk.append(useri)
                    Di_sum += di_list[useri]

            if Di_sum < _Vm:
                _Vm = Di_sum


            for vi in range(1, args.user_num + 1):
                useri = "user{}".format(vi)
                taui[useri] += 1
                if useri in Vk or taui[useri] > args.tau0:
                    ri[useri] = pi[useri]
                    taui[useri] = 0
                    ci[useri] += 1
                else:
                    ri[useri] -= tk
                    taumax = max(taumax, taui[useri])

        c_total = sum(list(ci.values()))
        # print(list(ci.values()))

        for vi in range(1, args.user_num + 1):
            useri = "user{}".format(vi)
            if ci[useri] == 0:
                continue
            fi[useri] = ci[useri] / c_total
            # print('fi[{}]: {} '.format(useri, fi[useri]))
            ni.append(args.lr / (args.user_num * fi[useri]))



        _n = max(ni)
        # print('_n: {}'.format(_n))
        _beta = _Vm / D
        alpha = m / args.user_num
        mu = 95.1
        L = 0.1
        psi = 1 - 2 * alpha * args.lr * _beta * (mu - _n * L * L)
        # psi = 0.93
        # print('alpha: {}, _beta: {}, _n: {}, psi: {}'.format(alpha, _beta, _n, psi))
        K = (1 + taumax) * math.log(args.phi, psi)
        # print('K: {}, args.B / (m * model_params_size): {}'.format(K, args.B / (m * model_params_size)))
        if K > args.B / (m * model_params_size):
            continue


        T = K * _t

        # print('m: {}, T: {}'.format(m, T))

        # print('m: {}, T: {} < Tmin:{} = {}'.format(m, T, Tmin, T < Tmin))
        if T < Tmin:
            Tmin = T
            _M = m

    if _M != 1:
        print('M: ', _M)
    return _M


#################
# 开始联邦学习过程 #
#################
print('start to run fl')

# 获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)

M = args.K

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
        model_lr = lrs[train_data.location.id]
        tau = itr - itrs[train_data.location.id] + 1

        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()

        Gradients_Sample, loss = train(model_lr, model_round, train_data, train_targets, device, gradient=True)
        Loss_train += loss
        tau_avg += tau

    # step 4
    # 对全局模型进行更新
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        # 当前worker的model
        model_round = models[train_data.location.id]

        Di_Div_D = di_list[train_data.location.id] / D

        for grad_idx, (params_server, params_client) in enumerate(zip(model.parameters(), model_round.parameters())):
            params_server.data.add_(-Di_Div_D, params_server.data)
            params_server.data.add_(Di_Div_D, params_client.data)

    # step 5
    # 下发最新模型并更新学习率
    for worker in workers_list:
        counti[worker] += 1
    countsum += len(workers_list)

    users_selectable = set(workers_list)
    for worker in set(users):
        if itr - itrs[worker] + 1 > args.tau0:
            users_selectable.add(worker)
    workers_refresh = list(users_selectable)
    refresh_workers_model(model, workers_refresh)
    refresh_workers_itr(itr, workers_refresh)

    for worker in workers_refresh:
        aggregating_dict[worker] = time_makers[worker].make()
        # 学习率更新
        if counti[worker] != 0:
            lrs[worker] = (args.lr * countsum) / (args.user_num * counti[worker])

    # step 6
    # logging thread
    for useri in users:
        if useri in workers_list:
            Qi[useri].append(_pi[useri])
            pi[useri] = sum(Qi[useri]) / len(Qi[useri])
            _pi[useri] = 0.

    # step 7
    # 计算下一轮的参与客户端个数并更新学习率
    # M = Algorithm2()
    M = args.K

    # 平均训练损失
    Loss_train /= (idx_outer + 1)
    tau_avg /= (idx_outer + 1)
    fl_time += aggregating_workers_time_max

    if itr == 1 or itr % args.itr_test == 0:
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
log_title = '\n' + date + 'FedSA Results (user_num is {}, K is {}, class_num is {}, batch_size is {}, LR is {}, itr_test is {}, total itr is {})\n'.\
    format(args.user_num, args.K, args.classes, args.batch_size, args.lr, args.itr_test, args.total_iterations)

with open('./results/MNIST_FedSA.txt', 'a+') as fl:
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