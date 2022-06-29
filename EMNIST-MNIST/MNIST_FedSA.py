import math
import time
from thop import profile
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import rawDatasetsLoader
from datetime import datetime
import copy

hook = sy.TorchHook(torch)
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')


class Argument():
    def __init__(self):
        self.user_num = 10  # number of total clients P
        self.K = 5  # number of participant clients K
        self.lr = 0.0005  # learning rate of global model 0.0005
        self.batch_size = 4  # batch size of each client for local training
        self.itr_test = 50  # number of iterations for the two neighbour tests on test datasets
        self.test_batch_size = 128  # batch size for test datasets
        self.total_iterations = 5000  # total number of iterations
        self.seed = 1  # parameter for the server to initialize the model
        self.classes = 5  # number of data classes on each client, which can determine the level of non-IID data
        self.cuda_use = True
        self.train_data_size = 100000
        self.test_data_size = 10000
        self.tau0 = 3
        self.K_star = 280
        self.phi = 0.05
        self.B = 582026 * 300  # 通信预算

args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
device_cpu = torch.device("cpu")
f_log = open("log_MNIST_GSGM.txt", "w")


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
        ZeroGradient.append(torch.zeros(Layers_shape[i], device = device))
    return ZeroGradient

################################归一化################################
def data_normal(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
    d_min = orign_data.min()
    d_max = orign_data.max()
    dst = d_max - d_min
    norm_data = torch.div((orign_data - d_min), dst)
    return norm_data

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
            # torchvision.transforms.functional.normalize(data, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # data = nn.LayerNorm(data.size()[1:])
            # data = data_normal(data)
            f_log.write('data: {}\ntarget: {}\n'.format(data, target))
            data = torch.squeeze(data)
            data = data.unsqueeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            f_log.write('output: {}\n'.format(output))
            cur_loss = F.nll_loss(output, target.long(), reduction='sum').item() # sum up batch loss
            # cur_loss = F.cross_entropy(output, target.long())  # sum up batch loss

            # print('cur_loss: ', cur_loss)
            f_log.write('cur_loss: {}\n'.format(cur_loss))
            test_loss += cur_loss
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print('test_loss: ', test_loss)
    f_log.write('test_loss: {}\n'.format(test_loss))
    test_loss /= test_loader_len
    # print('test_loss: ', test_loss)
    f_log.write('test_loss: {}\n'.format(test_loss))

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
    loss_func = F.nll_loss
    # print('output: ', output)
    loss = loss_func(output, train_target.long())
    # print('loss: ', loss)
    loss.backward()
    # print('loss after backward: ', loss)

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
queue = {}
lrs = {}
pi = {}
_pi = {}
Qi = {}

counti = {}      # 记录各个客户端的参与次数
countsum = 0     # 记录总的参与次数

input = torch.randn(1, 1, 28, 28, device=device)
_, model_params_size = profile(model, inputs=(input, ))
# 582026.0
# print('model_params_size: ', model_params_size)

for i in range(1, args.user_num + 1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i, i))
    exec('models["user{}"] = copy.deepcopy(model)'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=copy.deepcopy(args.lr))'.format(i, i))
    exec('workers.append(user{})'.format(i))  # 列表形式存储用户
    exec('taus["user{}"] = {}'.format(i, 1))  # 列表形式存储用户
    exec('queue["user{}"] = []'.format(i))  # 列表形式存储用户queue
    exec('lrs["user{}"] = {}'.format(i, args.lr))  # 列表形式存储用户学习率
    exec('pi["user{}"] = {}'.format(i, 0.))
    exec('_pi["user{}"] = {}'.format(i, 0.))
    exec('Qi["user{}"] = []'.format(i))
    exec('counti["user{}"] = {}'.format(i, 0))



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

# 训练集，测试集
federated_data, datasNum = rawDatasetsLoader.dataset_federate_noniid(datasets, workers, args)

# Di
di_list = {}
D = 0
for i in range(args.user_num):
    useri = "user{}".format(i+1)
    di = len(federated_data.datasets[useri])
    di_list[useri] = di
    D += di
    # print('len Di: ', di)
# print('len D: ', D)
print('di_list: {}'.format(di_list))


# Jaccard = JaDis(datasNum, args.user_num)
# print('Jaccard distance is {}'.format(Jaccard))
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
logs = {'train_loss': [], 'test_loss': [], 'test_acc': []}
test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
logs['test_acc'].append(test_acc)



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
        # psi = 1 - 2 * alpha * args.lr * _beta * (mu - _n * L * L)
        psi = 0.93
        # print('alpha: {}, _beta: {}, _n: {}, psi: {}'.format(alpha, _beta, _n, psi))
        K = (1 + taumax) * math.log(args.phi, psi)
        # print('K: ', K)
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





###################################################################################
print('start to run fl')
#################################联邦学习过程######################################
# 获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)

# 定义训练/测试过程

M = args.K

# 定义训练/测试过程
for itr in range(1, args.total_iterations + 1):

    start_time = time.time()

    # step 1
    # 按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
    # 为了对每个样本上的梯度进行裁剪，令batch_size=1，batch_num=args.batch_size*args.batchs_round，将样本逐个计算梯度
    federated_train_loader = sy.FederatedDataLoader(
        federated_data,
        batch_size=args.batch_size,
        shuffle=True,
        worker_num=M,
        batch_num=1
    )

    # workers_list ['user21', 'user96', 'user1', 'user75', 'user20', 'user92', 'user55', 'user54', 'user46', 'user70']
    workers_list = federated_train_loader.workers  # 当前回合抽取的用户列表
    # print('workers_list: {}'.format(workers_list))

    # step 2
    # 生成与模型梯度结构相同的元素=0的列表
    Loss_train = torch.tensor(0., device=device)

    # step 3
    # 对workers_list中选取的worker进行训练
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        # 当前worker的model和optimizer
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        model_lr = lrs[train_data.location.id]

        # tensor([[[]]], device='cuda:0') tensor([5., 5., 5., 5.], device='cuda:0')
        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()

        # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
        Gradients_Sample, loss = train(model_lr, model_round, train_data, train_targets, device, optimizer)
        # print('loss rtn: ', loss)

        Loss_train += loss

        # client本地更新模型
        for grad_idx, params_client in enumerate(model_round.parameters()):
            params_client.data.add_(-model_lr, Gradients_Sample[grad_idx])


    # step 4
    # 对全局模型进行更新
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        # 当前worker的model和optimizer
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        model_lr = lrs[train_data.location.id]

        Di_Div_D = di_list[train_data.location.id] / D
        # print('Di_Div_D: ', Di_Div_D)

        for grad_idx, (params_server, params_client) in enumerate(zip(model.parameters(), model_round.parameters())):
            params_server.data.add_(-Di_Div_D, params_server.data)
            params_server.data.add_(Di_Div_D, params_client.data)

    # step 5
    # 升级延时信息和参与次数信息
    for tau in taus:
        taus[tau] = taus[tau] + 1
    for worker in workers_list:
        taus[worker] = 1
        counti[worker] += 1
    countsum += len(workers_list)

    # step 6
    # 模型分发更新
    for worker_idx in range(args.user_num):
        useri = "user{}".format(worker_idx+1)
        if useri in workers_list or taus[useri] > args.tau0:

            # 模型更新
            worker_model = models[useri]
            for idx, (params_server, params_client) in enumerate(zip(model.parameters(), worker_model.parameters())):
                params_client.data = copy.deepcopy(params_server.data)
            models[useri] = worker_model  # 添加把更新后的模型返回给用户

            # 学习率更新
            if counti[useri] != 0:
                lrs[useri] = (args.lr * countsum) / (args.user_num * counti[useri])

    end_time = time.time()

    # step 7
    # logging thread
    for worker_idx in range(args.user_num):
        useri = "user{}".format(worker_idx+1)
        _pi[useri] += end_time - start_time
        if useri in workers_list:
            Qi[useri].append(_pi[useri])
            pi[useri] = sum(Qi[useri]) / len(Qi[useri])
            _pi[useri] = 0.
            # print('useri: {} pi[useri]: {} '.format(useri, pi[useri]))

    # step 8
    # 计算下一轮的参与客户端个数并更新学习率
    M = Algorithm2()



    # 平均训练损失
    Loss_train /= (idx_outer + 1)
    # print('Loss_train: ', Loss_train)

    if itr == 1 or itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        print('lrs: {}'.format([float('{:.7f}'.format(i)) for i in list(lrs.values())]))
        print('pi: {}'.format(pi))

        test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        # for grad_idx, params_sever in enumerate(model.parameters()):
        #     print(params_sever.data)

    if itr == 1 or itr % args.itr_test == 0:
        # 平均训练损失
        Loss_train /= (idx_outer + 1)
        logs['train_loss'].append(Loss_train)

with open('./results/MNIST_FedSA_testacc.txt', 'a+') as fl:
    fl.write(
        '\n' + date + ' Results (UN is {}, K is {}, classnum is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
        format(args.user_num, args.K, args.classes, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write('GSGM: ' + str(logs['test_acc']))

with open('./results/MNIST_FedSA_trainloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write('train_loss: ' + str(logs['train_loss']))

with open('./results/MNIST_FedSA_testloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write('test_loss: ' + str(logs['test_loss']))



