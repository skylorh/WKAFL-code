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
import torchvision.transforms as transforms  # 实现图片变换处理的包
import torchvision as tv  # 里面含有许多数据集
import torchvision
import copy
import time
import random
import heapq
import cifar10_dataloader

# 日志设置
logging.basicConfig(level=logging.INFO
                    , filename="./logs/CIFAR10_FedAvg.logs"
                    , filemode="w"
                    , format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                    , datefmt="%Y-%m-%d %H:%M:%S"
                    )
hook = sy.TorchHook(torch)
date = datetime.now().strftime('%Y-%m-%d %H:%M')


class Argument:
    def __init__(self):
        self.user_num = 100  # number of total clients P
        self.K = 10  # number of participant clients K
        self.lr = 0.05  # learning rate of global model 0.005
        self.batch_size = 32  # batch size of each client for local training
        self.itr_test = 100  # number of iterations for the two neighbour tests on test datasets
        self.itr_train = 100  # number of iterations for the two neighbour tests on training datasets
        self.test_batch_size = 128  # batch size for test datasets
        self.total_iterations = 10000  # total number of iterations
        self.seed = 1  # parameter for the server to initialize the model
        self.classNum = 10  # number of data classes on each client, which can determine the level of non-IID data
        self.cuda_use = True
        self.train_data_size = 50000
        self.test_data_size = 10000


args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
device_cpu = torch.device("cpu")


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(x.size()[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        # x = self.bn1(F.relu(self.conv1(x)))
        # x = self.bn1(F.relu(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout10(x)
        # x = self.bn2(F.relu(self.conv3(x)))
        # x = self.bn2(F.relu(self.conv4(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.avgpool(x)
        x = self.dropout10(x)
        # x = self.bn3(F.relu(self.conv5(x)))
        # x = self.bn3(F.relu(self.conv6(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def call_bn(bn, x):
    return bn(x)


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        # self.bn1=nn.BatchNorm2d(128)
        # self.bn2=nn.BatchNorm2d(128)
        # self.bn3=nn.BatchNorm2d(128)
        # self.bn4=nn.BatchNorm2d(256)
        # self.bn5=nn.BatchNorm2d(256)
        # self.bn6=nn.BatchNorm2d(256)
        # self.bn7=nn.BatchNorm2d(512)
        # self.bn8=nn.BatchNorm2d(256)
        # self.bn9=nn.BatchNorm2d(128)

    def forward(self, x, ):
        h = x
        h = self.c1(h)
        # h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.c2(h)
        # h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.c3(h)
        # h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        # h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.c5(h)
        # h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.c6(h)
        # h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        # h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.c8(h)
        # h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.c9(h)
        # h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.l_c1(h)
        # if self.top_bn:
        #     logit=call_bn(self.bn_c1, logit)
        return logit


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
    test_loss /= total
    print('10000张测试集: testacc is {:.4f}%, testloss is {}.'.format(test_acc, test_loss))
    return test_loss, test_acc


##########################定义训练过程，返回梯度########################
def train(learning_rate, train_model, train_data, train_target, device, gradient=True):
    optimizer = optim.SGD(train_model.parameters(), lr=learning_rate)
    train_model.train()
    train_targets = Variable(train_target.long())
    optimizer.zero_grad()
    outputs = train_model(train_data)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, train_targets)

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
        models[client] = copy.deepcopy(model_server)
        # model_client = models[client]
        # for idx, (params_server, params_client) in enumerate(zip(model_server.parameters(), model_client.parameters())):
        #     params_client.data = copy.deepcopy(params_server.data)
        # models[client] = model_client


#  更新客户端模型轮次
def refresh_workers_itr(itr, workers_refresh):
    for client in workers_refresh:
        itrs[client] = itr


#################
#  服务端和客户端  #
#################
print('start to gen model and workers')

model = Net()  # 全局模型
model.cuda()
workers = []
users = []
models = {}  # 模型
itrs = {}  # 各个客户端当前所处轮次
time_makers = {}  # 客户端运行时间生成器
aggregating_dict = {}  # 训练中或训练结束待聚合的客户端

for i in range(1, args.user_num + 1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i, i))
    exec('models["user{}"] = copy.deepcopy(model)'.format(i))
    exec('workers.append(user{})'.format(i))
    exec('users.append("user{}")'.format(i))
    exec('itrs["user{}"] = {}'.format(i, 1))
    exec('time_makers["user{}"] = TimeMaker(random.uniform(0.1, 5), random.uniform(0.1, 0.2))'.format(i))

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

testset = tv.datasets.CIFAR10('data2/', train=False, download=False, transform=transform)
testset = cifar10_dataloader.testLoader(testset, args.test_data_size)
test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.test_batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

# 统计各个客户端的数据数目
di_list = {}
D = 0
for i in range(args.user_num):
    useri = "user{}".format(i + 1)
    di = len(federated_data.datasets[useri])
    di_list[useri] = di
    D += di
print('di_list: {}'.format(di_list))

# 定义记录字典
logs = {'aggregation': [], 'itr': [], 'train_loss': [], 'test_loss': [], 'test_acc': [], 'staleness': [], 'time': []}

#################
# 开始联邦学习过程 #
#################
print('start to run fl')

fl_time = 0
for itr in range(1, args.total_iterations + 1):

    # step 1
    # 随机选取客户端下发最新模型进行训练
    workers_refresh = random.sample(users, args.K)
    refresh_workers_model(model, workers_refresh)
    refresh_workers_itr(itr, workers_refresh)

    for worker in workers_refresh:
        aggregating_dict[worker] = time_makers[worker].make()
    aggregating_workers_time_max = max(aggregating_dict.values())
    aggregating_dict.clear()

    # step 2
    # 聚合时执行实际的train 未selected_workers则按worker_num随机选择
    federated_train_loader = sy.FederatedDataLoader(
        federated_data,
        batch_size=args.batch_size,
        shuffle=True,
        worker_num=args.K,
        batch_num=1,
        selected_workers=workers_refresh
    )

    workers_list = federated_train_loader.workers  # 当前回合抽取的用户列表

    Loss_train = torch.tensor(0., device=device)

    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        model_round = models[train_data.location.id]
        tau = itrs[train_data.location.id]

        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()

        Gradients_Sample, loss = train(copy.deepcopy(args.lr), model_round, train_data, train_targets, device,
                                       gradient=True)
        Loss_train += loss

    # step 4
    # 对全局模型进行更新
    sum_ni = 0
    for worker_idx in range(len(workers_list)):
        sum_ni += di_list[workers_list[worker_idx]]

    for grad_idx, params_server in enumerate(model.parameters()):
        params_server.data.zero_()

    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        ni = di_list[workers_list[worker_idx]]
        for idx, (params_server, params_client) in enumerate(zip(model.parameters(), worker_model.parameters())):
            params_server.data.add_(ni / sum_ni, params_client.data)

    # 平均训练损失
    Loss_train /= (idx_outer + 1)
    fl_time += aggregating_workers_time_max

    if itr == 1 or itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        print('Loss_train: ', Loss_train.item())
        test_loss, test_acc = test(model, test_loader, device)
        logs['itr'].extend(workers_list)
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        logs['train_loss'].append(Loss_train.item())
        logs['time'].append(fl_time)
        logs['aggregation'].append(workers_list)

# 保存结果
log_title = '\n' + date + 'FedAvg Results (user_num is {}, K is {}, class_num is {}, batch_size is {}, LR is {}, itr_test is {}, total itr is {})\n'. \
    format(args.user_num, args.K, args.classes, args.batch_size, args.lr, args.itr_test, args.total_iterations)

with open('./results/CIFAR10_FedAvg.txt', 'a+') as fl:
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
    fl.write('\n')
