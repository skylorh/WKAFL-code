import syft as sy
import torch
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import cifar10_dataloader
from datetime import datetime
from torch.autograd import Variable
import torchvision as tv  # 里面含有许多数据集
import torch
import torchvision.transforms as transforms  # 实现图片变换处理的包
from torchvision.transforms import ToPILImage
import copy

hook = sy.TorchHook(torch)
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')


class Argument():
    def __init__(self):
        self.user_num = 100  # number of total clients P
        self.K = 5  # number of participant clients K
        self.lr = 0.0001  # learning rate of global model
        self.alpha = 0.1  # parameter for momentum
        self.batch_size = 8  # batch size of each client for local training
        self.test_batch_size = 128  # batch size for test datasets
        self.total_iterations = 10000  # total number of iterations
        self.classNum = 1  # number of data classes on each client, which can determine the level of non-IID data
        self.itr_test = 100  # number of iterations for the two tests on test datasets
        self.itr_train = 100  # number of iterations for the two tests on training datasets
        self.seed = 1  # parameter for the server to initialize the model
        self.cuda_use = True
        self.train_data_size = 50000
        self.test_data_size = 10000


args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
device_cpu = torch.device("cpu")


# 定义网络
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)


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
    print('10000张测试集: testacc is {:.4f}%, testloss is {}.'.format(test_acc, test_loss))
    return test_loss, test_acc


##########################定义训练过程，返回梯度########################
def train(learning_rate, train_model, train_data, train_targets, device, optimizer):
    train_model.train()
    train_model.zero_grad()
    optimizer.zero_grad()
    train_targets = Variable(train_targets.long())
    optimizer.zero_grad()
    outputs = train_model(train_data)
    # 计算准确率
    # _, predicted = torch.max(outputs.data, 1)
    # total = labels.size(0)
    # correct = (predicted == labels).sum()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, train_targets)
    loss.backward()
    optimizer.step()

    Gradients_Tensor = []
    for params in train_model.parameters():
        Gradients_Tensor.append(params.grad.data)  # 把各层的梯度添加到张量Gradients_Tensor
    return Gradients_Tensor, loss


##################################计算Non-IID程度##################################
def JaDis(datasNum, userNum):
    sim = []
    for i in range(userNum):
        data1 = datasNum[i]
        for j in range(i + 1, userNum):
            data2 = datasNum[j]
            sameNum = [min(x, y) for x, y in zip(data1, data2)]
            sim.append(sum(sameNum) / (sum(data1) + sum(data2) - sum(sameNum)))
    distance = 2 * sum(sim) / (userNum * (userNum - 1))
    return distance


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
    # exec('workers["user{}"] = user{}'.format(i,i))    # 字典形式存储用户

optim_sever = optim.SGD(params=model.parameters(), lr=args.lr)  # 定义服务器优化器

###################################################################################
print('start to load data')
###############################数据载入############################################

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

# Di
di_list = {}
D = 0
for i in range(args.user_num):
    useri = "user{}".format(i + 1)
    di = len(federated_data.datasets[useri])
    di_list[useri] = di
    D += di
print('di_list: {}'.format(di_list))

testset = tv.datasets.CIFAR10('data2/', train=False, download=False)
testset = cifar10_dataloader.testLoader(testset, args.test_data_size)
test_loader = torch.utils.data.DataLoader(
    testset,
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

# 定义训练/测试过程
fl_time = 0

for itr in range(1, args.total_iterations + 1):


    # 按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
    # 为了对每个样本上的梯度进行裁剪，令batch_size=1，batch_num=args.batch_size*args.batchs_round，将样本逐个计算梯度
    federated_train_loader = sy.FederatedDataLoader(
        federated_data,
        batch_size=args.batch_size,
        shuffle=True,
        worker_num=args.K,
        batch_num=1
    )

    # workers_list ['user21', 'user96', 'user1', 'user75', 'user20', 'user92', 'user55', 'user54', 'user46', 'user70']
    workers_list = federated_train_loader.workers  # 当前回合抽取的用户列表

    # 生成与模型梯度结构相同的元素=0的列表
    Loss_train = torch.tensor(0., device=device)

    # 客户端使用新模型训练
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for idx, (params_server, params_client) in enumerate(zip(model.parameters(), worker_model.parameters())):
            params_client.data = copy.deepcopy(params_server.data)
        models[workers_list[worker_idx]] = worker_model

    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]

        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()

        # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
        Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, device, optimizer)
        Loss_train += loss

    # 更新服务器模型参数
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

    if itr == 1 or itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        logs['train_loss'].append(Loss_train.item())
        logs['time'].append(fl_time)

with open('./results/cifar10_FedAvg_testacc.txt', 'a+') as fl:
    fl.write(
        '\n' + date + 'FedAvg: test_acc Results (UN is {}, K is {}, classnum is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
        format(args.user_num, args.K, args.classNum, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write(str(logs['test_acc']))

with open('./results/cifar10_FedAvg_trainloss.txt', 'a+') as fl:
    fl.write(
        '\n' + date + 'FedAvg: train_loss Results (UN is {}, K is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
        format(args.user_num, args.K, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write(str(logs['train_loss']))

with open('./results/cifar10_FedAvg_testloss.txt', 'a+') as fl:
    fl.write(
        '\n' + date + 'FedAvg: test_loss Results (UN is {}, K is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
        format(args.user_num, args.K, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write(str(logs['test_loss']))

with open('./results/cifar10_FedAvg_time.txt', 'a+') as fl:
    fl.write(
        '\n' + date + 'FedAvg: time Results (UN is {}, K is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
        format(args.user_num, args.K, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write(str(logs['time']))