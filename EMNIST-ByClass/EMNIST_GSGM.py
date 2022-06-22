import syft as sy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
import DatasetsLoader
import copy

hook = sy.TorchHook(torch)
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')


class Argument():
    def __init__(self):
        self.user_num = 3000  # number of total clients P
        self.K = 5  # number of participant clients K
        self.lr = 0.00001  # learning rate of global model
        self.batch_size = 16  # batch size of each client for local training
        self.test_batch_size = 128  # batch size for test datasets
        self.total_iterations = 10000  # total number of iterations
        self.itr_test = 100  # number of iterations for the two tests on test datasets
        self.itr_train = 100  # number of iterations for the two tests on training datasets
        self.seed = 1  # parameter for the server to initialize the model
        self.cuda_use = True
        self.alpha = 0.9  # parameter for momentum
        self.train_data_size = 700000
        self.test_data_size = 120000


args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
device_cpu = torch.device("cpu")
f_log = open("log_EMNIST_ByClass_EMNIST_GSGM.txt", "w")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 64, 512)
        self.fc2 = nn.Linear(512, 62)

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
    # print('start to test()')
    model.eval()
    test_loader_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data = torch.squeeze(data)
            data = data.unsqueeze(1)
            output = model(data.float())
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_loader_len
    test_acc = correct / test_loader_len
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, test_loader_len,
        100. * test_acc))
    return test_loss, test_acc


##########################定义训练过程，返回梯度########################
def train(learning_rate, model, train_data, train_target, device, optimizer, gradient=True):
    # print('start to train()')
    train_data, train_target = train_data.to(device), train_target.to(device)
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
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=args.lr)'.format(i, i))
    exec('workers.append(user{})'.format(i))  # 列表形式存储用户
    exec('taus["user{}"] = {}'.format(i, 1))  # 列表形式存储用户
    # exec('workers["user{}"] = user{}'.format(i,i))    #字典形式存储用户

optim_sever = optim.SGD(params=model.parameters(), lr=args.lr)  # 定义服务器优化器

###################################################################################
print('start to load data')
###############################数据载入############################################

dataType = 'byclass'  # 可选bymerge, byclass, digits, mnist, letters, balanced
datasets = DatasetsLoader.loadDatesets(
    trainDataSize=args.train_data_size,
    testDataSize=args.test_data_size,
    dataType=dataType
)

# 训练集，测试集
federated_data, datasNum, selectWriter = DatasetsLoader.dataset_federate_noniid(datasets, workers, args)
print('selectWritter len:', len(selectWriter))

test_data = DatasetsLoader.testImages(datasets, selectWriter)
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
logs['test_loss'].append(test_loss)

###################################################################################
print('start to fl')
#################################联邦学习过程######################################

# 获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)

# 定义训练/测试过程
worker_have_list = []

for itr in range(1, args.total_iterations + 1):

    # 按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
    # 为了对每个样本上的梯度进行裁剪，令batch_size=1，batch_num=args.batch_size*args.batchs_round，将样本逐个计算梯度
    federated_train_loader = sy.FederatedDataLoader(
        federated_data,
        worker_have_list=worker_have_list,
        worker_one_round=True,
        batch_size=args.batch_size,
        shuffle=True,
        worker_num=args.K,
        batch_num=1
    )

    workers_list = federated_train_loader.workers  # 当前回合抽取的用户列表
    worker_have_list = federated_train_loader.worker_have_list
    if len(worker_have_list) == args.user_num:
        worker_have_list = []

    # 生成与模型梯度结构相同的元素=0的列表
    Loss_train = torch.tensor(0., device=device)
    Collect_Gradients = ZerosGradients(Layers_shape, device)

    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]

        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()

        # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
        Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, device, optimizer,
                                       gradient=True)
        Loss_train += loss
        for i in range(Layers_num):
            Collect_Gradients[i] += Gradients_Sample[i] / args.K

    # 利用平均化梯度更新服务器模型
    for grad_idx, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-args.lr, Collect_Gradients[grad_idx])
    # 平均训练损失
    Loss_train /= (idx_outer + 1)

    # 同步更新不需要下面代码；异步更新需要下段代码
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for idx, (params_server, params_client) in enumerate(zip(model.parameters(), worker_model.parameters())):
            params_client.data = params_server.data
        models[workers_list[worker_idx]] = worker_model  ###添加把更新后的模型返回给用户

    if itr == 1 or itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        logs['train_loss'].append(Loss_train)

with open('./results/FEMNIST_GSGM_testacc.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, total itr is {}, itr_test is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.total_iterations, args.itr_test))
    fl.write('GSGM: ' + str(logs['test_acc']))

with open('./results/FEMNIST_GSGM_testloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, total itr is {}, itr_test is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.total_iterations, args.itr_test))
    fl.write('testloss: ' + str(logs['test_loss']))

with open('./results/FEMNIST_GSGM_trainloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, total itr is {}, itr_test is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.total_iterations, args.itr_test))
    fl.write('trainloss: ' + str(logs['train_loss']))