import torch
import torch.nn as nn
import DatasetsLoader


# 返回Dict 获取每类label对应的index
def labelDictToIndexList(labels):
    dict = {}
    for i in range(len(labels)):
        if labels[i] not in dict:
            dict[labels[i]] = []
        dict[labels[i]].append(i)
    return dict


def genWriters(DataSize, batch_size, workers, labelClassNum, isTrain, dataType='byclass'):
    if isTrain:
        trainOrTest = 'train'
    else:
        trainOrTest = 'test'

    # 要写入writer中的数据初始值为-1, 在range遍历中==i判定为false
    writers = torch.Tensor(DataSize)
    nn.init.constant_(writers, -1)

    labels_file = r'../data/EMNIST-ByClass/' + dataType + '/emnist-' + dataType + '-' + trainOrTest + '-labels-idx1-ubyte'
    labels = DatasetsLoader.decode_idx1_ubyte(labels_file, DataSize)

    # 每个label对应的images的idx
    labels_index = labelDictToIndexList(labels.tolist())

    for i in range(workers):

        if labelClassNum > 1:
            while True:
                # 从0~61类中随机选取几类数据 tensor([2, 9])
                labelClass = torch.randperm(62)[0:labelClassNum]

                # 每类数据随机分配一定百分比例 tensor([0.8884, 0.9903]) -> tensor([0.4729, 0.5271])
                dataRate = torch.rand([labelClassNum])
                dataRate = dataRate / torch.sum(dataRate)

                # batch_size ~ batch_size 间的随机数 tensor(16) -> tensor([8., 8.])
                dataNum = batch_size
                dataNum = torch.ceil(dataNum * dataRate)

                continueFlag = False
                for j in range(labelClassNum):
                    # 某一类数量不够，重新选取数据
                    if len(labels_index[int(labelClass[j])]) < int(dataNum[j]):
                        continueFlag = True
                if continueFlag:
                    continue
                break

            # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            datasnum = torch.zeros([62])
            # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            datasnum[labelClass.tolist()] = dataNum

            for j in range(labelClassNum):
                # 某类数据的数量
                datanum = int(dataNum[j].item())
                # 选取该类数据内的datanum个数据 tensor([1, 4, 7, 3, 9,  2,  0, 12])
                # 有bug，当某一类别数据不到dataNum时数量不够
                classSelect = int(labelClass[j])
                index = torch.randperm(len(labels_index[classSelect]))[0:datanum]
                # 选取labelClass[j]类别中对应的数据 tensor([351, 2644, 2398])
                label_index_select = torch.tensor(labels_index[classSelect])[index]
                # 删去已选择的数据,避免被别的worker覆盖该数据
                labels_index[classSelect] = [x for x in labels_index[classSelect] if
                                             x not in label_index_select.tolist()]
                # 将这些数据分配给i号worker
                writers[label_index_select.tolist()] = i
        else:
            while True:
                # 从0~61类中随机选取几类数据 tensor([2, 9])
                labelClass = torch.randperm(62)[0:labelClassNum]

                # 每类数据随机分配一定百分比例 tensor([0.8884, 0.9903]) -> tensor([0.4729, 0.5271])
                dataRate = torch.rand([labelClassNum])
                dataRate = dataRate / torch.sum(dataRate)

                # batch_size ~ batch_size+5 间的随机数 tensor(16) -> tensor([8., 8.])
                dataNum = batch_size
                dataNum = torch.round(dataNum * dataRate)

                continueFlag = False
                for j in range(labelClassNum):
                    # 某一类数量不够，重新选取数据
                    if len(labels_index[int(labelClass[j])]) < int(dataNum[j]):
                        continueFlag = True
                if continueFlag:
                    continue
                break

            # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            datasnum = torch.zeros([62])
            # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            datasnum[labelClass.tolist()] = dataNum
            # 某类数据的数量
            datanum = int(dataNum[j].item())
            # 选取该类数据内的datanum个数据 tensor([1, 4, 7, 3, 9,  2,  0, 12])
            # 有bug，当某一类别数据不到dataNum时可能会有IndexError
            index = torch.randperm(len(labels_index[int(labelClass[j])]))[0:datanum]
            # 选取labelClass[j]类别中对应的数据 tensor([351, 2644, 2398])
            label_index_select = torch.tensor(labels_index[int(labelClass[j])])[index]
            # 删去已选择的数据,避免被别的worker覆盖该数据
            labels_index[classSelect] = [x for x in labels_index[classSelect] if x not in label_index_select.tolist()]
            # 将这些数据分配给i号worker
            writers[label_index_select.tolist()] = i

    with open('../data/EMNIST-ByClass/byclass/emnist-byclass-' + trainOrTest + '-writers.txt', 'w') as f_train:
        f_train.write("\n".join(map(str, map(int, writers.tolist()))))

# 生成emnist-byclass-train-writers.txt 和 emnist-byclass-test-writers.txt
dataType = 'byclass'
trainDataSize = 697932
testDataSize = 116323
genWriters(trainDataSize, 16, 3000, 4, True, dataType)
genWriters(testDataSize, 16, 3000, 4, False, dataType)

