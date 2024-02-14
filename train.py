import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchsummary
import os
import numpy as np

import glob
from tqdm import tqdm
from ToEval import IoU_loss, saveNumData

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorV2
from data_loader import ColorJitterT
from data_loader import HorizontalFlip
from data_loader import VerticalFlip
from data_loader import SalObjDataset

from model import UNet
from model import Unet_SE_REG_Block
from model import U2NET
from model import U2NETP
from model import UnetPlusPlus
from model import BackBone
from model import BackBoneRes
from model import BackBoneSE
from model import BackBoneShape
from model import SUnet


def mutiBceLossFusion(d0, d1, d2, d3, d4, d5, d6, labelsV):
    bceLoss = nn.BCELoss()
    loss0 = bceLoss(d0, labelsV)
    loss1 = bceLoss(d1, labelsV)
    loss2 = bceLoss(d2, labelsV)
    loss3 = bceLoss(d3, labelsV)
    loss4 = bceLoss(d4, labelsV)
    loss5 = bceLoss(d5, labelsV)
    loss6 = bceLoss(d6, labelsV)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, loss      # loss0: 最后一层的单独输出  loss: 融合loss


def fuseLoss(preds, predsEdge, labels, labelsEdge, beta):
    bceLoss = nn.BCELoss()
    lossPred = bceLoss(preds, labels)
    lossEdge = bceLoss(predsEdge, labelsEdge)

    return (1 - beta) * lossPred + beta * lossEdge


def loadDRIVE(dataDir, traImageDir, traLabelDir, traLabelEdgeDir, imageExt, labelExt):
    traImgNameList = glob.glob(dataDir + traImageDir + '*' + imageExt)  # 拿到image的name列表
    traLblNameList = []
    traLbENameList = []
    for path in traImgNameList:
        imgName = path.split(os.sep)[-1]
        segImgName = imgName.split("_")[0:-1]
        idx = segImgName[0]

        traLblNameList.append(dataDir + traLabelDir + idx + "_manual1" + labelExt)  # 获取图像对应的标签
        traLbENameList.append(dataDir + traLabelEdgeDir + idx + "_manual1" + labelExt)  # 获取对应Edge标签

    return traImgNameList, traLblNameList, traLbENameList


def loadOCT(dataDir, traImageDir, traLabelDir, traLabelEdgeDir, imageExt, labelExt):
    traImgNameList = glob.glob(dataDir + traImageDir + '*' + imageExt)  # 拿到image的name列表
    traLblNameList = []
    traLbENameList = []
    for path in traImgNameList:
        imgName = path.split(os.sep)[-1]
        traLblNameList.append(dataDir + traLabelDir + imgName)
        traLbENameList.append(dataDir + traLabelEdgeDir + imgName)

    return traImgNameList, traLblNameList, traLbENameList


def save_num_data(epoch_num, dataset_name, IoU, acc, PPV, NPV, recall, spec, PA, f1score):
    file_dir = os.path.join(os.getcwd(), dataset_name + os.sep + 'num_data' + os.sep)
    IoU_file = open(file_dir + 'IoU.txt', 'a')
    acc_file = open(file_dir + 'acc.txt', 'a')
    PPV_file = open(file_dir + 'PPV.txt', 'a')
    NPV_file = open(file_dir + 'NPV.txt', 'a')
    recall_file = open(file_dir + 'recall.txt', 'a')
    spec_file = open(file_dir + 'spec.txt', 'a')
    PA_file = open(file_dir + 'PA.txt', 'a')
    f1score_file = open(file_dir + 'f1score.txt', 'a')

    IoU_file.write(str(IoU.numpy()) + '\n')
    acc_file.write(str(acc.numpy()) + '\n')
    PPV_file.write(str(PPV.numpy()) + '\n')
    NPV_file.write(str(NPV.numpy()) + '\n')
    recall_file.write(str(recall.numpy()) + '\n')
    spec_file.write(str(spec.numpy()) + '\n')
    PA_file.write(str(PA.numpy()) + '\n')
    f1score_file.write(str(f1score.numpy()) + '\n')

    if epoch_num == 149:
        print("Files Closed")
        IoU_file.close()
        acc_file.close()
        PPV_file.close()
        NPV_file.close()
        recall_file.close()
        spec_file.close()
        PA_file.close()
        f1score_file.close()


if __name__ == '__main__':
    preTrainModel = 'best_model.pth'
    modelName = 'sunet'
    datasetName = 'OCT'  # OCT or DRIVE
    dataDir = os.path.join(os.getcwd(), datasetName + os.sep)
    traImageDir = os.path.join('train_data' + os.sep + 'images' + os.sep)
    traLabelDir = os.path.join('train_data' + os.sep + 'labels' + os.sep)
    traLabelEdgeDir = os.path.join('train_data' + os.sep + 'labelsEdge' + os.sep)
    preTrainModelDir = os.path.join(os.getcwd(), 'saved_models' + os.sep + preTrainModel)
    modelDir = os.path.join(os.getcwd(), 'saved_models' + os.sep)

    imageExt = '.tif'   # DRIVE: .tif    OCT: .tif
    labelExt = '.tif'   # DRIVE: .tif    OCT: .tif

    traImgNameList, traLblNameList, traLbENameList = loadOCT(dataDir, traImageDir, traLabelDir,
                                                               traLabelEdgeDir, imageExt, labelExt)
    in_channel = 1  # DRIVE: 3      OCT: 1
    batch_size = 4 # DRIVE: 8     OCT: 4
    # DRIVE: RescaleT(576), RandomCrop(432)     OCT: RescaleT(1008), RandomCrop(784)
    traDataSet = SalObjDataset(img_name_list=traImgNameList, lbl_name_list=traLblNameList, lbE_name_list=traLbENameList,
                               transform=transforms.Compose([RescaleT(1008), RandomCrop(784), ToTensorV2()]))
    traDataLoader = DataLoader(dataset=traDataSet, batch_size=batch_size, shuffle=True, num_workers=1)

    # 模型选择
    if modelName == 'u2net':
        net = U2NET(in_channel, 1)
    elif modelName == 'u2netp':
        net = U2NETP(in_channel, 1)
    elif modelName == 'unetpp':
        print("use unetpp")
        net = UnetPlusPlus(in_channel, 1)
    elif modelName == 'unet':
        net = UNet(in_channel, 1)
    elif modelName == 'unet_se_rep':
        net = Unet_SE_REG_Block(in_channel, 1)
    elif modelName == 'backbone':
        print("usebackbone")
        net = BackBone(in_channel, 1)
    elif modelName == 'backboneres':
        print("usebackboneres")
        net = BackBoneRes(in_channel, 1)
    elif modelName == 'backbonese':
        print("usebackboneSE")
        net = BackBoneSE(in_channel, 1)
    elif modelName == 'backboneshape':
        print("usebackboneshape")
        net = BackBoneShape(in_channel, 1)
    elif modelName == 'sunet':
        print("use sunet")
        net = SUnet(in_channel, 1)

    if torch.cuda.is_available():
        # net.load_state_dict(torch.load(preTrainModelDir))
        # print('pretrained model loaded.')
        net.cuda()
# 0.001
    optimizer = optim.AdamW(params=net.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=60, gamma=0.1)

    IouCal = IoU_loss()
    epoch_num = 100

    # 评价指标
    traIoU = []
    traAcc = []
    traPPV = []
    traNPV = []
    traRecall = []
    traSpec = []
    traPA = []
    traF1score = []
    valIoU = []
    valAcc = []
    valPPV = []
    valNPV = []
    valRecall = []
    valSpec = []
    valPA = []
    valF1score = []
    
    lastf1score = 0.0

    for epoch in range(0, epoch_num):
        for i, data in enumerate(tqdm(traDataLoader)):
            inputs, labels, labelsEdge = data['image'], data['label'], data['labelEdge']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            labelsEdge = labelsEdge.type(torch.FloatTensor)

            beta = labelsEdge.sum() / labels.sum()  # loss分配权重
            # beta = 0.7
            inputsCuda, labelsCuda, labelsEdgeCuda = Variable(inputs.cuda()), Variable(labels.cuda()), Variable(labelsEdge.cuda())

            if i != len(traDataLoader) - 1:
                curType = 'train'
                net.train()
                if modelName == 'u2net':
                    d0, d1, d2, d3, d4, d5, d6 = net(inputsCuda)
                    loss2, loss = mutiBceLossFusion(d0, d1, d2, d3, d4, d5, d6, labelsCuda)  # 损失函数(u2net)
                    optimizer.zero_grad()
                    loss.backward()  # 反向传播（梯度值）
                    optimizer.step()  # 梯度更新（参数值）
                elif (modelName == 'unet_se_rep') or (modelName == 'backboneshape') or (modelName == 'sunet'):
                    d1, d0 = net(inputsCuda)
                    loss = fuseLoss(preds=d1, predsEdge=d0, labels=labelsCuda, labelsEdge=labelsEdgeCuda, beta=beta)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    d1 = net(inputsCuda)
                    singleLoss = nn.BCELoss()
                    loss = singleLoss(d1, labelsCuda)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            else:
                curType = 'validate'
                net.eval()
                with torch.no_grad():
                    if modelName == 'u2net':
                        d0, d1, d2, d3, d4, d5, d6 = net(inputsCuda)
                    elif modelName == (modelName == 'unet_se_rep') or (modelName == 'backboneshape') or (modelName == 'sunet'):
                        d1, d0 = net(inputsCuda)
                    else:
                        d1 = net(inputsCuda)

            # predicts
            pred = d1[:, 0, :, :].reshape([-1, 1, 784, 784]).round()
            # 计算评价指标
            IoU1, acc1, PPV1, NPV1, recall1, spec1, PA1, f1score1 = IouCal(pred.to('cpu'), labels.type(torch.IntTensor))

            if curType == 'train':
                traIoU.append(IoU1)
                traAcc.append(acc1)
                traPPV.append(PPV1)
                traNPV.append(NPV1)
                traRecall.append(recall1)
                traSpec.append(spec1)
                traPA.append(PA1)
                traF1score.append(f1score1)
            elif curType == 'validate':
                valIoU.append(IoU1)
                valAcc.append(acc1)
                valPPV.append(PPV1)
                valNPV.append(NPV1)
                valRecall.append(recall1)
                valSpec.append(spec1)
                valPA.append(PA1)
                valF1score.append(f1score1)

            torch.cuda.empty_cache()

        scheduler.step()
        print("Epoch: ", format(epoch))
        print("Current lr :        ", format(optimizer.state_dict()['param_groups'][0]['lr']))
        print("Train f1-score :    ", np.mean(traF1score))
        print("Valid f1-score :    ", np.mean(valF1score))

        # if (curType == 'validate') and np.mean(valF1score) > lastf1score:
        torch.save(net.state_dict(), './saved_models/best_model.pth')
        lastf1score = np.mean(valF1score)
        print("model saved!")
        # save train data
        saveNumData(epoch_num=epoch, dataset_name=datasetName, curType='train', IoU=np.mean(traIoU), acc=np.mean(traAcc),
                    PPV=np.mean(traPPV), NPV=np.mean(traNPV), recall=np.mean(traRecall), spec=np.mean(traSpec), PA=np.mean(traPA), f1score=np.mean(traF1score))
        # save valid data
        saveNumData(epoch_num=epoch, dataset_name=datasetName, curType='validate', IoU=np.mean(valIoU),acc=np.mean(valAcc),
                    PPV=np.mean(valPPV), NPV=np.mean(valNPV), recall=np.mean(valRecall), spec=np.mean(valSpec), PA=np.mean(valPA), f1score=np.mean(valF1score))

        # clear list
        traIoU.clear()
        traAcc.clear()
        traPPV.clear()
        traNPV.clear()
        traRecall.clear()
        traSpec.clear()
        traPA.clear()
        traF1score.clear()
        valIoU.clear()
        valAcc.clear()
        valPPV.clear()
        valNPV.clear()
        valRecall.clear()
        valSpec.clear()
        valPA.clear()
        valF1score.clear()

    torchsummary.summary(net, (in_channel, 784, 784))
    for module in net.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    
    torch.save(net.state_dict(), './saved_models/best_squeezed_model.pth')
    torchsummary.summary(net, (in_channel, 784, 784))
