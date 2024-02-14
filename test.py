from PIL import Image
import os
from skimage import io
import glob
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torchsummary
from sklearn.metrics import roc_auc_score
import numpy as np

from ToEval import IoU_loss, saveNumData, db_eval_boundary

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


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    img_idx = bbb[0]
    for i in range(1, len(bbb)):
        img_idx = img_idx + "." + bbb[i]

    imo.save(d_dir + img_idx + '.png')


def loadDRIVE(imageDir, labelDir, labelEdgeDir):
    imgNameList = sorted(glob.glob(imageDir + os.sep + '*_test.tif'))
    lblNameList = sorted(glob.glob(labelDir + os.sep + '*_manual1.gif'))
    lbENameList = sorted(glob.glob(labelEdgeDir + os.sep + '*_manual1.tif'))
    return imgNameList, lblNameList, lbENameList


def loadOCT(imageDir, labelDir, labelEdgeDir):
    imgNameList = sorted(glob.glob(imageDir + os.sep + "*.tif"))
    lblNameList = sorted(glob.glob(labelDir + os.sep + "*.tif"))
    lbENameList = sorted(glob.glob(labelEdgeDir + os.sep + '*.tif'))
    return imgNameList, lblNameList, lbENameList


if __name__ == '__main__':

    modelName = 'sunet'
    preModelName = 'best_model.pth'  # 预训练模型
    datasetName = 'OCT'  # OCT or DRIVE
    data = 'images'
    label = 'labels'
    labelEdge = 'labelsEdge'
    datasetDir = os.path.join(os.getcwd(), datasetName + os.sep)
    imageDir = os.path.join(os.getcwd(), datasetDir, 'test_data', data)  # 测试图像地址（test_data存放测试图像的上一级文件夹）
    labelDir = os.path.join(os.getcwd(), datasetDir, 'test_data', label)
    labelEdgeDir = os.path.join(os.getcwd(), datasetDir, 'test_data', labelEdge)
    predictionDir = os.path.join(os.getcwd(), datasetDir, 'test_data', data + '_results' + os.sep)  # 结果存放地址（若无，则自动新建文件夹）
    modelDir = os.path.join(os.getcwd(), 'saved_models', preModelName)

    imgNameList, lblNameList, lbENameList = loadOCT(imageDir, labelDir, labelEdgeDir)

    # DRIVE: RescaleT(576)    OCT: RescaleT: 1008
    in_channel = 1  # DRIVE: 3    OCT: 1
    re_size = 1008  # DRIVE: 576  OCT: 1008
    threshold = 0.5

    # 评价指标
    IouCal = IoU_loss()
    IoUSum = 0.0
    accSum = 0.0
    PPVSum = 0.0
    NPVSum = 0.0
    recallSum = 0.0
    specSum = 0.0
    PASum = 0.0
    f1scoreSum = 0.0
    AUCSum = 0.0
    FSum = 0.0
    times = 0

    testDataset = SalObjDataset(img_name_list=imgNameList, lbl_name_list=lblNameList, lbE_name_list=lbENameList,
                                transform=transforms.Compose([RescaleT(re_size), ToTensorV2()]))
    testDataloader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=1)

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
        print("use SUnet")
        net = SUnet(in_channel, 1, deploy=False)
        
    if torch.cuda.is_available():
        # net.switch_net_to_deploy()
        print(modelDir)
        net.load_state_dict(torch.load(modelDir))
        net.cuda()

    net.eval()  # 测试模型

    for i, data in enumerate(testDataloader):
        inputs = data['image']
        labels = data['label']
        labelsEdge = data['labelEdge']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        labelsEdge = labelsEdge.type(torch.FloatTensor)

        inputsCuda, labelsCuda, labelsEdgeCuda = Variable(inputs.cuda()), Variable(labels.cuda()), Variable(labelsEdge.cuda())

        net.load_state_dict(torch.load(modelDir))
        if modelName == 'u2net':
            d0, d1, d2, d3, d4, d5, d6 = net(inputsCuda)
        elif modelName == (modelName == 'unet_se_rep') or (modelName == 'backboneshape') or (modelName == 'sunet'):
            d1, d0 = net(inputsCuda)
            predEdge = d0[:, 0, :, :].reshape([-1, 1, re_size, re_size])
        else:
            d1 = net(inputsCuda)

        pred = d1[:, 0, :, :].reshape([-1, 1, re_size, re_size])
        pred_numpy = pred.data.cpu().numpy()
        pred_binary = (pred_numpy > threshold).astype(int)
        pred_binary = torch.from_numpy(pred_binary).type(torch.FloatTensor).reshape([-1, 1, re_size, re_size])
        
        # 计算评价指标
        IoU1Sum, acc1Sum, PPV1Sum, NPV1Sum, recall1Sum, spec1Sum, PA1Sum, f1score1Sum = IouCal(pred_binary.type(torch.IntTensor).to('cpu'), labels.type(torch.IntTensor))
        labels_n = labels.reshape(labels.shape[2] * labels.shape[3], 1).detach().numpy()
        pred_n = pred.reshape(pred.shape[2] * pred.shape[3], 1).detach().to('cpu').numpy()
        AUC = roc_auc_score(labels_n.astype('int16'), pred_n)
        
        IoUSum = IoUSum + IoU1Sum
        accSum = accSum + acc1Sum
        PPVSum = PPVSum + PPV1Sum
        NPVSum = NPVSum + NPV1Sum
        recallSum = recallSum + recall1Sum
        specSum = specSum + spec1Sum
        PASum = PASum + PA1Sum
        f1scoreSum = f1scoreSum + f1score1Sum
        AUCSum = AUCSum + AUC

        if not os.path.exists(predictionDir):
            os.makedirs(predictionDir, exist_ok=True)

        save_output(imgNameList[i], pred, predictionDir)  # 保存预测图像

        torch.cuda.empty_cache()
        times = times + 1

        F = db_eval_boundary((pred_n > threshold).astype(int), (labels_n > threshold).astype(int), bound_th=3)
        FSum = FSum + F
        print("F-score : ", F)
        # print(times)
        # print("Current Iou Score : ", format(IoU1Sum))
        # print("Current acc Score : ", format(acc1Sum))
        # print("Current PPV Score : ", format(PPV1Sum))
        # print("Current NPV Score : ", format(NPV1Sum))
        # print("Current recall Score : ", format(recall1Sum))
        # print("Current spec Score : ", format(spec1Sum))
        # print("Current PA Score : ", format(PA1Sum))
        # print("Current f1score Score : ", format(f1score1Sum))
        # print("Current AUC Score : ", format(AUC))
        print(pred_binary.type(torch.IntTensor).to('cpu').max())

    saveNumData(epoch_num=149, dataset_name=datasetDir, curType='test', IoU=IoUSum/times, acc=accSum/times, PPV=PPVSum/times,
                NPV=NPVSum/times, recall=recallSum/times, spec=specSum/times, PA=PASum/times, f1score=f1scoreSum/times)
    print("Total AUC : ", format(AUCSum/times))
    print("Total F   : ", format(FSum/times))
    
    torchsummary.summary(net, (in_channel, 1008, 1008))
