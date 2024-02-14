import torch
from torch.autograd import Variable
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def saveNumData(epoch_num, dataset_name, curType, IoU, acc, PPV, NPV, recall, spec, PA, f1score):
    if curType == 'train':
        file_dir = os.path.join(os.getcwd(), dataset_name + os.sep + 'num_data_train' + os.sep)
    elif curType == 'validate':
        file_dir = os.path.join(os.getcwd(), dataset_name + os.sep + 'num_data_valid' + os.sep)
    elif curType == 'test':
        file_dir = os.path.join(os.getcwd(), dataset_name + os.sep + 'num_data_test' + os.sep)
    IoU_file = open(file_dir + 'IoU.txt', 'a')
    acc_file = open(file_dir + 'acc.txt', 'a')
    PPV_file = open(file_dir + 'PPV.txt', 'a')
    NPV_file = open(file_dir + 'NPV.txt', 'a')
    recall_file = open(file_dir + 'recall.txt', 'a')
    spec_file = open(file_dir + 'spec.txt', 'a')
    PA_file = open(file_dir + 'PA.txt', 'a')
    f1score_file = open(file_dir + 'f1score.txt', 'a')

    IoU_file.write(str(IoU) + '\n')
    acc_file.write(str(acc) + '\n')
    PPV_file.write(str(PPV) + '\n')
    NPV_file.write(str(NPV) + '\n')
    recall_file.write(str(recall) + '\n')
    spec_file.write(str(spec) + '\n')
    PA_file.write(str(PA) + '\n')
    f1score_file.write(str(f1score) + '\n')


    IoU_file.close()
    acc_file.close()
    PPV_file.close()
    NPV_file.close()
    recall_file.close()
    spec_file.close()
    PA_file.close()
    f1score_file.close()


class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()
        # self.write = SummaryWriter("logs")
        # self.i = 1
        
    def forward(self, pred, target):
        b = pred.shape[0]
        h = pred.shape[2]
        w = pred.shape[3]
        zes = Variable(torch.zeros([1, 1, h, w]).type(torch.IntTensor))  # 全0变量
        ons = Variable(torch.ones([1, 1, h, w]).type(torch.IntTensor))  # 全1变量

        IoU = 0.0
        acc = 0.0
        PPV = 0.0
        NPV = 0.0
        recall = 0.0
        spec = 0.0
        PA = 0.0
        f1score = 0.0

        for i in range(0, b):
            # compute the IoU of the foreground
            TP = ((pred[i, :, :, :] == ons) & (target[i, :, :, :] == ons)).sum()
            FP = ((pred[i, :, :, :] == ons) & (target[i, :, :, :] == zes)).sum()
            TN = ((pred[i, :, :, :] == zes) & (target[i, :, :, :] == zes)).sum()
            FN = ((pred[i, :, :, :] == zes) & (target[i, :, :, :] == ons)).sum()
            IoU = IoU + TP / (TP + FN + FP + 1e-5)
            acc = acc + (TP + TN) / (TP + FN + FP + TN + 1e-5)  # 准确率（错误验证占比）
            PPV = PPV + TP / (TP + FP + 1e-5)  # precision/精确率/阳性预测率(模型预测出来的所有positive中，预测正确的占比)
            NPV = NPV + TN / (TN + FN + 1e-5)  # 阴性预测率(模型预测出来所有的negative中，预测正确的占比)
            recall = recall + TP / (TP + FN)  # 召回率/灵敏度/TPR(所有真实positive，模型预测正确的positive占比)
            spec = spec + TN / (TN + FP + 1e-5)  # 特异度/TNR(所有真实negative，模型预测正确的negative占比)
            PA = PA + (TP + TN) / (TP + TN + FP + FN + 1e-5)

        f1score = (2 * PPV * recall / b / b) / (PPV / b + recall / b)
        return IoU / b, acc / b, PPV / b, NPV / b, recall / b, spec / b, PA / b, f1score

    
def db_eval_boundary(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);
    # print("fg_boundary : ", np.sum(fg_boundary))
    # print("gt_boundary : ", np.sum(gt_boundary))
    
    from skimage.morphology import binary_dilation,disk

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall);

    return F

def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
        'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg^e | seg^s | seg^se
    b[-1, :] = seg[-1, :]^e[-1, ]
    b[:, -1] = seg[:, -1]^s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1+math.floor((y-1)+height / h);
                    i = 1+math.floor((x-1)+width  / h);
                    bmap[j, i] = 1;

    return bmap