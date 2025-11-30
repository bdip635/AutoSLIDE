from collections import defaultdict
from os.path import join
from random import randint
from scipy import ndimage
from statistics import median
import os
import shutil
import sys
import numpy

from torch import nn
import torch
import torch.nn.functional as F
import nibabel as nib

def transfer_weights(target_model, saved_model):
    """
    target_model: a model instance whose weight params are to be overwritten
    saved_model: a model whose weight params will be transfered to target.
        saved_model can be a string(path to a snapshot), an instance of model
        or a state dict of a model
    """
    target_dict = target_model.state_dict()
    if isinstance(saved_model, str):
        source_dict = torch.load(saved_model)
    else:
        source_dict = saved_model
    if not isinstance(source_dict, dict):
        source_dict = source_dict.state_dict()
    source_dict = {k: v for k, v in source_dict.items() if
                   k in target_model.state_dict() and source_dict[k].size() == target_model.state_dict()[k].size()}
    target_dict.update(source_dict)
    target_model.load_state_dict(target_dict)


def generate_ex_list(directory):
    """
    Generate list of MRI objects
    """
    inputs = []
    labels = []
    for dirpath, dirs, files in os.walk(directory):
        label_list = list()
        for file in files:
            if not file.startswith('.') and file.endswith('.nii.gz'):
                if ("Lesion" in file):
                    label_list.append(join(dirpath, file))
                elif ("mask" not in file):
                    inputs.append(join(dirpath, file))
        if label_list:
            labels.append(label_list)

    return inputs, labels


def gen_mask(lesion_files):
    """
    Given a list of lesion files, generate a mask
    that incorporates data from all of them
    """
    first_lesion = nib.load(lesion_files[0]).get_data()
    if len(lesion_files) == 1:
        return first_lesion
    lesion_data = numpy.zeros(first_lesion.shape[:3])
    for file in lesion_files:
        l_file = correct_dims(nib.load(file).get_data())
        if l_file.shape == lesion_data.shape:
            lesion_data = numpy.maximum(l_file, lesion_data)
    return lesion_data


def correct_dims(img):
    """
    Fix the dimension of the image, if necessary
    """
    if len(img.shape) > 3:
        img = img.reshape(img.shape[:3])
    return img


def get_weight_vector(labels, weight, is_cuda):
    """ Generates the weight vector for BCE loss
    You can only control positive weight, and negative weight is
    default to 1.
    So if ratio of positive and negative samples are 1:3,
    then give weight 3, and this functio returns 3 for positive and
    1 for negative samples.
    """
    if is_cuda:
        labels = labels.cpu()
    labels = labels.data.numpy()
    labels = labels * (weight-1) + 1
    weight_label = torch.from_numpy(labels).type(torch.FloatTensor)
    if is_cuda:
        weight_label = weight_label.cuda()
    return weight_label


def resize_img(input_img, label_img, size):
    """
    size: int or list of int
        when it's a list, it should include x, y, z values
    Resize image to (size x size x size)
    """
    if isinstance(size, int):
        size = [size]*3
    assert len(size) == 3
    zoom = numpy.array(size) / numpy.shape(input_img)
    ex = ndimage.zoom(input_img, zoom)
    label = ndimage.zoom(label_img, zoom)
    return ex, label


def center_crop(input_img, label_img, size):
    """
    Crop center section from image
    size: int or list of int
        when it's a list, it should include x, y, z values
    Use for testing.
    """
    if isinstance(size, int):
        size = [size]*3
    assert len(size) == 3
    coords = [0]*3
    for i in range(3):
        coords[i] = int((input_img.shape[i]-size[i])//2)
    x, y, z = coords
    ex = input_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    label = label_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    return ex, label


def find_and_crop_lesions(input_img, label_img, size, deterministic=False):
    """
    Find and crop image based on center of lesions
    size: int or list of int
        when it's a list, it should include x, y, z values
    Use for validation.
    """
    if isinstance(size, int):
        size = [size]*3
    assert len(size) == 3
    nonzeros = label_img.nonzero()
    d = [0]*3
    if not deterministic:
        for i in range(3):
            d[i] = randint(-size[i]//4, size[i]//4)

    coords = [0]*3
    for i in range(3):
        coords[i] = max(min(int(median(nonzeros[i])) - (size[i] // 2) + d[i], input_img.shape[i] - size[i] - 1), 0)
    x, y, z = coords
    ex = input_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    label = label_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    return ex, label


def random_crop(input_img, label_img, size, remove_background=False):
    """
    Crop random section from image
    size: int or list of int
        when it's a list, it should include x, y, z values
    remove_background: boolean
        use this option when input contains larger background or crop size is very small
    Use for training
    """
    if isinstance(size, int):
        size = [size]*3
    assert len(size) == 3
    non_zero_percentage = 0
    while non_zero_percentage < 0.7:
        """draw x,y,z coords
        """
        coords = [0]*3
        for i in range(3):
            if input_img.shape[i]!=size[i]:
                coords[i] = numpy.random.choice(input_img.shape[i] - size[i])
        x, y, z = coords
        ex = input_img[x:x+size[0], y:y+size[1], z:z+size[2]]
        non_zero_percentage = numpy.count_nonzero(ex) / float(size[0]*size[1]*size[2])
        if not remove_background:
            break
        if non_zero_percentage < 0.7:
            del ex

    label = label_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    return ex, label


class Report:
    EPS = sys.float_info.epsilon
    TP_KEY = 0
    TN_KEY = 1
    FP_KEY = 2
    FN_KEY = 3

    def __init__(self, threshold=0.5, smooth=sys.float_info.epsilon, apply_square=False, need_feedback=False):
        """
        apply_square: use squared elements in the denominator of soft Dice
        need_feedback: returns a tensor storing KEYS(0 to 3) for each output element
        """
        self.pos = 0
        self.neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.true_pos = 0
        self.true_neg = 0
        self.soft_I = 0
        self.soft_U = 0
        self.hard_I = 0
        self.hard_U = 0
        self.smooth = smooth
        self.apply_square = apply_square  # this variable: mainly for testing
        self.need_feedback = need_feedback
        self.threshold = threshold
        self.pathdic = defaultdict(list)

    def feed(self, pred, label, paths=None):
        """ pred size: batch x dim1 x dim2 x...
            label size: batch x dim1 x dim2 x...
            First dim should be a batch size
        """
        self.soft_I += (pred * label).sum().item()
        power_coeff = 2 if self.apply_square else 1
        if power_coeff == 1:
            self.soft_U += (pred.sum() + label.sum()).item()
        else:
            self.soft_U += (pred.pow(power_coeff).sum() + label.pow(power_coeff).sum()).item()
        pred = pred.view(-1)
        label = label.view(-1)
        pred = (pred > self.threshold).squeeze()
        not_pred = (pred == 0).squeeze()
        label = label.byte().squeeze()
        not_label = (label == 0).squeeze()
        self.pos += label.sum().item()
        self.neg += not_label.sum().item()
        pxl = pred * label
        self.hard_I += (pxl).sum().item()
        self.hard_U += (pred.sum() + label.sum()).item()
        pxnl = pred * not_label
        fp = (pxnl).sum().item()
        self.false_pos += fp
        npxl = not_pred * label
        fn = (npxl).sum().item()
        self.false_neg += fn
        tp = (pxl).sum().item()
        self.true_pos += tp
        npxnl = not_pred * not_label
        tn = (npxnl).sum().item()
        self.true_neg += tn

        feedback = None
        if self.need_feedback:
            feedback = pxl*self.TP_KEY +\
                npxnl*self.TN_KEY +\
                pxnl*self.FP_KEY +\
                npxl*self.FN_KEY
            if paths is not None:
                # Variable -> list of int
                feedback_int = [int(feedback.data[i]) for i in range(feedback.numel())]
                for i in range(len(feedback_int)):
                    if feedback_int[i] == self.TP_KEY:
                        self.pathdic["TP"].append(paths[i])
                    elif feedback_int[i] == self.TN_KEY:
                        self.pathdic["TN"].append(paths[i])
                    elif feedback_int[i] == self.FP_KEY:
                        self.pathdic["FP"].append(paths[i])
                    elif feedback_int[i] == self.FN_KEY:
                        self.pathdic["FN"].append(paths[i])
        return feedback

    def stats(self):
        text = ("Total Positives: {}".format(self.pos),
                "Total Negatives: {}".format(self.neg),
                "Total TruePos: {}".format(self.true_pos),
                "Total TrueNeg: {}".format(self.true_neg),
                "Total FalsePos: {}".format(self.false_pos),
                "Total FalseNeg: {}".format(self.false_neg))
        return "\n".join(text)

    def accuracy(self):
        return (self.true_pos+self.true_neg) / max((self.pos+self.neg), self.EPS)

    def hard_dice(self):
        numer = 2 * self.hard_I + self.smooth
        denom = self.hard_U + self.smooth
        return numer / denom

    def soft_dice(self):
        numer = 2 * self.soft_I + self.smooth
        denom = self.soft_U + self.smooth
        return numer / denom

    def __summarize(self):
        self.ACC = self.accuracy()
        self.HD = self.hard_dice()
        self.SD = self.soft_dice()

        self.P_TPR = self.true_pos / max(self.pos, self.EPS)
        self.P_PPV = self.true_pos / max((self.true_pos + self.false_pos), self.EPS)
        self.P_F1 = 2*self.true_pos / max((2*self.true_pos + self.false_pos + self.false_neg), self.EPS)

        self.N_TPR = self.true_neg / max(self.neg, self.EPS)
        self.N_PPV = self.true_neg / max((self.true_neg + self.false_neg), self.EPS)
        self.N_F1 = 2*self.true_neg / max((2*self.true_neg + self.false_neg + self.false_pos), self.EPS)

    def __str__(self):
        self.__summarize()
        summary = ("Accuracy: {:.4f}".format(self.ACC),
                   "Hard Dice: {:.4f}".format(self.HD),
                   "Soft Dice: {:.4f}".format(self.SD),
                   "For positive class:",
                   "TP(sensitivity,recall): {:.4f}".format(self.P_TPR),
                   "PPV(precision): {:.4f}".format(self.P_PPV),
                   "F-1: {:.4f}".format(self.P_F1),
                   "",
                   "For normal class:",
                   "TP(sensitivity,recall): {:.4f}".format(self.N_TPR),
                   "PPV(precision): {:.4f}".format(self.N_PPV),
                   "F-1: {:.4f}".format(self.N_F1)
                   )
        return "\n".join(summary)

def save_checkpoint(model, optimizer, file_name):

  checkpoint= {'state_dict': model.state_dict(),
             'optimizer_dict': optimizer.state_dict()}
  torch.save(checkpoint,file_name)

def load_checkpoint(model, optimizer, file_name, device):
  check_pt= torch.load(file_name, map_location= torch.device(device))
  model.load_state_dict(check_pt['state_dict'])
  optimizer.load_state_dict(check_pt['optimizer_dict'])

  return model, optimizer
  
def check_accuracy(scores, targets):
    num_correct=0
    # scores= scores.flatten()
    # targets= targets.flatten()
    _, predictions= scores.max(1)
    num_correct+= (predictions== targets).sum()
    num_samples= predictions.size(0)
    return num_correct/num_samples

def dice_coef(scores, targets):
  smooth = 1
  assert scores.shape==targets.shape
  scores = scores.view(-1)
  targets = targets.view(-1)
  intersection = (scores*targets).sum()
  union = scores.sum() + targets.sum()
  dice = (2*intersection + smooth)/(union+smooth)
  return dice

def bce_loss(scores, targets):
    assert scores.shape==targets.shape
    scores = scores.view(-1)
    targets = targets.view(-1)
    return F.binary_cross_entropy(scores, targets, reduction='mean')
    
class DiceBCELoss(nn.Module):
    def __init__(self, weights=[0.1, 0.9]):
        super(DiceBCELoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets, smooth=1e-3):
        # if inputs.shape!=targets.shape:
        #     print(inputs.shape, targets.shape)
        assert inputs.shape==targets.shape
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # weights= self.weights
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        Dice_BCE = self.weights[0]*BCE + self.weights[1]*dice_loss
        return Dice_BCE

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = 1-alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss



def Hausdorff_dist(vol_a,vol_b):
    dist_lst = []
    for idx in range(len(vol_a)):
        dist_min = 1000.0        
        for idx2 in range(len(vol_b)):
            dist= numpy.linalg.norm(vol_a[idx]-vol_b[idx2])
            if dist_min > dist:
                dist_min = dist
        dist_lst.append(dist_min)
    return numpy.max(dist_lst)
    
def batch_HD(vol_a, vol_b):
    
    assert vol_a.shape==vol_b.shape
    
    if len(vol_a.shape)==3:
        # H,W,D
        return Hausdorff_dist(vol_a, vol_b)
    
    elif len(vol_a.shape)==4:
        # B,H,W,D
        distances= [Hausdorff_dist(vol_a[i], vol_b[i]) for i in range(len(vol_a))]
        return numpy.mean(distances)
    
    elif len(vol_a.shape)==5:
        # B, 1, H, W, D
        distances= [Hausdorff_dist(vol_a[i,0], vol_b[i,0]) for i in range(len(vol_a))]
        return numpy.mean(distances)