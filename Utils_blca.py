import os
import torch
import math
import numpy as np


###BLCA
interval_cut = [0, 20, 57, 82, 110, 163, 191, 220, 251, 272, 294, 330, 344, 376, 394, 428, 466, 481, 495, 524, 544,
                577, 602, 630, 649, 696, 696, 758, 812, 832, 893, 945, 997, 1004, 1072, 1423, 1460, 1670, 1804,
                1884, 1971]

num_of_interval = len(interval_cut)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_pred_label(out):
    return torch.argmax(out,dim=2)

def cox_loss(preds,labels,status):
    labels = labels.unsqueeze(1)
    status = status.unsqueeze(1)

    mask = torch.ones(labels.shape[0],labels.shape[0]).cuda()

    mask[(labels.T - labels)>0] = 0


    log_loss = torch.exp(preds)*mask
    log_loss = torch.sum(log_loss,dim = 0)
    log_loss = torch.log(log_loss).reshape(-1,1)
    log_loss = -torch.sum((preds-log_loss)*status)

    return log_loss


def binary_label(label_l):
    survival_vector = torch.zeros(len(label_l),len(interval_cut))
    for i in range(len(label_l)):
        for j in range(len(interval_cut)):
            if label_l[i]>interval_cut[j]:
                survival_vector[i,j] = 1
    return survival_vector

def binary_last_follow(label_l):

    label_vector = torch.zeros((len(label_l),len(interval_cut)))
    for i in range(len(label_l)):
        for j in range(len(interval_cut)):
            if label_l[i] > interval_cut[j]:
                label_vector[i,j] = 1
            else:
                label_vector[i,j] = -1
    return label_vector


def calculate_time(b_pred):
    pred_ = torch.zeros(len(b_pred), dtype=float).to(device)

    for i in range(len(pred_)):
        idx = (b_pred[i] == 1).nonzero().squeeze(1)
        if len(idx) == 0:
            idx = torch.zeros(1)
        if int(idx.max().item())<len(interval_cut)-1:

            # pred_[i] = ((interval_cut[int(idx.max().item() )]+interval_cut[int(idx.max().item() + 1)])/2)
            pred_[i] = ((interval_cut[int(idx.max().item() )]+(interval_cut[int(idx.max().item() + 1)]-interval_cut[int(idx.max().item())])/2))
            # pred_[i] = interval_cut[int(idx.max().item() +1)]
        else:
            pred_[i] = (interval_cut[-1]+5)
    return pred_

def calculate_MAE_with_prob(b_pred, pred, label,status,last_follow):  ###b_pred N*I   label N

    interval = torch.zeros(len(interval_cut)).cuda()

    for i in range(len(interval_cut)):
        if i == 0:
            interval[i] = interval_cut[i+1]
        else:
            interval[i] = interval_cut[i] - interval_cut[i - 1]

    pred = pred.permute(1, 0, 2)
    estimated = torch.mul(b_pred.cuda(), pred[:, :, 1]).cuda()
    observed = torch.mul(last_follow,1-status)+torch.mul(label,status)

    estimated = torch.sum(torch.mul(estimated, interval), dim=1).cuda()

    compare = torch.zeros(len(estimated)).cuda()
    compare_invers = torch.zeros(len(estimated)).cuda()

    for i in range(len(compare)):
        compare[i] = observed[i] > estimated[i]
        compare_invers[i] = observed[i]<=estimated[i]

    MAE = torch.mul(compare,observed-estimated)+torch.mul(torch.mul(status,compare_invers),estimated-observed)


    return torch.sum(MAE)

def calculate_MAE(b_pred, label,status,last_follow):  ###b_pred N*I   label N

    pred_ = torch.zeros(len(label), dtype=float).to(device)

    for i in range(len(label)):
        idx = (b_pred[i] == 1).nonzero().squeeze(1)
        # print(len(idx))
        if len(idx) == 0:
            idx = torch.zeros(1)
        if int(idx.max().item()) < len(interval_cut) - 1:

            pred_[i] = ((interval_cut[int(idx.max().item() )]+interval_cut[int(idx.max().item() + 1)])/2)
            # pred_[i] = interval_cut[int(idx.max().item() +1)]
        else:
            pred_[i] = (interval_cut[-1] + 5)

    observed = torch.mul(last_follow,1-status)+torch.mul(label,status)
    compare = torch.zeros(len(pred_)).cuda()
    compare_invers = torch.zeros(len(pred_)).to(device)
    for i in range(len(compare)):
        compare[i] = observed[i] > pred_[i]
        compare_invers[i] = observed[i]<=pred_[i]

    MAE = torch.mul(compare,observed-pred_)+torch.mul(torch.mul(status,compare_invers),pred_-observed)

    return torch.sum(MAE)

def cross_entropy_all(b_label, pred,status,b_last_follow,weight = 1,cost = 'False'):    ###I * N
    criterion = torch.nn.CrossEntropyLoss(ignore_index= -1,reduction='none')
    total_loss = torch.zeros((num_of_interval,len(pred[0]))).to(device)

    status_ = status.unsqueeze(1)
    b_label = b_label.permute(1,0)
    weight_matrix = torch.zeros(b_label.shape).to(device)

    b_last_follow_ = b_last_follow.permute(1,0)

    combined = torch.mul(b_label,status_)+torch.mul(b_last_follow_,1-status_)
    combined = combined.permute(1,0).to(device).to(torch.long)
    for i in range(len(b_label)):
        a = torch.arange(0,len(weight_matrix[i])).to(device)
        try:
            idx = (combined.permute(1,0)[i] == 1).nonzero().max()
        except:
            idx = torch.zeros(1).to(torch.int).to(device)
        weight_matrix[i] = torch.abs(a-idx)
    for i in range(num_of_interval):
        loss = criterion(pred[i], combined[i])
        if cost == 'True':
            total_loss[i] = loss*weight
        else:
            total_loss[i] = loss
    total_loss = total_loss*weight_matrix.permute(1,0)
    total_loss = torch.sum(total_loss)

    return total_loss


def calculate_time_MAE(pred_,label,status_):  ###b_pred N*I   label N

    compare = torch.zeros(len(pred_)).to(device)
    compare_invers = torch.zeros(len(pred_)).to(device)
    for i in range(len(compare)):
        compare[i] = label[i] > pred_[i]
        compare_invers[i] = label[i]<=pred_[i]
    MAE = torch.mul(compare,label-pred_)+torch.mul(torch.mul(status_,compare_invers),pred_-label)

    return torch.sum(MAE)


