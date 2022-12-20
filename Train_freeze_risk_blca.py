from torch_geometric.loader import DataLoader,DataListLoader
import torch
import numpy as np
from My_datasets import dataset_cox
from torch_geometric.nn import DataParallel
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import os
from model import GCN_BIG,GCN_Time_Only,GCN_Freeze_Risk
from lifelines.utils import concordance_index
import math
from Utils_blca import *
import json
import matplotlib.pyplot as plt
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0，1'
cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--train_npz_dir',type=str, default = '.', help='training set dir')
parser.add_argument('--val_npz_dir',type=str, default = '.', help='validation set dir')
parser.add_argument('--test_npz_dir',type=str, default = '.', help='test set dir')
parser.add_argument('--log_dir',type=str, default = '.', help='validation set dir')
parser.add_argument('--model_save_dir',type=str, default = '.', help='model save dir')
parser.add_argument('--risk_model_save_dir',type=str, default = '.', help='model save dir')
parser.add_argument('--seed', type=int, default=1024, help='Random seed.')
parser.add_argument('--batch_size',type=int,default=128, help = 'mini_batch size')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--save', type=bool, default=False,
                    help='.')
parser.add_argument('--tensorboard', type=bool, default=False,
                    help='.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='cox loss trade off.')
args = parser.parse_args()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False


def train(epoch, save='True', tsboard='True'):
    global val_highest
    global risk_highest
    global MAE
    global MAE_last_lowest

    model.train()
    scheduler.step(epoch)
    epoch_loss = 0
    epoch_MAE_prob = 0
    epoch_MAE_last = 0

    for ii, data in tqdm(enumerate(train_batch_data)):
        optimizer.zero_grad()


        '''if not DataParallel, Delete codes below, use data.surv_time, data.last_follow_up ,etc'''
        surv_time = [d.surv_time.unsqueeze(0) for d in data]
        surv_time = torch.cat(surv_time).to(device)
        last_follow_up = [d.last_follow_up.unsqueeze(0) for d in data]
        last_follow_up = torch.cat(last_follow_up).to(device)
        status = [d.status for d in data]
        status = torch.cat(status).to(device)
        y = [d.y.unsqueeze(0) for d in data]


        b_label = torch.tensor(binary_label(surv_time), dtype=torch.long).to(device)  # N*I
        b_last_follow = torch.tensor(binary_last_follow(last_follow_up), dtype=torch.long).to(device)  # N*I

        pred,risk = model(data)
        pred = pred.reshape(len(pred),num_of_interval,-1).permute(1,0,2)

        '''if not Datapallel delete this and predictions = predictions.permute(1,0,2) in BigModel'''

        pred_softmax = torch.softmax(pred, dim=2).to(device)

        b_pred = get_pred_label(pred.permute(1, 0, 2))

        MAE_prob = calculate_MAE_with_prob(b_pred, pred_softmax, surv_time, status, last_follow_up)
        MAE_last = calculate_MAE(b_pred, surv_time, status, last_follow_up)


        loss = cross_entropy_all(b_label.permute(1, 0), pred_softmax, status, b_last_follow.permute(1, 0),cost='False')
        epoch_MAE_prob += MAE_prob.item()
        epoch_MAE_last += MAE_last.item()
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    if save == True:
        torch.save(model, args.model_save_dir + 'epoch%d.pth' % epoch)
    print('==========epoch%s===========' % epoch)
    print('loss:%f' % (epoch_loss / len(train_data)))
    print('MAE_prob: %f' % (epoch_MAE_prob / len(train_data)))
    print('MAE_last: %f' % (epoch_MAE_last / len(train_data)))

    if tsboard == 'True':
        writer.add_scalars("train_loss", {'Train':epoch_loss / len(train_data)}, e)
        writer.add_scalar("train_MAE_prob_fx", epoch_MAE_prob / len(train_data), e)
        writer.add_scalar("train_MAE_last_fx", epoch_MAE_last / len(train_data), e)

    ####validation

    model.eval()
    val_epoch_time_index = 0
    val_total_MAE = 0
    val_epoch_risk_index = 0
    val_epoch_loss = 0

    for jj, v_data in enumerate(val_batch_data):

        with torch.no_grad():
            v_pred,v_risk = model(v_data)

        v_pred = v_pred.reshape(len(v_pred), num_of_interval, -1).permute(1, 0, 2)

        v_status = [d.status for d in v_data]
        v_status = torch.cat(v_status).to(device)
        v_y = [d.y.unsqueeze(0) for d in v_data]
        v_y = torch.cat(v_y).to(device)
        v_last_follow = [d.last_follow_up.unsqueeze(0) for d in v_data]
        v_last_follow = torch.cat(v_last_follow).to(device)
        v_patient_id = [d.patient_id for d in v_data]
        v_surv_time = [d.surv_time.unsqueeze(0) for d in v_data]
        v_surv_time = torch.cat(v_surv_time).to(device)

        v_b_label = torch.tensor(binary_label(v_surv_time), dtype=torch.long).to(device)
        v_b_last_follow = torch.tensor(binary_last_follow(v_last_follow), dtype=torch.long).to(device)
        v_pred_softmax = torch.softmax(v_pred, dim=2).to(device)

        with torch.no_grad():
            v_loss = cross_entropy_all(v_b_label.permute(1, 0), v_pred_softmax, v_status, v_b_last_follow.permute(1, 0), cost='False')

        patient_list,uni_idx = np.unique(v_patient_id,return_index=True)
        num_patients = len(patient_list)
        result_matrix = torch.zeros((num_patients)).to(device)
        counter_matrix = torch.zeros((num_patients)).to(device)
        risk_matrix = torch.zeros((num_patients)).to(device)
        label_matrix = v_y[uni_idx].to(device)
        status_matrix = v_status[uni_idx].to(device)

        v_b_pred = get_pred_label(v_pred.permute(1, 0, 2))

        v_time = calculate_time(v_b_pred)
        # print(v_time)

        for ii,patient in enumerate(v_patient_id):
            result_matrix[np.argwhere(patient_list == patient)] += v_time[ii]
            risk_matrix[np.argwhere(patient_list == patient)] += v_risk[ii]
            counter_matrix[np.argwhere(patient_list == patient)] += 1
        result_matrix = result_matrix/counter_matrix
        risk_matrix = risk_matrix/counter_matrix

        v_MAE = calculate_time_MAE(result_matrix,label_matrix,status_matrix)

        v_c_index = concordance_index(label_matrix.cpu().detach().numpy(), result_matrix.cpu().detach().numpy(),
                                      status_matrix.cpu().detach().numpy())
        v_risk_index = concordance_index(label_matrix.cpu().detach().numpy(), -risk_matrix.cpu().detach().numpy(),
                                      status_matrix.cpu().detach().numpy())



        val_epoch_loss += v_loss.item()
        val_total_MAE += v_MAE.item()
        val_epoch_time_index += v_c_index
        val_epoch_risk_index += v_risk_index

    print('val loss: %f'%(val_epoch_loss / len(val_data)))
    print('MAE_val %f' % (val_total_MAE / len(val_data)))
    print('val-c-index:%f' % (val_epoch_time_index / (jj + 1)))
    print('val-risk-index:%f'%(val_epoch_risk_index/(jj+1)))
    if MAE > val_total_MAE / len(val_data):
        MAE = val_total_MAE / len(val_data)
        val_highest = val_epoch_time_index / (jj + 1)
        if save == True:
            torch.save(model, args.model_save_dir + 'best_validation.pth' )
    print('val_highest:%f' % val_highest)
    print('MAE_lowest:%f' % MAE)
    if tsboard == True:
        writer.add_scalar("val_c-index", (val_epoch_time_index / (jj + 1)), e)
        writer.add_scalar("val_MAE", val_total_MAE / len(val_data), e)
        writer.add_scalars("train_loss", {'Validation':val_epoch_loss / len(val_data)}, e)
    make_dirs(args.log_dir)
    with open(os.path.join(args.log_dir,'BLCA_freeze_risk_nocost_val.json'),'a') as j:
        json.dump((str(epoch),str(val_highest),str(MAE),str(val_epoch_risk_index)),j)
        j.write('\n')

def test(require_json='False'):
    print('-----------------------test-----------------------')
    model.eval()
    test_epoch_time_index = 0
    test_total_MAE = 0
    test_epoch_risk_index = 0

    for jj, t_data in enumerate(test_batch_data):

        with torch.no_grad():
            t_pred,t_risk = model(t_data)

        t_pred = t_pred.reshape(len(t_pred), num_of_interval, -1).permute(1, 0, 2)

        t_status = [d.status for d in t_data]
        t_status = torch.cat(t_status).to(device)
        t_y = [d.y.unsqueeze(0) for d in t_data]
        t_y = torch.cat(t_y).to(device)
        t_patient_id = [d.patient_id for d in t_data]

        patient_list,uni_idx = np.unique(t_patient_id,return_index=True)
        num_patients = len(patient_list)
        result_matrix = torch.zeros((num_patients)).to(device)
        counter_matrix = torch.zeros((num_patients)).to(device)
        risk_matrix = torch.zeros((num_patients)).to(device)
        label_matrix = t_y[uni_idx].to(device)
        status_matrix = t_status[uni_idx].to(device)

        t_b_pred = get_pred_label(t_pred.permute(1, 0, 2))

        t_time = calculate_time(t_b_pred)
        # print(t_time)

        for ii,patient in enumerate(t_patient_id):
            result_matrix[np.argwhere(patient_list == patient)] += t_time[ii]
            risk_matrix[np.argwhere(patient_list == patient)] += t_risk[ii]
            counter_matrix[np.argwhere(patient_list == patient)] += 1
        result_matrix = result_matrix/counter_matrix
        risk_matrix = risk_matrix/counter_matrix

        t_MAE = calculate_time_MAE(result_matrix,label_matrix,status_matrix)

        t_c_index = concordance_index(label_matrix.cpu().detach().numpy(), result_matrix.cpu().detach().numpy(),
                                      status_matrix.cpu().detach().numpy())
        t_risk_index = concordance_index(label_matrix.cpu().detach().numpy(), -risk_matrix.cpu().detach().numpy(),
                                      status_matrix.cpu().detach().numpy())

        # val_epoch_loss += v_loss.item()
        test_total_MAE += t_MAE.item()
        test_epoch_time_index += t_c_index
        test_epoch_risk_index += t_risk_index

    print(label_matrix)
    print(result_matrix)
    print(status_matrix)
    print('MAE_test %f' % (test_total_MAE / len(label_matrix)))
    print('test-c-index:%f' % (test_epoch_time_index / (jj + 1)))
    print('test-risk-index:%f'%(test_epoch_risk_index/(jj+1)))


    if require_json == 'True':
        make_dirs(args.log_dir)
        with open(os.path.join(args.log_dir,'BLCA.json'),'a') as j:
            json.dump((str(test_total_MAE/len(test_data)),str(test_epoch_time_index)),j)
            j.write('\n')



if __name__ == '__main__':

    seed_torch(args.seed)

    num_of_interval = len(interval_cut)

    ''' if DataParallel -> DataListLoader
        if not DataParallel -> DataLoader
        check https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.data_parallel'''
    train_data = dataset_cox(args.train_npz_dir)
    train_batch_data = DataListLoader(train_data,batch_size=args.batch_size,shuffle=True,drop_last=True)

    val_data = dataset_cox(args.val_npz_dir)
    val_batch_data = DataListLoader(val_data,batch_size =len(val_data),shuffle=False)

    test_data = dataset_cox(args.test_npz_dir)
    test_batch_data = DataListLoader(test_data,batch_size=len(test_data),shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tsboard = True
    save = True

    risk_dir = os.path.join(args.risk_model_save_dir,'best_validation.pth')

    paras = torch.load(risk_dir).state_dict()

    new_paras = {}

    for para in paras:
        new_paras[para[7:]] = paras[para]

    model = GCN_Freeze_Risk(num_interval=num_of_interval).to(device)

    model.GCN_cox.load_state_dict(state_dict=new_paras)

    model.GCN_cox.requires_grad_(False)
    model = DataParallel(model)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50], gamma=0.5)
    make_dirs(args.model_save_dir)

    if tsboard == True:
        writer = SummaryWriter()

    val_highest = 0
    risk_highest = 0
    MAE = 9999
    MAE_last_lowest = 9999
    for e in range(args.epochs):
        train(e,save=save,tsboard=tsboard)


    risk_freeze_dir = os.path.join(args.model_save_dir,'best_validation.pth')
    paras_2 = torch.load(risk_freeze_dir).state_dict()

    new_paras_2 = {}

    for para in paras_2:
        new_paras_2[para[7:]] = paras_2[para]

    model = GCN_Freeze_Risk(num_interval=num_of_interval).to(device)
    model.load_state_dict(new_paras_2)
    model = DataParallel(model)
    test(require_json='True',require_resutls='False')



