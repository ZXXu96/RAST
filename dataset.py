import torch
import os
from torch_geometric.data import Dataset,Data,InMemoryDataset,DataLoader
import numpy as np

class dataset_cox(Dataset):
    def __init__(self,dir):
        super(dataset_cox,self).__init__()
        files = os.listdir(dir)
        self.dir = [os.path.join(dir,file) for file in files]

    def __getitem__(self,index):
        path_dir = self.dir[index]
        file = np.load(path_dir,allow_pickle=True)
        patient_id = file['patient_id'][0]
        feature = file['features']
        edge_index = torch.tensor(file['edge_index'])
        status = file['status']
        try:
            follow = file['last_follow_up']
        except:
            follow = 0
        survival_time = file['survival_time']
        surv_time = torch.as_tensor(survival_time)
        if status == 0:
            last_follow_up = torch.as_tensor(follow)
        else:
            last_follow_up = torch.as_tensor(survival_time)
        if status == 0:
            out_status = torch.zeros(1)
            time = torch.as_tensor(follow)
        else:
            out_status = torch.ones(1)
            time = torch.from_numpy(survival_time)

        data = Data(x=torch.Tensor(feature), edge_index=edge_index, y=time, status=out_status,surv_time =surv_time,last_follow_up =last_follow_up,patient_id=patient_id)
        return data


    def __len__(self):
        return len(self.dir)