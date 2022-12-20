import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import SAGPooling
from torch_sparse import SparseTensor
import numpy as np
from Utils_BLCA import calculate_time
from torch_geometric.nn import BatchNorm


class BinaryClassifier(torch.nn.Module):

    def __init__(self, classifier_in):
        super(BinaryClassifier, self).__init__()
        in_channels = classifier_in
        out_channels = 2

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=out_channels),
        )

    def forward(self, x):
        return self.classifier(x)


class BigModel(torch.nn.Module):

    def __init__(self, num_of_interval, classifier_in):
        super(BigModel, self).__init__()
        self.interval = num_of_interval
        self.num_class = 2
        for i in range(self.interval):
            exec('self.classifier%s= BinaryClassifier(classifier_in)' % i)

    def forward(self, x):
        predictions = torch.zeros((self.interval, len(x), self.num_class)).cuda()
        for i in range(self.interval):
            exec('x%s = self.classifier%s(x)' % (i, i))
            exec('predictions[%s] = x%s' % (i, i))
        predictions = predictions.permute(1, 0, 2)
        return predictions


class BigModel_onevector(torch.nn.Module):

    def __init__(self, num_of_interval, classifier_in):
        super(BigModel_onevector, self).__init__()
        self.interval = num_of_interval
        self.num_class = 2
        self.fc = torch.nn.Linear(classifier_in, self.interval * self.num_class)

    def forward(self, x):
        x = self.fc(x)

        return x


class GCN_Risk(torch.nn.Module):

    def __init__(self):
        super(GCN_Risk, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)
        self.lin = torch.nn.Linear(512, 256)
        self.lin2 = torch.nn.Linear(256, 1)

        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)
        self.norm1 = BatchNorm(512, track_running_stats=False)
        self.norm2 = BatchNorm(512, track_running_stats=False)
        self.norm3 = BatchNorm(256, track_running_stats=False)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x = self.conv4(x, edge_index)

        x1 = torch.cat((global_max_pool(x, batch=batch, size=data.batch.max() + 1),
                        global_mean_pool(x, batch=batch, size=data.batch.max() + 1)), dim=1)

        return x, x1


class GCN_Time(torch.nn.Module):

    def __init__(self, num_interval, classifier_in):
        super(GCN_Time, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)

        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)

        self.norm1 = BatchNorm(512, track_running_stats=False)
        self.norm2 = BatchNorm(512, track_running_stats=False)
        self.norm3 = BatchNorm(256, track_running_stats=False)


    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.norm1(x)
        x = F.tanh(x)

        x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.tanh(x)

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.tanh(x)

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x = self.conv4(x, edge_index)

        x = torch.cat((global_max_pool(x, batch=batch, size=data.batch.max() + 1),
                       global_mean_pool(x, batch=batch, size=data.batch.max() + 1)), dim=1)


        return x





class GCN_Risk_Only(torch.nn.Module):

    def __init__(self):
        super(GCN_Risk_Only, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)
        self.lin = torch.nn.Linear(512, 256)
        self.lin2 = torch.nn.Linear(256, 1)

        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)
        self.norm1 = BatchNorm(512, track_running_stats=False)
        self.norm2 = BatchNorm(512, track_running_stats=False)
        self.norm3 = BatchNorm(256, track_running_stats=False)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x = self.conv4(x, edge_index)

        x1 = torch.cat((global_max_pool(x, batch=batch, size=data.batch.max() + 1),
                        global_mean_pool(x, batch=batch, size=data.batch.max() + 1)), dim=1)

        x = self.lin(x1)
        x = F.relu(x)
        x = self.lin2(x)

        return x, x1




class GCN_Time_Only(torch.nn.Module):

    def __init__(self, num_interval, classifier_in):
        super(GCN_Time_Only, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)

        self.relu = torch.nn.ReLU()
        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)
        self.classifier = BigModel(num_of_interval=num_interval, classifier_in=classifier_in)



    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.leaky_relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x2 = self.conv4(x, edge_index)

        x1 = torch.cat((global_max_pool(x2, batch=batch, size=data.batch.max() + 1),
                        global_mean_pool(x2, batch=batch, size=data.batch.max() + 1)), dim=1)

        x = self.classifier(x1)

        return x



class GCN_Freeze_Risk(torch.nn.Module):

    def __init__(self, num_interval):
        super(GCN_Freeze_Risk, self).__init__()
        classifier_in = 512

        self.GCN_cox = GCN_Risk()
        self.GCN_ordinal = GCN_Time(num_interval=num_interval, classifier_in=classifier_in)

        self.classifier = BigModel(num_of_interval=num_interval, classifier_in=classifier_in)

    def forward(self, x, train='True'):

        r, r_feature = self.GCN_cox(x)

        x = self.GCN_ordinal(x)     ### N*512
        r_feature = r_feature.detach_()
        r = r.detach_()
        x = torch.add(x,r_feature)
        x = x.div(2)

        x = self.classifier(x)

        return x, r



