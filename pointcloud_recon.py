import torch.nn as nn
import torch
import torch.nn.functional as F
# from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
# from torchsummary import summary

class PointCloudAE(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointCloudAE, self).__init__()
        self.loss = ChamferLoss()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.02, nsample=32, in_channel=6+additional_channel, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.04, nsample=64, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[128, 128, 256], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[128, 128])
        self.fp2 = PointNetFeaturePropagation(in_channel=192, mlp=[128, 64])
        self.fp1 = PointNetFeaturePropagation(in_channel=64+3+additional_channel, mlp=[64, 32, 3+additional_channel])

        

    def forward(self, xyz):
        # Set Abstraction layers
        # B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # return l1_points
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # return l2_points
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # return l3_points
        # return torch.cat((l3_xyz, l3_points),1)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # return l2_points
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # return l1_points

        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        return l0_points

    def get_loss(self, input, output):
        # input shape  (batch_size, 2048, 3)
        # output shape (batch_size, 2025, 3)
        return self.loss(input, output)

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2
if __name__ == '__main__':
    # import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,364))
    label = torch.randn(8,16)
    model = PointCloudAE(normal_channel=False)
    output= model(input)
    print(output.size())

    # summary(model, (1,6,1024), 8)