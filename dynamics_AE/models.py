import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

class DeformerAE(nn.Module):

    def __init__(self, num_classes, normal_channel=False, decode=False):
        super().__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.decode = decode

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        if self.decode:
            self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
            self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
            self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, -1)   # latent feature vector

        if self.decode:
            # Feature Propagation layers
            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)  # reconstructed point cloud
            return x, l0_points          
        
        return x

class TransitionSimple(nn.Module):
    def __init__(self, z_dim, action_dim=3, trans_type='linear', hidden_size=64):
        super().__init__()
        self.trans_type = trans_type
        self.z_dim = z_dim

        if self.trans_type == 'linear':
            self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        
        elif self.trans_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(z_dim + action_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, z_dim)
            )
        else:
            raise Exception('Invalid trans_type', trans_type)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        x = self.model(x)
        return x


class Encoder(nn.Module):

    def __init__(self, normal_channel=False):
        super().__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 256], group_all=True)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, xyz, decode=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, -1)

        if decode:
            return x, (l0_xyz, l1_xyz, l2_xyz, l3_xyz,l0_points,l1_points, l2_points, l3_points)
        
        return x


class Decoder(nn.Module):

    def __init__(self, normal_channel=False):
        super().__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        
        self.fp3 = PointNetFeaturePropagation(in_channel=512, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+3+additional_channel, mlp=[128, 128, 3+additional_channel])

    def forward(self, saved_points):
        
        # Feature Propagation layers
        l0_xyz, l1_xyz, l2_xyz, l3_xyz,l0_points,l1_points, l2_points, l3_points = saved_points
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        return l0_points

if __name__ == '__main__':

    # input = torch.randn((8,3,2048))
    # model = Encoder()
    # output= model(input)
    # print(output.size())  

    input = torch.randn((8,3,2048))
    encoder = Encoder()
    decoder = Decoder()
    z, saved_points = encoder(input, decode=True)
    obs_recon = decoder(saved_points)
    print(obs_recon.size())      
          

