import torch.nn as nn
import torch
import torch.nn.functional as F
# from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
# from torchsummary import summary
# from pointconv_util import PointConvDensitySetAbstraction
from pointconv_util_groupnorm import PointConvDensitySetAbstraction


  


class PointNetShapeServoKp(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNetShapeServoKp, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=64, nsample=8, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=64 + 3, mlp=[128], bandwidth = 0.4, group_all=True)    

        self.fc1 = nn.Linear(128, 64)
        # self.bn1 = nn.BatchNorm1d(128, momentum = 0.5)
        self.bn1 = nn.GroupNorm(1, 64)
        # self.fc2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)


        self.fc3 = nn.Linear(64, 32)
        # self.bn3 = nn.BatchNorm1d(64, momentum = 0.5)
        self.bn3 = nn.GroupNorm(1, 32)
        self.fc4 = nn.Linear(32, 16)
        # self.bn4 = nn.BatchNorm1d(32, momentum = 0.5)
        self.bn4 = nn.GroupNorm(1, 16)
        self.fc5 = nn.Linear(16, 3)


    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l1_xyz, l1_points)        
        
        x = l3_points.view(B, 128)


        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:,:3,:]
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal
       
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l1_xyz, l1_points)        
        
        
        g = l3_points.view(B, 128)
        x = g - x
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))

        x = self.fc5(x)

        return x



class PointNetShapeServoKp2(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNetShapeServoKp2, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)    

        # self.sa1 = PointConvDensitySetAbstraction(npoint=32, nsample=4, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=16, nsample=8, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)            

        self.fc1 = nn.Linear(256, 128)
        # self.bn1 = nn.BatchNorm1d(128, momentum = 0.5)
        self.bn1 = nn.GroupNorm(1, 128)
        self.drop1 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)


        self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64, momentum = 0.5)
        self.bn3 = nn.GroupNorm(1, 64)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 32)
        # self.bn4 = nn.BatchNorm1d(32, momentum = 0.5)
        self.bn4 = nn.GroupNorm(1, 32)
        self.drop4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(32, 3)


    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 256)


        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:,:3,:]
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        g = l3_points.view(B, 256)


        x = g - x
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn3(self.fc3(x)))
        # x = F.relu(self.bn4(self.fc4(x)))

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.relu(self.bn4(self.fc4(x))))

        x = self.fc5(x)

        return x

class PointNetShapeServoKp3(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNetShapeServoKp3, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel=128 + 3, mlp=[256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[512], bandwidth = 0.4, group_all=True)    

        # self.sa1 = PointConvDensitySetAbstraction(npoint=32, nsample=4, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=16, nsample=8, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)            

        self.fc1 = nn.Linear(512, 256)
        # self.bn1 = nn.BatchNorm1d(128, momentum = 0.5)
        self.bn1 = nn.GroupNorm(1, 256)
        self.drop1 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)


        self.fc3 = nn.Linear(256, 128)
        # self.bn3 = nn.BatchNorm1d(64, momentum = 0.5)
        self.bn3 = nn.GroupNorm(1, 128)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 32)
        # self.bn4 = nn.BatchNorm1d(32, momentum = 0.5)
        self.bn4 = nn.GroupNorm(1, 32)
        self.drop4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(32, 3)


    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 512)


        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:,:3,:]
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        g = l3_points.view(B, 512)


        x = g - x
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))

        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.drop4(F.relu(self.bn4(self.fc4(x))))

        x = self.fc5(x)

        return x        


if __name__ == '__main__':
    # import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,364))
    input2 = torch.randn((8,3,364))
    grasp_pose = torch.randn((8,3))

    # label = torch.randn(8,16)
    model = PointNetShapeServo3(normal_channel=False)
    output= model(input, input2)
    print(output.size())

    # summary(model, (1,6,1024), 8)



