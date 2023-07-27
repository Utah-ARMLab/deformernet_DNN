import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.append("../")

from pointconv_util_groupnorm import PointConvDensitySetAbstraction


class DeformerNet(nn.Module):
    def __init__(self, normal_channel=False):
        super(DeformerNet, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=3+3+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.GroupNorm(1, 128)
        # self.drop1 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.GroupNorm(1, 64)
        # self.drop3 = nn.Dropout(0.5)    

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.GroupNorm(1, 32)
        # self.drop4 = nn.Dropout(0.5)

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
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 256)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))

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
        # g = F.relu(self.bn1(self.fc1(g)))
        # g = F.relu(self.bn2(self.fc2(g)))        
        
        # x = torch.cat((x, g), dim=1)
        x = g - x
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))

        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)

        return x


class DeformerNetMP(nn.Module):
    def __init__(self, normal_channel=False, use_mp_input=True):
        super(DeformerNetMP, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.use_mp_input = use_mp_input

        if self.use_mp_input:            
            self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=4+3+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        else:
            self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=3+3+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)

        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)

        self.sa1_g = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=3+3+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2_g = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3_g = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.GroupNorm(1, 256)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.GroupNorm(1, 128) 

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.GroupNorm(1, 64)

        self.fc5 = nn.Linear(64, 3)


    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        # # print(l1_points)
        # print("*****", any(np.isnan(l0_points.detach().cpu().numpy().flatten())))
        x = l3_points.view(B, 256)

        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))

        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:,:3,:]
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal
        l1_xyz, l1_points = self.sa1_g(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2_g(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3_g(l2_xyz, l2_points)        
        g = l3_points.view(B, 256)
        # g = F.relu(self.bn1(self.fc1(g)))
        # g = F.relu(self.bn2(self.fc2(g)))        
        
        x = torch.cat((x, g),dim=-1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))


        x = self.fc5(x)

        return x


class DeformerNetManiPoint(nn.Module):
    def __init__(self, normal_channel=True):
        super(DeformerNetManiPoint, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 3, mlp=[256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[512], bandwidth = 0.4, group_all=True)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.GroupNorm(1, 256)
        # self.drop1 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.GroupNorm(1, 128)
        # self.drop3 = nn.Dropout(0.5)    

        self.fc4 = nn.Linear(128,64)
        self.bn4 = nn.GroupNorm(1, 64)
        # self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(64, 3)


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

        x = self.fc5(x)

        return x     

        
           
class DeformerNet2(nn.Module):
    '''
    simpler archiecture
    '''
    def __init__(self, normal_channel=True):
        super(DeformerNet2, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel=32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=64 + 3, mlp=[128], bandwidth = 0.4, group_all=True)   

        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.GroupNorm(1, 64)
        # self.drop1 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.GroupNorm(1, 32)
        # self.drop3 = nn.Dropout(0.5)    

        # self.fc4 = nn.Linear(32, 16)
        # self.bn4 = nn.GroupNorm(1, 16)
        # # self.drop4 = nn.Dropout(0.5)

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
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 128)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))

        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:,:3,:]
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        g = l3_points.view(B, 128)
        # g = F.relu(self.bn1(self.fc1(g)))
        # g = F.relu(self.bn2(self.fc2(g)))        
        
        # x = torch.cat((x, g), dim=1)
        x = g - x
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        # x = F.relu(self.bn4(self.fc4(x)))

        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)

        return x







