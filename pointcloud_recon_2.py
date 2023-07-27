import torch.nn as nn
import torch
import torch.nn.functional as F
# from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
# from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
# from torchsummary import summary
# from pointconv_util import PointConvDensitySetAbstraction
from pointconv_util_groupnorm import PointConvDensitySetAbstraction


class PointNetShapeServo3(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNetShapeServo3, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
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




class PointNetShapeServo(nn.Module):
    '''
    With manipulation pose
    '''
    def __init__(self, normal_channel=True):
        super(PointNetShapeServo, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)

        self.fc1 = nn.Linear(259, 128)
        # self.bn1 = nn.BatchNorm1d(128, momentum = 0.5)
        self.bn1 = nn.GroupNorm(1, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)


        self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64, momentum = 0.5)
        self.bn3 = nn.GroupNorm(1, 64)
        self.fc4 = nn.Linear(64, 32)
        # self.bn4 = nn.BatchNorm1d(32, momentum = 0.5)
        self.bn4 = nn.GroupNorm(1, 32)
        self.fc5 = nn.Linear(32, 3)
        # self.bn5 = nn.BatchNorm1d(8)
        # self.fc6 = nn.Linear(8, 1)

        # # Subtract goal - current
        # self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64)
        # self.fc4 = nn.Linear(64, 32)
        # self.bn4 = nn.BatchNorm1d(32)
        # self.fc5 = nn.Linear(32, 8)
        # self.bn5 = nn.BatchNorm1d(8)
        # self.fc6 = nn.Linear(8, 1)

    def forward(self, xyz, xyz_goal, grasp_pose, get_feature_vector = False):
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
        if get_feature_vector:
            return x    
                
        x = torch.cat((x, grasp_pose), dim=1)
        # print(x.shape)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)

        return x        


class PointNetShapeServo2(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNetShapeServo2, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)

        # model = PointNetShapeServo(normal_channel=False)
        # model.load_state_dict(torch.load("/home/baothach/shape_servo_data/multi_grasps/weights/batch_1-epoch 32"))
        # self.sa1 = model.sa1
        # self.sa2 = model.sa2
        # self.sa3 = model.sa3
        # for param in self.sa1.parameters():
        #     param.requires_grad = False  
        # for param in self.sa2.parameters():
        #     param.requires_grad = False  
        # for param in self.sa3.parameters():
        #     param.requires_grad = False              

        self.fc1 = nn.Linear(256, 128)
        # self.bn1 = nn.BatchNorm1d(128, momentum = 0.5)
        self.bn1 = nn.GroupNorm(1, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)


        self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64, momentum = 0.5)
        self.bn3 = nn.GroupNorm(1, 64)
        self.fc4 = nn.Linear(64, 32)
        # self.bn4 = nn.BatchNorm1d(32, momentum = 0.5)
        self.bn4 = nn.GroupNorm(1, 32)
        self.fc5 = nn.Linear(32, 3)
        # self.bn5 = nn.BatchNorm1d(8)
        # self.fc6 = nn.Linear(8, 1)

        # # Subtract goal - current
        # self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64)
        # self.fc4 = nn.Linear(64, 32)
        # self.bn4 = nn.BatchNorm1d(32)
        # self.fc5 = nn.Linear(32, 8)
        # self.bn5 = nn.BatchNorm1d(8)
        # self.fc6 = nn.Linear(8, 1)

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



