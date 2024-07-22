import torch.nn as nn
import torch
import torch.nn.functional as F
import sys

sys.path.append("../")

from pointconv_util_groupnorm import PointConvDensitySetAbstraction


class ManiPointSegment(nn.Module):
    def __init__(self, normal_channel=False):
        super(ManiPointSegment, self).__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(
            npoint=512,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64],
            bandwidth=0.1,
            group_all=False,
        )
        self.sa2 = PointConvDensitySetAbstraction(
            npoint=128,
            nsample=64,
            in_channel=64 + 3,
            mlp=[128],
            bandwidth=0.2,
            group_all=False,
        )
        self.sa3 = PointConvDensitySetAbstraction(
            npoint=1,
            nsample=None,
            in_channel=128 + 3,
            mlp=[256],
            bandwidth=0.4,
            group_all=True,
        )

        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.GroupNorm(1, 512)
        self.drop1 = nn.Dropout(0.5)

        # self.fc3 = nn.Linear(512, 512)
        # self.bn3 = nn.GroupNorm(1, 512)
        # self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 1024)
        self.bn4 = nn.GroupNorm(1, 1024)
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(1024, 2048)

    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_points.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_points.shape)

        x = l3_points.view(B, 256)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))

        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:, :3, :]
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        g = l3_points.view(B, 256)
        # g = F.relu(self.bn1(self.fc1(g)))
        # g = F.relu(self.bn2(self.fc2(g)))

        x = torch.cat((x, g), dim=-1)
        # x = g - x

        # x = F.relu(self.bn1(self.fc1(x)))
        # # x = F.relu(self.bn3(self.fc3(x)))
        # x = F.relu(self.bn4(self.fc4(x)))

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        x = x.view(B, 2, 1024)
        x = F.log_softmax(x, dim=1)

        return x


class ManiPointSegment2(nn.Module):
    def __init__(self, normal_channel=False):
        super(ManiPointSegment2, self).__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(
            npoint=512,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64],
            bandwidth=0.1,
            group_all=False,
        )
        self.sa2 = PointConvDensitySetAbstraction(
            npoint=128,
            nsample=64,
            in_channel=64 + 3,
            mlp=[128],
            bandwidth=0.2,
            group_all=False,
        )
        self.sa3 = PointConvDensitySetAbstraction(
            npoint=1,
            nsample=None,
            in_channel=128 + 3,
            mlp=[256],
            bandwidth=0.4,
            group_all=True,
        )

        self.fc1 = nn.Linear(256, 512)
        self.bn1 = nn.GroupNorm(1, 512)
        # self.drop1 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.GroupNorm(1, 512)
        # self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 1024)
        self.bn4 = nn.GroupNorm(1, 1024)
        # self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(1024, 2048)

    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
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
            l0_xyz = xyz_goal[:, :3, :]
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
        x = x.view(B, 2, 1024)
        x = F.log_softmax(x, dim=1)

        return x


class ManiPointSegment3(nn.Module):
    def __init__(self, normal_channel=False):
        super(ManiPointSegment3, self).__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(
            npoint=512,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64],
            bandwidth=0.1,
            group_all=False,
        )
        self.sa2 = PointConvDensitySetAbstraction(
            npoint=128,
            nsample=64,
            in_channel=64 + 3,
            mlp=[128],
            bandwidth=0.2,
            group_all=False,
        )
        self.sa3 = PointConvDensitySetAbstraction(
            npoint=1,
            nsample=None,
            in_channel=128 + 3,
            mlp=[256],
            bandwidth=0.4,
            group_all=True,
        )

        self.conv1 = nn.Conv1d(512, 256, 1)
        self.bn1 = nn.GroupNorm(1, 256)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(256, 2, 1)

        # self.fc1 = nn.Linear(512, 512)
        # self.bn1 = nn.GroupNorm(1, 512)
        # # self.drop1 = nn.Dropout(0.5)

        # # self.fc3 = nn.Linear(512, 512)
        # # self.bn3 = nn.GroupNorm(1, 512)
        # # self.drop3 = nn.Dropout(0.5)

        # self.fc4 = nn.Linear(512, 1024)
        # self.bn4 = nn.GroupNorm(1, 1024)
        # # self.drop4 = nn.Dropout(0.5)

        # self.fc5 = nn.Linear(1024, 2048)

    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_points.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_points.shape)

        x = l3_points.view(B, 256)
        # # x = F.relu(self.bn1(self.fc1(x)))
        # # x = F.relu(self.bn2(self.fc2(x)))

        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:, :3, :]
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points_g = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        g = l3_points.view(B, 256)
        # # g = F.relu(self.bn1(self.fc1(g)))
        # # g = F.relu(self.bn2(self.fc2(g)))

        # x = torch.cat([l2_points, l2_points_g], 1)
        x = torch.cat((x, g), dim=-1).view(B, -1, 1)
        x = x.repeat(1, 1, 1024)
        # print(x.shape)

        feat = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        # return x

        return x


if __name__ == "__main__":

    device = torch.device("cuda")  # cuda
    pc = torch.randn((8, 3, 1024)).float().to(device)
    pc_goal = torch.randn((8, 3, 1024)).float().to(device)
    # labels = torch.randn((8,16)).float().to(device)
    model = ManiPointSegment3().to(device)
    # out = model(pc)
    out = model(pc, pc_goal)
    # print(out[0].shape)
    # print(out[1].shape)
    # print(out)
    print(out.shape)
