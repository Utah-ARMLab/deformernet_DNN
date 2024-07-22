import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet2_utils_groupnorm import PointNetSetAbstraction, PointNetFeaturePropagation


class ManiPointSegment(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(ManiPointSegment, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=64 + 3,
            mlp=[128],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=128 + 3,
            mlp=[256],
            group_all=True,
        )

        self.fp3 = PointNetFeaturePropagation(in_channel=128 + 256, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=64 + 128, mlp=[64])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=64 + additional_channel, mlp=[64, 64]
        )

        # self.sa1_g = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64], group_all=False)
        # self.sa2_g = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=64 + 3, mlp=[128], group_all=False)
        # self.sa3_g = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[256], group_all=True)

        # self.fp3_g = PointNetFeaturePropagation(in_channel=128+256, mlp=[128])
        # self.fp2_g = PointNetFeaturePropagation(in_channel=64+128, mlp=[64])
        # self.fp1_g = PointNetFeaturePropagation(in_channel=64+additional_channel, mlp=[64, 64])

        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.GroupNorm(1, 64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

        # self.fp3 = PointNetFeaturePropagation(in_channel=(128+256)*2, mlp=[256])
        # self.fp2 = PointNetFeaturePropagation(in_channel=(64+128)*2, mlp=[128])
        # self.fp1 = PointNetFeaturePropagation(in_channel=(64+additional_channel)*2, mlp=[128, 128])

        # self.conv1 = nn.Conv1d(128, 128, 1)
        # self.bn1 = nn.GroupNorm(1, 128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

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
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        print(l1_points.shape, l2_points.shape, l3_points.shape)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        print(l2_points.shape)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        if self.normal_channel:
            l0_points_g = xyz_goal
            l0_xyz_g = xyz_goal[:, :3, :]
        else:
            l0_points_g = xyz_goal
            l0_xyz_g = xyz_goal
        l1_xyz_g, l1_points_g = self.sa1(l0_xyz_g, l0_points_g)
        l2_xyz_g, l2_points_g = self.sa2(l1_xyz_g, l1_points_g)
        l3_xyz_g, l3_points_g = self.sa3(l2_xyz_g, l2_points_g)

        l2_points_g = self.fp3(l2_xyz_g, l3_xyz_g, l2_points_g, l3_points_g)
        l1_points_g = self.fp2(l1_xyz_g, l2_xyz_g, l1_points_g, l2_points_g)
        l0_points_g = self.fp1(l0_xyz_g, l1_xyz_g, None, l1_points_g)

        # print(l0_points.shape)
        x = torch.cat([l0_points, l0_points_g], 1)
        print(x.shape)

        # print(l2_points.shape)
        # print(l1_points.shape)
        # print(l0_points.shape)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x
        # return l3_points
        return x, l3_points


if __name__ == "__main__":

    num_classes = 2
    device = torch.device("cuda")  # cuda
    pc = torch.randn((8, 3, 1024)).float().to(device)
    pc_goal = torch.randn((8, 3, 1024)).float().to(device)
    # labels = torch.randn((8,16)).float().to(device)
    model = ManiPointSegment(num_classes).to(device)
    # out = model(pc)
    out = model(pc, pc_goal)
    # print(out[0].shape)
    # print(out[1].shape)
    print(out.shape)
    # print(out)

    # pc2 = pc[:,torch.randperm(pc.size()[1])]
    # out2 = model(pc, pc_goal)
    # print(out - out2)
