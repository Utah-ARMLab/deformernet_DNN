
import torch
import torch.nn as nn


import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
# import sys
# sys.path.append("../")
# from pointcloud_recon_2 import PointNetShapeServo3 as DeformerNet # original partial point cloud

# from architecture import DeformerNet
from architecture import DeformerNetMP as DeformerNet
from dataset_loader import TubeConnectDataset

from torch.utils.tensorboard import SummaryWriter


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
    
        pc = sample["pcs"][0].to(device)
        pc_goal = sample["pcs"][1].to(device)
        target_pos_1 = sample["pos_1"].to(device)
        target_rot_mat_1 = sample["rot_1"].to(device)       
        target_pos_2 = sample["pos_2"].to(device)
        target_rot_mat_2 = sample["rot_2"].to(device)       
        
        optimizer.zero_grad()
        pos_1, rot_mat_1, pos_2, rot_mat_2 = model(pc, pc_goal)

        loss_pos_1 = F.mse_loss(pos_1, target_pos_1)
        loss_rot_1 = model.compute_geodesic_loss(target_rot_mat_1, rot_mat_1)     
        loss_pos_2 = F.mse_loss(pos_2, target_pos_2)
        loss_rot_2 = model.compute_geodesic_loss(target_rot_mat_2, rot_mat_2)
        loss = loss_pos_1 + loss_rot_1 + loss_pos_2 + loss_rot_2
        


        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('====> Epoch: {} Average loss: {:.6f}'.format(
              epoch, train_loss/num_batch))  
    # writer.add_scalar('training loss', train_loss/num_batch, epoch)   




def test(model, device, test_loader, epoch):
    model.eval()
   
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:

            pc = sample["pcs"][0].to(device)
            pc_goal = sample["pcs"][1].to(device)
            target_pos_1 = sample["pos_1"].to(device)
            target_rot_mat_1 = sample["rot_1"].to(device)       
            target_pos_2 = sample["pos_2"].to(device)
            target_rot_mat_2 = sample["rot_2"].to(device)       
            
            pos_1, rot_mat_1, pos_2, rot_mat_2 = model(pc, pc_goal)

            loss_pos_1 = F.mse_loss(pos_1, target_pos_1, reduction='sum')
            loss_rot_1 = model.compute_geodesic_loss(target_rot_mat_1, rot_mat_1)     
            loss_pos_2 = F.mse_loss(pos_2, target_pos_2, reduction='sum')
            loss_rot_2 = model.compute_geodesic_loss(target_rot_mat_2, rot_mat_2)
            loss = loss_pos_1 + loss_rot_1 + loss_pos_2 + loss_rot_2

            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    # writer.add_scalar('test loss',test_loss, epoch)
    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)



if __name__ == "__main__":
    writer = SummaryWriter('runs/PointConv_method')
    
    torch.manual_seed(2021)
    device = torch.device("cuda")


    
    train_len = 10000 #9500
    test_len = 642 #500
    total_len = train_len + test_len

    dataset = TubeConnectDataset(percentage = 1.0)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=40, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=40)


    
    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))

    model = DeformerNet(normal_channel=False).to(device)
    model.apply(weights_init)
    
    weight_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/weights/run1/"
    # model.load_state_dict(torch.load(weight_path + "run1epoch " + str(150)))
  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    for epoch in range(0, 300):
        train(model, device, train_loader, optimizer, epoch)
        # scheduler.step()
        test(model, device, test_loader, epoch)
        
        if epoch % 2 == 0:            
            torch.save(model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch)))

