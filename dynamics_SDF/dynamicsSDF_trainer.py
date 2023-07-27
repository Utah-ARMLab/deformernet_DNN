
import torch
import torch.nn as nn


import torch.optim as optim
import torch.nn.functional as F

# import sys
# sys.path.append("../")
# from pointcloud_recon_2 import PointNetShapeServo3 as DeformerNet # original partial point cloud

from architecture import TransitionSDF
from dataset_loader_dynamicsSDF import SingleBoxSDFDataset

from torch.utils.tensorboard import SummaryWriter


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
    
        state = sample["state"].to(device)
        next_state = sample["next_state"].to(device)
        action = sample["action"].to(device)       
  
        
        optimizer.zero_grad()
        output = model(state, action)

        loss = F.mse_loss(output, next_state)
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
           
            state = sample["state"].to(device)
            next_state = sample["next_state"].to(device)
            action = sample["action"].to(device)     
            
            output = model(state, action)

            test_loss += F.mse_loss(output, next_state, reduction='sum').item()

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


    
    train_len = 9500
    test_len = 500
    total_len = train_len + test_len

    dataset = SingleBoxSDFDataset(percentage = 1.0)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    
    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))

    model = TransitionSDF().to(device)
    model.apply(weights_init)
    
    weight_path = "/home/baothach/shape_servo_data/RL_shapeservo/box/weights_SDF_dynamics/run2/"
    # model.load_state_dict(torch.load(weight_path + "epoch " + str(128)))
  
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for epoch in range(0, 151):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)
        
        if epoch % 2 == 0:            
            torch.save(model.state_dict(), weight_path + "epoch " + str(epoch))

