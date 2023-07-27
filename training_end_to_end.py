import torch
import torch.nn as nn


import torch.optim as optim
import torch.nn.functional as F
from pointcloud_recon_2 import PointNetShapeServo
from dataset_loader import PointNetShapeServoDataset
# from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
    
        pc = sample["input"][0].to(device)
        pc_goal = sample["input"][1].to(device)
        target = sample["target"].to(device)
        
        optimizer.zero_grad()
        output = model(pc, pc_goal)

        loss = F.mse_loss(output, target)
    
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('====> Epoch: {} Average loss: {:.6f}'.format(
              epoch, train_loss / 1))  
    writer.add_scalar('training loss', train_loss/num_batch, epoch)   


# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     train_loss = 0
#     num_batch = 0
#     for batch_idx, sample in enumerate(train_loader):
#         total = 0
#         num_batch += 1
#         for j in range(len(sample['input'])):
            
            
#             pc = sample["input"][j][0].unsqueeze(0).to(device)
#             pc_goal = sample["input"][j][1].unsqueeze(0).to(device)
#             target = sample["target"][j].unsqueeze(0).to(device)
            
#             optimizer.zero_grad()
#             output = model(pc, pc_goal)

#             loss = F.mse_loss(output, target)
#             total += loss
#         total.backward()
#         train_loss += loss.item()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(sample), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#     print('====> Epoch: {} Average loss: {:.6f}'.format(
#               epoch, train_loss / 1))  
#     writer.add_scalar('training loss', train_loss/num_batch, epoch)  


def test(model, device, test_loader, epoch):
    model.eval()
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.track_runing_stats=False    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            pc = sample["input"][0].to(device)
            pc_goal = sample["input"][1].to(device)
            target = sample["target"].to(device)
            output = model(pc, pc_goal)
            # print("test", output.reshape(1,-1).cpu().detach().numpy(), target.reshape(1,-1).cpu().detach().numpy())
            # print(target, "--------", output)
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            # test_loss += F.mse_loss(output, target).item()
            # print("test loss:", test_loss)
    test_loss /= len(test_loader.dataset)
    writer.add_scalar('test loss',test_loss, epoch)
    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

# def my_collate(batch):
#     # print(batch)
#     # print("------------------")
#     data = [item['input'] for item in batch]
#     # pc = [item['input'][0] for item in batch]
#     # pc_goal = [item['input'][1] for item in batch]
#     target = [item['target'] for item in batch]
#     # print(data)
#     # print("------------------")
#     # print(data[0])
#     # target = torch.LongTensor(target)
#     return {'input': data, 'target': target}

if __name__ == "__main__":
    writer = SummaryWriter('runs/PointConv_method')
    
    torch.manual_seed(2021)
    device = torch.device("cuda")

    train_dataset = PointNetShapeServoDataset(percentage = 0.6) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = PointNetShapeServoDataset(percentage = 0.4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    print("training data: ", len(train_dataset.pc))
    print("test data: ", len(test_dataset.pc))

    model = PointNetShapeServo(normal_channel=False).to(device)
    model.apply(weights_init)
    
    
    # model.load_state_dict(torch.load("/home/baothach/shape_servo_data/weights/PointNet/batch_1d"))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for epoch in range(1, 51):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        if epoch % 10 == 0:
            test(model, device, test_loader, epoch)
            torch.save(model.state_dict(), "/home/baothach/shape_servo_data/multi_grasps/weights/one_grasp")

