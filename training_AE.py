import torch
import torch.optim as optim
import torch.nn.functional as F
from pointcloud_recon import PointCloudAE
from pointcloud_recon_3 import PointCloudAE_v2
from dataset_loader import PointCloudAEDataset


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        data = sample.to(device)
    
        
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = F.mse_loss(data, recon_batch)
        # loss = chamfer_distance(torch.swapaxes(data, 1, 2), torch.swapaxes(recon_batch, 1, 2))
        # loss = model.get_loss(torch.swapaxes(data, 1, 2), torch.swapaxes(recon_batch, 1, 2))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('====> Epoch: {} Average loss: {:.6f}'.format(
              epoch, train_loss / len(train_loader.dataset)))  

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data = sample.to(device)
            recon_batch = model(data)
            test_loss += F.mse_loss(data, recon_batch).item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss))



if __name__ == "__main__":
    torch.manual_seed(2021)
    device = torch.device("cuda")

    train_dataset = PointCloudAEDataset(percentage = 1.0) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

    test_dataset = PointCloudAEDataset(percentage = .2)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    
    print("training data: ", len(train_dataset.pc))
    print("test data: ", len(test_dataset.pc))

    model = PointCloudAE_v2(normal_channel=False).to(device)
    model.load_state_dict(torch.load("/home/baothach/shape_servo_data/weights/AE_weights_only_one_chamfer_new"))
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    for epoch in range(1, 101):
        train(model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        # scheduler.step()


    torch.save(model.state_dict(), "/home/baothach/shape_servo_data/weights/AE_weights_only_one_chamfer_new")

