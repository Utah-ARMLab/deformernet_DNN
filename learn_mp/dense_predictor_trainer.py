import torch
import torch.nn as nn


import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
import logging
import socket

# import sys
# sys.path.append("../")
# from pointcloud_recon_2 import PointNetShapeServo3 as DeformerNet # original partial point cloud

# from architecture_seg import ManiPointSegment
# from architecture_seg_2 import ManiPointSegment3 as ManiPointSegment
from dense_predictor_pointconv_architecture import DensePredictor
from dataset_loader_mani_point import DensePredictorDataset

from torch.utils.tensorboard import SummaryWriter


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1

        pc = sample["pcs"][0].to(device)
        pc_goal = sample["pcs"][1].to(device)
        target = sample["label"].to(device)

        optimizer.zero_grad()
        output = model(pc, pc_goal)

        loss = F.nll_loss(output, target)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(sample),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    print("====> Epoch: {} Average loss: {:.6f}".format(epoch, train_loss / num_batch))
    logger.info("Train: Average loss: {:.6f}".format(train_loss / num_batch))


def test(model, device, test_loader, epoch):
    model.eval()

    test_loss = 0
    correct = 0
    positive_correct = 0
    with torch.no_grad():
        for sample in test_loader:

            pc = sample["pcs"][0].to(device)
            pc_goal = sample["pcs"][1].to(device)
            target = sample["label"].to(device)

            output = model(pc, pc_goal)

            test_loss += F.nll_loss(output, target, reduction="sum").item()

            pred = output.argmax(
                dim=1, keepdim=True
            ).squeeze()  # get the index of the max log-probability
            correct += torch.sum(pred == target)

    test_loss /= len(test_loader.dataset) * 1024
    # writer.add_scalar('test loss',test_loss, epoch)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset) * 1024,
            100.0 * correct / (len(test_loader.dataset) * 1024),
        )
    )
    logger.info("Test: Average loss: {:.6f}".format(test_loss))
    logger.info(
        "Test: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(test_loader.dataset) * 1024,
            100.0 * correct / (len(test_loader.dataset) * 1024),
        )
    )
    # print('\ntotal correct positive:', positive_correct )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--obj_category",
        default="None",
        type=str,
        help="object category. Ex: boxes_10kPa",
    )
    parser.add_argument(
        "--batch_size", default=40, type=int, help="batch size for training and testing"
    )
    args = parser.parse_args()

    weight_path = f"/home/baothach/shape_servo_data/manipulation_points/multi_{args.obj_category}/weights/seg/run1"
    os.makedirs(weight_path, exist_ok=True)

    logger = logging.getLogger(weight_path)
    logger.propagate = False  # no output to console
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(weight_path, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Machine: {socket.gethostname()}")

    torch.manual_seed(2021)
    device = torch.device("cuda")

    dataset_path = f"/home/baothach/shape_servo_data/manipulation_points/multi_{args.obj_category}/processed_seg_data"
    train_len = round(len(os.listdir(dataset_path)) * 0.95)
    test_len = round(len(os.listdir(dataset_path)) * 0.05)
    total_len = train_len + test_len

    dataset = DensePredictorDataset(percentage=1.0, dataset_path=dataset_path)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))
    print("data path:", dataset.dataset_path)
    logger.info(f"Train len: {len(train_dataset)}")
    logger.info(f"Test len: {len(test_dataset)}")
    logger.info(f"Data path: {dataset.dataset_path}\n")

    model = DensePredictor(num_classes=2).to(device)  # single robot
    # model = ManiPointSegment(num_classes=3).to(device)  # bimanual robot
    model.apply(weights_init)
    # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch " + str(86))))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1)
    for epoch in range(0, 161):
        logger.info(f"Epoch {epoch}")
        logger.info(f"Lr: {optimizer.param_groups[0]['lr']}")
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)

        if epoch % 2 == 0:
            torch.save(
                model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch))
            )
