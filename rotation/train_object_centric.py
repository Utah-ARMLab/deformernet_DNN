import torch
import torch.nn as nn


import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
import logging
import socket

# import sys
# sys.path.append("../")
# from pointcloud_recon_2 import PointNetShapeServo3 as DeformerNet # original partial point cloud

# from architecture import DeformerNet
from architecture_2 import DeformerNetMP as DeformerNet
from dataset_loader_single_box import SingleBoxDataset

# from torch.utils.tensorboard import SummaryWriter


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1

        pc = sample["pcs"][0].to(device)
        pc_goal = sample["pcs"][1].to(device)
        target_pos = sample["pos"].to(device)
        target_rot_mat = sample["rot"].to(device)

        optimizer.zero_grad()
        pos, rot_mat = model(pc, pc_goal)

        loss_pos = F.mse_loss(pos, target_pos)
        loss_rot = model.compute_geodesic_loss(target_rot_mat, rot_mat) * 176
        loss = loss_pos + loss_rot

        # # print(rot_mat)
        # # print(target_rot_mat)
        # test = pos.detach().cpu().numpy()
        # # print("*****", any(np.isnan(test.flatten())))
        # print(loss_pos.detach().cpu().numpy(), loss_rot.detach().cpu().numpy())
        # print("===============")
        # print(loss_pos.detach().cpu().numpy())
        # print(loss_rot.detach().cpu().numpy())
        # print("=============")

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(sample),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

            print(
                "loss pos, loss rot, ratio loss_rot/loss_pos:",
                int(loss_pos.detach().cpu().numpy()),
                int(loss_rot.detach().cpu().numpy()),
                loss_rot.detach().cpu().numpy() / loss_pos.detach().cpu().numpy(),
            )  # ratio should be ~= 1/3

    print("====> Epoch: {} Average loss: {:.6f}".format(epoch, train_loss / num_batch))
    logger.info("Train: Average loss: {:.6f}".format(train_loss / num_batch))
    logger.info(
        "loss pos: {}, loss rot: {}, ratio: {}".format(
            int(loss_pos.detach().cpu().numpy()),
            int(loss_rot.detach().cpu().numpy()),
            loss_rot.detach().cpu().numpy() / loss_pos.detach().cpu().numpy(),
        )
    )


def test(model, device, test_loader, epoch):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:

            pc = sample["pcs"][0].to(device)
            pc_goal = sample["pcs"][1].to(device)
            target_pos = sample["pos"].to(device)
            target_rot_mat = sample["rot"].to(device)

            pos, rot_mat = model(pc, pc_goal)

            loss_pos = F.mse_loss(pos, target_pos, reduction="sum")
            loss_rot = model.compute_geodesic_loss(target_rot_mat, rot_mat)

            # print(loss_pos.detach().cpu().numpy())
            # print(loss_rot.detach().cpu().numpy())
            # print("=============")

            loss = loss_pos + loss_rot * 1000
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    # writer.add_scalar('test loss',test_loss, epoch)
    print("\nTest set: Average loss: {:.6f}\n".format(test_loss))
    logger.info("Test: Average loss: {:.6f}\n".format(test_loss))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    # writer = SummaryWriter('runs/PointConv_method')

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--use_mp_input", default="True", type=str, help="use MP as an input to the NN"
    )
    parser.add_argument(
        "--obj_category",
        default="None",
        type=str,
        help="object category. Ex: box_10kPa",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="batch size for training and testing",
    )
    args = parser.parse_args()

    use_mp_input = args.use_mp_input == "True"

    if use_mp_input:
        weight_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{args.obj_category}/weights/object_centric/run1_w_rot_w_MP/"
    else:
        weight_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{args.obj_category}/weights/object_centric/run1_w_rot_no_MP/"
    # weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinder_10kPa/weights/thin_cylinder"
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

    dataset_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{args.obj_category}/processed_data_w_mp"
    # dataset_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinder_10kPa/single_thin_cylinder_10kPa_processed_data_w_mp"
    train_len = round(len(os.listdir(dataset_path)) * 0.95)  # 11000
    test_len = round(len(os.listdir(dataset_path)) * 0.05)  # 1000
    total_len = train_len + test_len

    dataset = SingleBoxDataset(
        percentage=1.0,
        use_mp_input=use_mp_input,
        dataset_path=dataset_path,
        shift_to_centroid=True,
    )
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
    print("Using MP input: ", use_mp_input)
    logger.info(f"Train len: {len(train_dataset)}")
    logger.info(f"Test len: {len(test_dataset)}")
    logger.info(f"Data path: {dataset.dataset_path}")
    logger.info(f"Using MP input: {use_mp_input}\n")

    model = DeformerNet(normal_channel=False, use_mp_input=use_mp_input).to(device)
    model.apply(weights_init)
    # model.load_state_dict(torch.load(weight_path + "epoch " + str(8)))

    num_epoch_total = 200 if use_mp_input else 160
    scheduler_step = 100 if use_mp_input else 80
    # num_epoch_total = 160
    # scheduler_step = 80

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=0.1)

    for epoch in range(0, num_epoch_total + 1):
        logger.info(f"Epoch {epoch}")
        logger.info(f"Lr: {optimizer.param_groups[0]['lr']}")
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)

        if epoch % 2 == 0:
            torch.save(
                model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch))
            )
