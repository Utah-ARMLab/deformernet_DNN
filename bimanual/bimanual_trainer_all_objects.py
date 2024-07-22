import torch
import torch.nn as nn


import torch.optim as optim
import torch.nn.functional as F
import os
from itertools import product
import random


from bimanual_architecture import DeformerNetBimanualRot
from dataset_loader import SingleBoxDatasetAllObjects

from torch.utils.tensorboard import SummaryWriter

import argparse
import logging
import socket
import timeit


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1

        pc = sample["pcs"][0].to(device)
        pc_goal = sample["pcs"][1].to(device)
        target_pos = sample["pos"].to(device)
        target_rot_mat_1 = sample["rot_1"].to(device)
        target_rot_mat_2 = sample["rot_2"].to(device)

        # print(target_pos[0])

        optimizer.zero_grad()
        pos, rot_mat_1, rot_mat_2 = model(pc, pc_goal)

        loss_pos = F.mse_loss(pos, target_pos)
        loss_rot_1 = model.compute_geodesic_loss(target_rot_mat_1, rot_mat_1)
        loss_rot_2 = model.compute_geodesic_loss(target_rot_mat_2, rot_mat_2)

        loss_rot = (loss_rot_1 + loss_rot_2) * 48  # make loss_pos = 10 * loss_rot

        loss = loss_pos + loss_rot

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
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
            )  # ratio should be ~= 1/10

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

    with torch.no_grad():
        for sample in test_loader:

            pc = sample["pcs"][0].to(device)
            pc_goal = sample["pcs"][1].to(device)
            target_pos = sample["pos"].to(device)
            target_rot_mat_1 = sample["rot_1"].to(device)
            target_rot_mat_2 = sample["rot_2"].to(device)

            pos, rot_mat_1, rot_mat_2 = model(pc, pc_goal)

            loss_pos = F.mse_loss(pos, target_pos, reduction="sum")
            loss_rot_1 = model.compute_geodesic_loss(target_rot_mat_1, rot_mat_1)
            loss_rot_2 = model.compute_geodesic_loss(target_rot_mat_2, rot_mat_2)
            loss = loss_pos + loss_rot_1 * 1000 + loss_rot_2 * 1000

            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
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

    use_mp_input = False

    parser = argparse.ArgumentParser(description=None)

    # parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: box_10kPa")
    # parser.add_argument('--batch_size', default=128, type=int, help="batch size for training and testing")
    args = parser.parse_args()
    args.batch_size = 180

    weight_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/all_objects/weights/test"
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

    torch.manual_seed(2022)
    random.seed(2022)
    device = torch.device("cuda")

    prim_names = ["box", "cylinder", "hemis"]
    stiffnesses = ["1k", "5k", "10k"]
    object_names = [
        f"{prim_name}_{stiffness}Pa"
        for (prim_name, stiffness) in list(product(prim_names, stiffnesses))
    ]

    dataset_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual"
    dataset = SingleBoxDatasetAllObjects(dataset_path, object_names, use_mp_input)

    train_len = 50000  # round(len(dataset)*0.9)
    test_len = 1000  # round(len(dataset)*0.1)
    total_len = train_len + test_len

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

    logger.info(f"\nObject list: {object_names}\n")
    logger.info(f"Train len: {len(train_dataset)}")
    logger.info(f"Test len: {len(test_dataset)}")
    logger.info(f"Data path: {dataset.dataset_path}")
    logger.info(f"Using MP input: {use_mp_input}\n")

    # model = DeformerNetBimanual(normal_channel=False).to(device)
    # model = DeformerNetTube(normal_channel=False).to(device)
    model = DeformerNetBimanualRot(normal_channel=False, use_mp_input=use_mp_input).to(
        device
    )
    model.apply(weights_init)

    # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch " + str(80))))

    num_epoch_total = 160  # 200
    scheduler_step = 80  # 100

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=0.1)

    start_time = timeit.default_timer()
    for epoch in range(0, num_epoch_total + 1):
        logger.info(f"\n")
        logger.info(f"Epoch {epoch}")
        logger.info(f"Lr: {optimizer.param_groups[0]['lr']}")
        print(f"\n================ Epoch {epoch}")
        print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n")
        logger.info(
            f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n"
        )

        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch))
            )
