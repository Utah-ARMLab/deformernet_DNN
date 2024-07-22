import open3d
import os
import numpy as np
import pickle
import torch
from architecture_classifier import ManiPointNet2
from sklearn.neighbors import NearestNeighbors
from farthest_point_sampling import *
from copy import copy, deepcopy


def down_sampling(pc):
    farthest_indices, _ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]
    return pc


device = torch.device("cuda")
model = ManiPointNet2(normal_channel=False).to(device)
# weight_path = "/home/baothach/shape_servo_data/manipulation_points/box/weights/classifer/run2(simpler)"
weight_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/weights/classifer/run1"
model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 300")))
model.eval()


# data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/box/data"
# data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/data2"
data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/heatmaps/goal_data"


file_names = sorted(os.listdir(data_recording_path))


i = 545
num_candidates = 400
batch_size = 64
# with open(os.path.join(data_recording_path, file_names[9000]), 'rb') as handle:
#     data_2 = pickle.load(handle)
# for i in [500, 515, 530, 545, 560, 585, 620]:
# for i in [525, 535]:
# for i in [620]:
# for i in [500, 515, 525, 530, 545, 560, 585, 620]:
for i in range(0, 2):
    # for i in list(np.random.randint(low=9000, high=9379, size=11)):
    file_name = os.path.join(data_recording_path, file_names[i])

    with open(file_name, "rb") as handle:
        data = pickle.load(handle)
    pc = down_sampling(data["partial pcs"][0])

    pc_goal = down_sampling(data["partial pcs"][1])
    pc_goal_tensor = (
        torch.from_numpy(pc_goal).permute(1, 0).unsqueeze(0).float().to(device)
    )
    # gt_mp = np.array(list(data["mani_point"]["pose"][0]))
    gt_mp = np.array(list(data["mani_point"][0]))

    # mp_candidates_idxs = np.where(pc[:,1] >= -0.42)[0]
    mp_candidates_idxs = np.where(pc[:, 1] >= -42)[0]
    mp_candidates_idxs = np.random.choice(
        mp_candidates_idxs, size=num_candidates
    )  # sub-sample 32 candidates
    mp_candidates = pc[mp_candidates_idxs]  # points on the correct half
    # print(mp_candidates.shape)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(pc))
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(np.array(pc_goal))
    print(
        "Gt chamfer:",
        np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd))),
    )
    # open3d.visualization.draw_geometries([pcd, deepcopy(pcd_goal).translate((0.15,0,0))])

    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pc)
    _, nearest_idxs = neigh.kneighbors(mp_candidates)
    mp_channel = np.zeros((mp_candidates.shape[0], pc.shape[0]))
    # print(np.array([i // 50 for i in range(mp_candidates.shape[0]*50)]))
    mp_channel[
        np.array([i // 50 for i in range(mp_candidates.shape[0] * 50)]),
        nearest_idxs.flatten(),
    ] = 1

    pcs_tensor = (
        torch.from_numpy(pc)
        .permute(1, 0)
        .unsqueeze(0)
        .repeat(mp_candidates.shape[0], 1, 1)
    )
    modified_pc_tensor = (
        torch.cat((pcs_tensor, torch.from_numpy(mp_channel).unsqueeze(1)), dim=1)
        .float()
        .to(device)
    )

    pcs_goal_tensor = pc_goal_tensor.repeat(mp_candidates.shape[0], 1, 1)
    # print(modified_pc_tensor.shape, pcs_goal_tensor.shape)
    with torch.no_grad():
        outputs = []
        for batch_pc, batch_pc_goal in zip(
            torch.split(modified_pc_tensor, num_candidates // batch_size),
            torch.split(pcs_goal_tensor, num_candidates // batch_size),
        ):
            outputs.append(model(batch_pc, batch_pc_goal))
            # outputs.append(model(modified_pc_tensor, pcs_goal_tensor))
        output = torch.cat(tuple(outputs), dim=0)
        # print(output.shape)
        success = output.argmax(dim=1, keepdim=True)
        # print("output:", np.exp(output.cpu().detach().numpy()))
        # print("class:", success.cpu().detach().numpy())

        success_probs = np.exp(output.cpu().detach().numpy())[:, 1]
        success_probs = success_probs / max(success_probs)
        # print("success_prob:",success_probs)
        heats = np.array([[prob, 0, 0] for prob in success_probs])

        best_mp = mp_candidates[np.argmax(success_probs)]
        best_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        best_mani_point.paint_uniform_color([0, 1, 0])

        colors = np.zeros(pc.shape)
        colors[mp_candidates_idxs] = heats
        pcd.colors = open3d.utility.Vector3dVector(colors)

        mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mani_point.paint_uniform_color([0, 0, 1])
        # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), pcd_goal.translate((0.2,0,0))])
        open3d.visualization.draw_geometries(
            [
                pcd,
                mani_point.translate(tuple(gt_mp)),
                best_mani_point.translate(tuple(best_mp)),
                pcd_goal.translate((0.2, 0, 0)),
            ]
        )


# with open(file_name, 'rb') as handle:
#     data = pickle.load(handle)
# pc = down_sampling(data["partial pcs"][0])
# pc_goal = down_sampling(data["partial pcs"][1])
# pc_goal_tensor = torch.from_numpy(pc_goal).permute(1,0).unsqueeze(0).float().to(device)
# mp_pos = np.array(list(data["mani_point"]["pose"][0])) + np.array([-0.08,0.01,0])

# neigh = NearestNeighbors(n_neighbors=50)
# neigh.fit(pc)
# _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
# mp_channel = np.zeros(pc.shape[0])
# mp_channel[nearest_idxs.flatten()] = 1
# modified_pc = np.vstack([pc.transpose(1,0), mp_channel])# pc with 4th channel with value of 1 if near the MP point and 0 elsewhere
# modified_pc_tensor = torch.from_numpy(modified_pc).unsqueeze(0).float().to(device)

# print(modified_pc_tensor.shape, pc_goal_tensor.shape)
# output = model(modified_pc_tensor, pc_goal_tensor)
# success = output.argmax(dim=1, keepdim=True)
# print("output:", np.exp(output.cpu().detach().numpy()))
# print("class:", success.cpu().detach().numpy())

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(np.array(pc))
# colors = np.zeros((1024,3))
# colors[nearest_idxs.flatten()] = [1,0,0]
# pcd.colors =  open3d.utility.Vector3dVector(colors)
# pcd_goal = open3d.geometry.PointCloud()
# pcd_goal.points = open3d.utility.Vector3dVector(np.array(pc_goal))
# print("Gt chamfer:", np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd))))

# mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
# mani_point.paint_uniform_color([0,0,1])
# open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(mp_pos)), pcd_goal.translate((0.15,0,0))])
