from numpy.core.defchararray import translate
import open3d
import os
import numpy as np
import pickle
import torch
from architecture_seg import ManiPointSegment
from sklearn.neighbors import NearestNeighbors
from farthest_point_sampling import *
from copy import copy, deepcopy


def down_sampling(pc):
    farthest_indices, _ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]
    return pc


device = torch.device("cuda")
# model = ManiPointSegment(num_classes=2).to(device)
# # weight_path = "/home/baothach/shape_servo_data/manipulation_points/box/weights/segmentation/run3_augmented"
# weight_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/weights/seg/run1"
# model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 150")))
# model.eval()

# from architecture_seg_2 import ManiPointSegment3 as ManiPointSegment
# model = ManiPointSegment().to(device)
from test_pointconv import ManiPointSegment

model = ManiPointSegment(num_classes=2).to(device)
weight_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/weights/seg/run5"
model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 26")))
model.eval()


# data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/box/data"
data_recording_path = (
    "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/data2"
)
# data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/heatmaps/goal_data"
# data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/evaluate_success/goal_data/box_1k/inside"
# data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/evaluate_success/goal_data/box_1k/inside"

file_names = sorted(os.listdir(data_recording_path))


i = 545
num_candidates = 400
batch_size = 64
# with open(os.path.join(data_recording_path, file_names[9171]), 'rb') as handle:
#     data_2 = pickle.load(handle)
# for i in [500, 515, 530, 545, 560, 585, 620]:
# for i in [525, 535]:
# for i in [5001]:
# for i in [500, 515, 525, 530, 545, 560, 585, 620]:
# for i in range(8500, 8511):
# for i in list(np.random.randint(low=8000, high=9379, size=11)):
for i in range(0, 20):
    file_name = os.path.join(data_recording_path, file_names[i])
    print(file_name)

    with open(file_name, "rb") as handle:
        data = pickle.load(handle)  # [0]

    pc = down_sampling(data["partial pcs"][0])
    # pc = down_sampling(data[0]["partial pcs"])
    pc_tensor = torch.from_numpy(pc).permute(1, 0).unsqueeze(0).float().to(device)

    # pcd = open3d.io.read_point_cloud("/home/baothach/shape_servo_data/manipulation_points/box/init_box_pc.pcd")
    # pc = down_sampling(np.asarray(pcd.points))
    translation = np.array([0, 0, 0])
    pc = np.array([p + translation for p in pc])
    pc = pc[np.random.permutation(pc.shape[0])]
    pc_tensor = torch.from_numpy(pc).permute(1, 0).unsqueeze(0).float().to(device)

    pc_goal = down_sampling(data["partial pcs"][1])
    # pc_goal = down_sampling(data[0]["partial pcs"][1])
    pc_goal = np.array([p + translation for p in pc_goal])
    pc_goal = pc_goal[np.random.permutation(pc_goal.shape[0])]
    pc_goal_tensor = (
        torch.from_numpy(pc_goal).permute(1, 0).unsqueeze(0).float().to(device)
    )
    gt_mp = np.array(list(data["mani_point"][0])) + translation
    # gt_mp = np.array(list(data[0]["mani_point"][0])) + translation

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(pc))
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(np.array(pc_goal))
    print(
        "Gt chamfer:",
        np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd))),
    )
    # open3d.visualization.draw_geometries([pcd, deepcopy(pcd_goal).translate((0.15,0,0))])

    with torch.no_grad():

        output = model(pc_tensor, pc_goal_tensor)

        print(output)
        # print(output.shape)
        # success = output.argmax(dim=1, keepdim=True)
        # print("output:", np.exp(output.cpu().detach().numpy()))
        # print("class:", success.cpu().detach().numpy())

        success_probs = np.exp(output.squeeze().cpu().detach().numpy())[1, :]
        print(success_probs.shape)
        print(max(success_probs), min(success_probs))
        # print("num of candidates:", sum(1 for i in success_probs if i >= 0.5))
        # success_probs = [1 if prob >= 0.8 else 0 for prob in success_probs]

        success_probs = success_probs / max(success_probs)
        # print("success_prob:",success_probs)
        heats = np.array([[prob, 0, 0] for prob in success_probs])

        best_mp = pc[np.argmax(success_probs)]
        best_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        best_mani_point.paint_uniform_color([0, 1, 0])

        colors = heats
        pcd.colors = open3d.utility.Vector3dVector(colors)

        mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mani_point.paint_uniform_color([0, 0, 1])
        # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), pcd_goal.translate((0.2,0,0))])
        # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), \
        #                                         pcd_goal.translate((0.2,0,0))])
        open3d.visualization.draw_geometries(
            [
                pcd,
                mani_point.translate(tuple(gt_mp)),
                best_mani_point.translate(tuple(best_mp)),
                pcd_goal.translate((0.2, 0, 0)),
                open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1),
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
