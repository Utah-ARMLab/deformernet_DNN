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
for i in range(0, 9):
    # for i in list(np.random.randint(low=5000, high=9379, size=11)):
    print("i:", i)

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

    full_pc = data["full pcs"][0]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(pc))
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(np.array(pc_goal))
    print(
        "Gt chamfer:",
        np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd))),
    )
    # open3d.visualization.draw_geometries([pcd, deepcopy(pcd_goal).translate((0.15,0,0))])

    full_pcd = open3d.geometry.PointCloud()
    full_pcd.points = open3d.utility.Vector3dVector(np.array(full_pc))
    obb = full_pcd.get_oriented_bounding_box()
    center = full_pcd.get_center()

    # Get width, height, depth
    points = np.asarray(obb.get_box_points())
    x_axis = points[1] - points[0]
    y_axis = points[2] - points[0]
    z_axis = points[3] - points[0]
    width = np.linalg.norm(
        x_axis
    )  # Length of x axis (https://www.cs.utah.edu/gdc/projects/alpha1/help/man/html/shape_edit/primitives.html)
    height = np.linalg.norm(y_axis)
    depth = np.linalg.norm(z_axis)

    if 0.9 * height <= width <= 1.1 * height:
        x_axis, y_axis = y_axis, x_axis

    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # print("width, height, depth:", width, height, depth)
    # print("x_axis, y_axis, z_axis:", x_axis, y_axis, z_axis)

    m = y_axis[1] / y_axis[0]
    if abs(m) >= 1:
        m = y_axis[0] / y_axis[1]
    b = center[1] - m * center[0]

    mp_candidates_idxs = np.where(m * pc[:, 0] + b - pc[:, 1] <= 0)[0]
    # mp_candidates_idxs = np.where(pc[:,1] >= -42)[0]
    mp_candidates_idxs = np.random.choice(
        mp_candidates_idxs, size=num_candidates
    )  # sub-sample 32 candidates
    mp_candidates = pc[mp_candidates_idxs]  # points on the correct half

    # origin = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # # origin.paint_uniform_color([0,0,1])

    # center_pcd = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # center_pcd.paint_uniform_color([0.5,0.5,0.5])
    # # print(center)
    # center_pcd.translate(tuple(points[0]))

    # filtered_pcd = open3d.geometry.PointCloud()
    # filtered_pcd.points = open3d.utility.Vector3dVector(np.array(mp_candidates))
    # open3d.visualization.draw_geometries([filtered_pcd, pcd.translate((0,0,0.2)), obb, center_pcd, origin])

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
        # # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), pcd_goal.translate((0.2,0,0))])
        open3d.visualization.draw_geometries(
            [
                pcd,
                mani_point.translate(tuple(gt_mp)),
                best_mani_point.translate(tuple(best_mp)),
                pcd_goal.translate((0.2, 0, 0)),
            ]
        )

    print("=====================")
