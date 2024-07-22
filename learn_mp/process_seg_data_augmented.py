import open3d
import os
import numpy as np
import pickle
import timeit
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors
import torch
from copy import deepcopy


def down_sampling(pc):
    farthest_indices, _ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]
    return pc


data_recording_path = (
    "/home/baothach/shape_servo_data/manipulation_points/box/processed_seg_data"
)
data_processed_path = "/home/baothach/shape_servo_data/manipulation_points/box/processed_seg_data_augmented"

mp_data_path = "/home/baothach/shape_servo_data/manipulation_points/box/data"

start_time = timeit.default_timer()
all_chamfers = []


start_index = 0
max_len_data = 10000
file_names = sorted(os.listdir(data_recording_path))
mp_filenames = sorted(os.listdir(mp_data_path))

pcd = open3d.io.read_point_cloud(
    "/home/baothach/shape_servo_data/manipulation_points/box/init_box_pc.pcd"
)
pc_resampled_ori = down_sampling(np.asarray(pcd.points))

with open(os.path.join(data_recording_path, file_names[343]), "rb") as handle:
    data_2 = pickle.load(handle)

for i in range(start_index, max_len_data):
    if i % 50 == 0:
        print(
            "current count:", i, " , time passed:", timeit.default_timer() - start_time
        )

    file_name = os.path.join(data_recording_path, file_names[i])

    if not os.path.isfile(file_name):
        continue

    with open(file_name, "rb") as handle:
        data = pickle.load(handle)

    with open(os.path.join(mp_data_path, mp_filenames[i]), "rb") as handle:
        mp_data = pickle.load(handle)

    pc_resampled = pc_resampled_ori[np.random.permutation(pc_resampled_ori.shape[0])]
    pc_goal_resampled = data["partial pcs"][1].transpose(1, 0)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc_resampled)
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(pc_goal_resampled)

    max_x = max_y = max_z = 1
    max_rot = 1.57
    delta_x = np.random.uniform(low=-max_x, high=max_x)
    delta_y = np.random.uniform(low=-max_y, high=max_y)
    delta_z = np.random.uniform(low=0.0, high=max_z)
    rot = np.random.uniform(low=-max_rot, high=max_rot)
    center = np.array([[0.0, -0.42, 0.01818]]).T

    # pcd2 = deepcopy(pcd)
    # delta_x, delta_y, delta_z, rot = 0.1,0.1,0.1,0.707

    R = pcd.get_rotation_matrix_from_zxy([[rot], [0], [0]])

    pcd.rotate(R, center)
    pcd.translate((delta_x, delta_y, delta_z))
    pcd_goal.rotate(R, center)
    pcd_goal.translate((delta_x, delta_y, delta_z))
    pc_resampled = np.asarray(pcd.points)
    pc_goal_resampled = np.asarray(pcd_goal.points)

    mp_pos = np.array(list(mp_data["mani_point"]["pose"][0]))
    mp_pcd = open3d.geometry.PointCloud()
    mp_pcd.points = open3d.utility.Vector3dVector(np.array(mp_pos).reshape(1, -1))
    mp_pcd.rotate(R, center)
    mp_pcd.translate((delta_x, delta_y, delta_z))
    transformed_mp_pos = np.asarray(mp_pcd.points).squeeze()
    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pc_resampled)
    _, nearest_idxs = neigh.kneighbors(transformed_mp_pos.reshape(1, -1))
    mp_channel = np.zeros(pc_resampled.shape[0])
    mp_channel[nearest_idxs.flatten()] = 1

    # colors = np.zeros((1024,3))
    # colors[nearest_idxs.flatten()] = [1,0,0]
    # pcd.colors =  open3d.utility.Vector3dVector(colors)
    # mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # mani_point.paint_uniform_color([0,0,1])
    # pcd_goal.paint_uniform_color([0,1,0])

    # mani_point2 = deepcopy(mani_point)
    # new_mp_pos = np.asarray(mp_pcd.points).squeeze()
    # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(transformed_mp_pos)), pcd_goal, pcd2, mani_point2.translate(tuple(mp_pos))])

    pcs = (pc_resampled.transpose(1, 0), pc_goal_resampled.transpose(1, 0))
    processed_data = {
        "partial pcs": pcs,
        "mp_labels": mp_channel,
        "obj_name": data["obj_name"],
    }
    with open(os.path.join(data_processed_path, file_names[i]), "wb") as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
