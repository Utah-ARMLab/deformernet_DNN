import open3d
import os
import numpy as np
import pickle
import timeit
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors
import torch


def down_sampling(pc):
    farthest_indices, _ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]
    return pc


data_recording_path = (
    "/home/baothach/shape_servo_data/manipulation_points/box/processed_seg_data"
)
data_processed_path = (
    "/home/baothach/shape_servo_data/manipulation_points/box/processed_seg_data2"
)

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
    pc_goal_resampled = data["partial pcs"][1]
    # print(pc_resampled.shape, pc_goal_resampled.shape)

    mp_pos = np.array(list(mp_data["mani_point"]["pose"][0]))
    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pc_resampled)
    _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
    mp_channel = np.zeros(pc_resampled.shape[0])
    mp_channel[nearest_idxs.flatten()] = 1

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(pc_resampled))
    # colors = np.zeros((1024,3))
    # colors[nearest_idxs.flatten()] = [1,0,0]
    # pcd.colors =  open3d.utility.Vector3dVector(colors)
    # mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # mani_point.paint_uniform_color([0,0,1])
    # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(mp_pos))])

    pcs = (np.transpose(pc_resampled, (1, 0)), pc_goal_resampled)
    processed_data = {
        "partial pcs": pcs,
        "mp_labels": mp_channel,
        "obj_name": data["obj_name"],
    }
    with open(os.path.join(data_processed_path, file_names[i]), "wb") as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
