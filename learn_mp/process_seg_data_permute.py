import open3d
import os
import numpy as np
import pickle
import timeit
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors
import torch

np.random.seed(0)


def down_sampling(pc):
    farthest_indices, _ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]
    return pc


data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/processed_seg_data"
data_processed_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/processed_seg_data_2"

# mp_data_path = "/home/baothach/shape_servo_data/manipulation_points/box/data"

start_time = timeit.default_timer()
all_chamfers = []


start_index = 0
max_len_data = 12000
file_names = sorted(os.listdir(data_recording_path))


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

    pc = data["partial pcs"][0].transpose((1, 0))
    permuted_idxs = np.random.permutation(pc.shape[0])
    pc_permuted = pc[permuted_idxs]
    print(pc_permuted.shape)

    mp_channel = data["mp_labels"]
    permuted_mp_channel = mp_channel[permuted_idxs]

    pc_goal = data["partial pcs"][1].transpose((1, 0))
    pc_goal_permuted = pc_goal[np.random.permutation(pc_goal.shape[0])]
    print(pc_goal_permuted.shape)

    pcs = (np.transpose(pc_permuted, (1, 0)), np.transpose(pc_goal_permuted, (1, 0)))
    processed_data = {
        "partial pcs": pcs,
        "mp_labels": permuted_mp_channel,
        "obj_name": data["obj_name"],
    }
    with open(os.path.join(data_processed_path, file_names[i]), "wb") as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
