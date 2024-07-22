import open3d
import os
import numpy as np
import pickle
import timeit
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors
from utils import find_knn, vis_mp
from random import shuffle
import argparse


parser = argparse.ArgumentParser(description=None)
parser.add_argument(
    "--obj_category", default="None", type=str, help="object category. Ex: boxes_1kPa"
)
args = parser.parse_args()

ROBOT_Z_OFFSET = 0.25


def down_sampling(pc):
    farthest_indices, _ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]
    return pc


def save_pickle(point_cloud, can_mp_pos, label, data, idx, vis=False, gt_mp=None):
    can_mp_channel = find_knn(point_cloud, can_mp_pos, num_nn=50)
    modified_pc = np.vstack([point_cloud, can_mp_channel])

    pcs = (modified_pc, data["partial pcs"][1])

    processed_data = {
        "partial pcs": pcs,
        "label": label,
        "mani_point": mani_point,
        "obj_name": data["obj_name"],
    }

    with open(
        os.path.join(data_processed_path, f"sample {idx}.pickle"), "wb"
    ) as handle:
        pickle.dump(processed_data, handle, protocol=3)

    if vis:
        vis_mp(point_cloud, data["partial pcs"][1], can_mp_pos, gt_mp)


data_recording_path = f"/home/baothach/shape_servo_data/manipulation_points/multi_{args.obj_category}/processed_seg_data"
data_processed_path = f"/home/baothach/shape_servo_data/manipulation_points/multi_{args.obj_category}/processed_classifer_data"
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer()
all_chamfers = []


start_index = 0
max_len_data = 3000  # len(os.listdir(data_recording_path)) #6003#5000#12000
file_names = sorted(os.listdir(data_recording_path))
shuffle(file_names)

save_file_idx = 0

num_candidate = 4  # 5 #4

# with open(os.path.join(data_recording_path, file_names[343]), 'rb') as handle:
#     data_2 = pickle.load(handle)

for i in range(start_index, max_len_data):
    if i % 50 == 0:
        print(
            "current count:", i, " , time passed:", timeit.default_timer() - start_time
        )

    file_name = os.path.join(data_recording_path, file_names[i])

    if not os.path.isfile(file_name):
        print(f"{file_name} not found")
        continue

    with open(file_name, "rb") as handle:
        data = pickle.load(handle)

    pc = data["partial pcs"][0][:3, :]

    mani_point = data["mani_point"]
    mp_pos = np.array(
        list(data["mani_point"][0])
    )  # np.array([-mani_point[0,3], -mani_point[1,3], mani_point[2,3] + ROBOT_Z_OFFSET])

    ys = pc[1]
    avg_y = (max(ys) + min(ys)) / 2
    above_idx = np.where(ys >= avg_y)[0]

    mp_channel = find_knn(pc, mp_pos, num_nn=50)  # num_candidate)
    positive_idx = np.where(mp_channel == 1)[0]
    positive_idx = np.array(list(set(positive_idx) & set(above_idx)))

    negative_channel = find_knn(pc, mp_pos, num_nn=100)
    negative_idx = np.where(negative_channel == 0)[0]
    negative_idx = np.array(list(set(negative_idx) & set(above_idx)))

    shuffle(positive_idx)
    shuffle(negative_idx)

    for i in range(num_candidate):
        if i >= positive_idx.shape[0] or i >= negative_idx.shape[0]:
            break

        # print("positive")
        can_pos = pc[:, positive_idx[i]]
        save_pickle(pc, can_pos, 1, data, save_file_idx, vis=False, gt_mp=mp_pos)
        save_file_idx += 1

        # print("negative")
        can_neg = pc[:, negative_idx[i]]
        save_pickle(pc, can_neg, 0, data, save_file_idx, vis=False, gt_mp=mp_pos)
        save_file_idx += 1
