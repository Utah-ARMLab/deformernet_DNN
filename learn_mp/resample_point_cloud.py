import open3d
import os
import numpy as np
import pickle
import timeit
from farthest_point_sampling import *


def down_sampling(pc):
    farthest_indices, _ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]
    return pc


data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/mp_classifer_data"
data_processed_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/resampled_mp_classifer_data"

start_time = timeit.default_timer()
all_chamfers = []


start_index = 0
max_len_data = 5000
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

    pc_resampled = down_sampling(data["partial pcs"][0])
    pc_goal_resampled = down_sampling(data["partial pcs"][1])

    pcs = (np.transpose(pc_resampled, (1, 0)), np.transpose(pc_goal_resampled, (1, 0)))
    processed_data = {
        "partial pcs": pcs,
        "mani_point": data["mani_point"],
        "chamfer": data["chamfer"],
        "gt mani_point": data["gt mani_point"],
        "gt chamfer": data["gt chamfer"],
        "obj_name": data["obj_name"],
    }
    with open(os.path.join(data_processed_path, file_names[i]), "wb") as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # all_chamfers.append(data["chamfer"])


for threshold in [0.2, 0.25, 0.3, 0.4, 0.5]:
    print(
        f"Chamfer < {threshold}m count:",
        sum(1 for chamf in all_chamfers if chamf <= threshold),
    )

print("===========")

for threshold in [0.8, 1.0]:
    print(
        f"Chamfer > {threshold}m count:",
        sum(1 for chamf in all_chamfers if chamf >= threshold),
    )
