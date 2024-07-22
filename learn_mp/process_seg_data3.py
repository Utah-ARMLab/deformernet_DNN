import open3d
import os
import numpy as np
import pickle
import timeit
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors


def down_sampling(pc):
    farthest_indices, _ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]
    return pc


# data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa_2/data"
# data_processed_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa_2/processed_seg_data"

data_recording_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa_2/processed_data_w_mp"
data_processed_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa_2/processed_seg_data"

start_time = timeit.default_timer()
all_chamfers = []


start_index = 0
max_len_data = 12000
file_names = sorted(os.listdir(data_recording_path))

# with open(os.path.join(data_recording_path, file_names[343]), 'rb') as handle:
#     data_2 = pickle.load(handle)

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

    pcs = (data["partial pcs"][0][:3, :], data["partial pcs"][1])
    mp_channel = data["partial pcs"][0][3, :]
    # print(mp_channel.shape)

    processed_data = {
        "partial pcs": pcs,
        "mp_labels": mp_channel,
        "obj_name": data["obj_name"],
    }
    with open(os.path.join(data_processed_path, file_names[i]), "wb") as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
