import open3d
import os
import numpy as np
import pickle
import timeit

import sys

sys.path.append("../")
from farthest_point_sampling import *
import transformations


data_recording_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/processed_data_w_mp"
data_processed_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/processed_data_w_gripper_eulers"
# i = 1001
start_time = timeit.default_timer()

start_index = 0
max_len_data = 13000

for i in range(start_index, max_len_data):
    file_name = os.path.join(data_recording_path, "mp sample " + str(i) + ".pickle")
    if not os.path.isfile(file_name):
        continue
    if i % 50 == 0:
        print(
            "current count:", i, " , time passed:", timeit.default_timer() - start_time
        )

    with open(file_name, "rb") as handle:
        data = pickle.load(handle)

    gripper_eulers = np.array(transformations.euler_from_matrix(data["mani_point"]))
    # print(gripper_eulers.shape)

    processed_data = {
        "partial pcs": data["partial pcs"],
        "full pcs": data["full pcs"],
        "pos": data["pos"],
        "rot": data["rot"],
        "gripper_eulers": gripper_eulers,
        "mani_point": data["mani_point"],
        "obj_name": data["obj_name"],
    }

    with open(
        os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"),
        "wb",
    ) as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
