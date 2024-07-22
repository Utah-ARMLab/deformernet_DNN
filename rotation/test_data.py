import os
import pickle
import numpy as np


# data_recording_path = "/home/baothach/shape_servo_data/rotation_extension/box/data"
# i = 1460
# file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")
# with open(file_name, 'rb') as handle:
#     data = pickle.load(handle)

# print(data)

datas = []
data_recording_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/processed_data_w_mp_twist"
start_index = 0
max_len_data = 13000
for i in range(start_index, max_len_data):
    file_name = os.path.join(data_recording_path, "mp sample " + str(i) + ".pickle")
    if not os.path.isfile(file_name):
        continue
    with open(file_name, "rb") as handle:
        data = pickle.load(handle)
    datas.append(abs(data["twist"]))

datas = np.array(datas).squeeze()
print(datas.shape)
# pos = datas[:,3:]
print("mean:", np.mean(datas, axis=0))
print("max:", np.max(datas, axis=0))
print("min:", np.min(datas, axis=0))
print("std:", np.std(datas, axis=0))
