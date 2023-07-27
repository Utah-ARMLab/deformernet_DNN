import numpy as np
import pickle
import sklearn

with open('/home/baothach/shape_servo_data/batch_1', 'rb') as handle:
    data1 = pickle.load(handle)


# with open('/home/baothach/shape_servo_data/batch_2', 'rb') as handle:
#     data2 = pickle.load(handle)


# with open('/home/baothach/shape_servo_data/batch_3', 'rb') as handle:
#     data3 = pickle.load(handle)


# final_point_clouds = data1["point clouds"] + data2["point clouds"] + data3["point clouds"]
# final_desired_positions = data1["positions"] + data2["positions"] + data3["positions"]

final_point_clouds = data1["point clouds"] 
final_desired_positions = data1["positions"] 

final_point_clouds, final_desired_positions = sklearn.utils.shuffle(final_point_clouds, final_desired_positions)
data = {"point clouds": final_point_clouds, "positions": final_desired_positions}
with open('/home/baothach/shape_servo_data/batch_1_shuffled', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



