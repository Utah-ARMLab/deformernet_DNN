import open3d
import os
import numpy as np
import pickle
import timeit

import sys
sys.path.append("../")
from farthest_point_sampling import *


data_recording_path = "/home/baothach/shape_servo_data/generalization/multi_hemis_10kPa/data"
data_processed_path = "/home/baothach/shape_servo_data/generalization/multi_hemis_10kPa/processed_data"
# i = 1001
start_time = timeit.default_timer() 

start_index = 0
max_len_data = 13000

for i in range(start_index, max_len_data):
    if i % 50 == 0:
        print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
    
    file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)

    pc = data["full pcs"][0] 
    pc_goal = data["full pcs"][1]


    # print("shape: ", pc.shape, pc_goal.shape)

    farthest_indices,_ = farthest_point_sampling(pc, 1024)
    pc_resampled = pc[farthest_indices.squeeze()]

    farthest_indices_goal,_ = farthest_point_sampling(pc_goal, 1024)
    pc_goal_resampled = pc_goal[farthest_indices_goal.squeeze()]

    pcs = (np.transpose(pc_resampled, (1, 0)), np.transpose(pc_goal_resampled, (1, 0)))
    processed_data = {"full pcs": pcs, "positions": data["positions"], "grasp_pose": data["grasp_pose"], "obj_name":data["obj_name"]}
    with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


    



















# with open('/home/baothach/shape_servo_data/batch_3b(using_camera)', 'rb') as handle:
#     data1 = pickle.load(handle)



# final_point_clouds = data1["point clouds"] 
# final_desired_positions = data1["positions"] 
# count = 0
# modified_pcs = []
# for (pc, pc_goal) in final_point_clouds:
#     count += 1
#     print(count)
#     # if count == 2:
#     #     break

#     pcd = open3d.geometry.PointCloud()

#     pcd.points = open3d.utility.Vector3dVector(np.array(pc))
#     pcd.estimate_normals(
#         search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1,
#                                                             max_nn=10))

#     radii = [0.005, 0.01, 0.02, 0.04] 
#     # radii =  [0.1, 0.2, 0.4, 0.8] 
#     rec_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#         pcd, open3d.utility.DoubleVector(radii))

#     rec_mesh.compute_vertex_normals()
#     pcd = rec_mesh.sample_points_uniformly(number_of_points=3000)

#     pcd_2 = open3d.geometry.PointCloud()

#     pcd_2.points = open3d.utility.Vector3dVector(np.array(pc_goal))
#     pcd_2.estimate_normals(
#         search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1,
#                                                             max_nn=10))

#     radii = [0.005, 0.01, 0.02, 0.04] 
#     # radii =  [0.1, 0.2, 0.4, 0.8] 
#     rec_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#         pcd_2, open3d.utility.DoubleVector(radii))

#     rec_mesh.compute_vertex_normals()
#     pcd_2 = rec_mesh.sample_points_uniformly(number_of_points=3000)
#     modified_pcs.append((np.asarray(pcd.points), np.asarray(pcd_2.points)))



# data = {"point clouds": modified_pcs, "positions": final_desired_positions}
# with open('/home/baothach/shape_servo_data/batch_3b(using_camera)_modified', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)