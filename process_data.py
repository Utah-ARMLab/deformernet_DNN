import h5py
import numpy as np
import pickle


import open3d
import os
import numpy as np
import pickle
import timeit
from farthest_point_sampling import *
# from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation

# resample = lambda points, n: gather_operation(points.transpose(1 , 2).contiguous(), furthest_point_sample(points, n))

data_recording_path = "/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/batch3_original_partial_pc/processed_2"
data_processed_path = "/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/batch3_original_partial_pc/processed_2"
# i = 1001
start_time = timeit.default_timer() 

start_index = 0
max_len_data = 13000

for i in range(start_index, max_len_data):
    if i % 50 == 0:
        print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
    
    file_name = os.path.join(data_recording_path, "processed sample " + str(i) + ".pickle")
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)

    # pc = data["point clouds"][0] 
    # pc_goal = data["point clouds"][1]
    
   
   
    max_x = 0.4 
    max_y = 0.4               
    max_z = 0.2       

    delta_x = np.random.uniform(low = -max_x, high = max_x)   
    delta_y = np.random.uniform(low = -max_y, high = max_y)
    delta_z = np.random.uniform(low = 0, high = max_z) 

    augmented_pc = data["point clouds"][0]
    # augmented_pc[0] += delta_x
    # augmented_pc[1] += delta_y
    # augmented_pc[2] += delta_z
    augmented_pc_goal = data["point clouds"][1]
    augmented_pc_goal[0] += delta_x
    augmented_pc_goal[1] += delta_y
    augmented_pc_goal[2] += delta_z

    positions = data["positions"]
    positions[0] -= delta_x
    positions[1] -= delta_y
    positions[2] += delta_z


    pcs = (augmented_pc, augmented_pc_goal)
    augmented_data = {"point clouds": pcs, "positions": positions, "grasp_pose": data["grasp_pose"]}
    with open(os.path.join(data_processed_path, "processed sample " + str(i+max_len_data) + ".pickle"), 'wb') as handle:
        pickle.dump(augmented_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    



















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






































# f = h5py.File("/home/baothach/shape_servo_data/multi_grasps/batch_1")
# total_grasp = f['cur_grasp_id'][()] + 1
# print("total:", total_grasp)
# # grasp_poses = []
# # point_clouds = []
# # positions = []
# datas = []
# count = 0


# for i in range(total_grasp):
#     # pc_pairs = f['point clouds '+ str(i)][()]
#     count += 1
#     print(count)
#     for k, pc_pair in enumerate(f['point clouds '+ str(i)][()]):
#         position = f['positions '+ str(i)][()][k]
#         grasp_pose = f['manipulation pose '+ str(i)][()]
#         modified_pc = (np.swapaxes(pc_pair[0],0,1), np.swapaxes(pc_pair[1],0,1)) 
#         data = {"grasp pose": grasp_pose, "point clouds": modified_pc, "positions": position}
#         datas.append(data)

# with open('/home/baothach/shape_servo_data/multi_grasps/batch1_processed', 'wb') as handle:
#     pickle.dump(datas, handle, protocol=pickle.HIGHEST_PROTOCOL)
