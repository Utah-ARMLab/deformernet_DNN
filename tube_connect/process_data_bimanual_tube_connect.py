import open3d
import os
import numpy as np
import pickle
import timeit
# import torch

import sys
sys.path.append("../")
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors

def down_sampling(pc):
    farthest_indices,_ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]  
    return pc

ROBOT_Z_OFFSET = 0.25
two_robot_offset = 1.0

data_recording_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/data"
data_processed_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/processed_data"
# i = 1001
start_time = timeit.default_timer() 

start_index = 9750
max_len_data = 13000


for i in range(start_index, max_len_data):
    # if 11161 <= i <= 11168:
    #     continue
    
    if i % 50 == 0:
        print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
    
    file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)

    pc = down_sampling(data["partial pcs"][0]).transpose(1,0)
    pc_goal = down_sampling(data["partial pcs"][1]).transpose(1,0)
    
    two_robot_offset = data["two_robot_offset"]

    mani_point_1 = data["mani_point_1"]        
    mp_pos_1 = np.array([mani_point_1[0,3]+two_robot_offset.x, two_robot_offset.y + mani_point_1[1,3], mani_point_1[2,3] + ROBOT_Z_OFFSET])
    # mp_pos_1 = np.array([mani_point_1[0,3], -two_robot_offset + mani_point_1[1,3], mani_point_1[2,3] + ROBOT_Z_OFFSET])
    
    mani_point_2 = data["mani_point_2"]        
    mp_pos_2 = np.array([-mani_point_2[0,3], -mani_point_2[1,3], mani_point_2[2,3] + ROBOT_Z_OFFSET])

    
    
    

    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pc.transpose(1,0))
    
    _, nearest_idxs_1 = neigh.kneighbors(mp_pos_1.reshape(1, -1))
    mp_channel_1 = np.zeros(pc.shape[1])
    mp_channel_1[nearest_idxs_1.flatten()] = 1
    
    _, nearest_idxs_2 = neigh.kneighbors(mp_pos_2.reshape(1, -1))
    mp_channel_2 = np.zeros(pc.shape[1])
    mp_channel_2[nearest_idxs_2.flatten()] = 1        
    
    modified_pc = np.vstack([pc, mp_channel_1, mp_channel_2])# pc with 4th and 5th channel with value of 1 if near the MP point and 0 elsewhere
    
    # # # # print(modified_pc[:4,100])
    # # assert modified_pc.shape == (5,1024) and pc_goal.shape == (3,1024)
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(pc.transpose(1,0)))
    # colors = np.zeros((1024,3))
    # colors[nearest_idxs_1.flatten()] = [1,0,0]
    # colors[nearest_idxs_2.flatten()] = [0,1,0]
    # pcd.colors =  open3d.utility.Vector3dVector(colors)
    # mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # mani_point_1_sphere.paint_uniform_color([0,0,1])
    # mani_point_1_sphere.translate(tuple(mp_pos_1))
    # mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # mani_point_2_sphere.paint_uniform_color([1,0,0])
    # mani_point_2_sphere.translate(tuple(mp_pos_2))
    # open3d.visualization.draw_geometries([pcd, mani_point_1_sphere, mani_point_2_sphere])           


    pcs = (modified_pc, pc_goal)
    processed_data = {"partial pcs": pcs, "full pcs": data["full pcs"],
                    "pos_1": data["pos_1"], "rot_1": data["rot_1"], "twist_1": data["twist_1"],
                    "pos_2": data["pos_2"], "rot_2": data["rot_2"], "twist_2": data["twist_2"],
                    "mani_point_1": data["mani_point_1"], "mani_point_2": data["mani_point_2"], 
                    "two_robot_offset": data["two_robot_offset"]}
    
    with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


    



















