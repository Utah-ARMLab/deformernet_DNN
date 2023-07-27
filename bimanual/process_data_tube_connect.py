import open3d
import os
import numpy as np
import pickle
import timeit
import torch

import sys
sys.path.append("../")
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors

def down_sampling(pc, num_point=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_point)
    pc = pc[farthest_indices.squeeze()]  
    return pc

ROBOT_Z_OFFSET = 0.25

data_recording_path = "/home/baothach/shape_servo_data/tube_connect/cylinder/data"
data_processed_path = "/home/baothach/shape_servo_data/tube_connect/cylinder/processed_data"
# i = 1001
start_time = timeit.default_timer() 

start_index =  5700
max_len_data = 5846

with torch.no_grad():
    for i in range(start_index, max_len_data):
        
        if i % 50 == 0:
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
        
        file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)

        pc = down_sampling(data["partial pcs"][0]).transpose(1,0)
        pc_goal = down_sampling(data["partial pcs"][1]).transpose(1,0)
    
        

        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(two_pcs)
        # # pcd_2 = open3d.geometry.PointCloud()
        # # pcd_2.points = open3d.utility.Vector3dVector(down_sampling(data["partial pcs"], 2048))        
        # open3d.visualization.draw_geometries([pcd, pcd_2.translate((0.2,0,0))])  


        pcs = (pc, pc_goal)
        processed_data = {"partial pcs": pcs, "full pcs": data["full pcs"], "positions": data["positions"]}
                        
        
        with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


    



















