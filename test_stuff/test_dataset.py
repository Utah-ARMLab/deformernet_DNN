import open3d
import os
import numpy as np
import pickle5 as pickle
import timeit
import torch
import argparse
import sys
# sys.path.append("../")
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors

def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

def pcd_ize(pc, color=None, vis=False):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd

ROBOT_Z_OFFSET = 0.25
two_robot_offset = 1.0

data_recording_path = f"/home/baothach/Downloads/ICRA2022_deformernet_dataset/hemis_10k/"
index_range = [0,100]


for i in range(*index_range):       
    file_name = os.path.join(data_recording_path, "processed sample " + str(i) + ".pickle")

    if not os.path.isfile(file_name):
        print(f"{file_name} not found")
        continue 

    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
        
    print("Keys:", data.keys())

    pc = down_sampling(data["partial pcs"][0].transpose(1,0))
    pc_goal = down_sampling(data["partial pcs"][1].transpose(1,0))
    mani_point_position = data["grasp_pose"][0]
    print("pc.shape, pc_goal.shape:", pc.shape, pc_goal.shape)
    print("mani_point_position:", mani_point_position)   
    
    pcd = pcd_ize(pc, color=[0,0,0])
    pcd_goal = pcd_ize(pc_goal, color=[1,0,0])
    mani_point_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mani_point_sphere.paint_uniform_color([0,1,0])
    mani_point_sphere.translate(tuple(mani_point_position))
    
    open3d.visualization.draw_geometries([pcd, pcd_goal, mani_point_sphere]) 
        
         




    



















