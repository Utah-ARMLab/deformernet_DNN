import open3d
import numpy as np
import torch
import pickle

from pointcloud_recon import PointCloudAE
from pointcloud_recon_3 import PointCloudAE_v2

model = PointCloudAE_v2(normal_channel=False)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/weights/AE_weights_only_one_chamfer_new"))

with open('/home/baothach/shape_servo_data/data_for_training_AE', 'rb') as handle:
    data = pickle.load(handle)["point clouds"]
    points = data[0]  

# reconstructed_points = data[0]

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(np.array(points))  
pcd.paint_uniform_color([1, 0.706, 0])

pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=8))   
normals = np.asarray(pcd.normals)
processed_points = np.concatenate((points, normals), axis = 1)
processed_points = points
# print(processed_points.shape)

reconstructed_points = model(torch.from_numpy(np.swapaxes(processed_points,0,1)).unsqueeze(0).float())
# print(reconstructed_points.shape)
reconstructed_points = np.swapaxes(reconstructed_points.squeeze().detach().numpy(),0,1)
reconstructed_points = reconstructed_points[:,:3]
print(reconstructed_points.shape)
pcd2 = open3d.geometry.PointCloud()
pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points)) 
open3d.visualization.draw_geometries([pcd, pcd2])  