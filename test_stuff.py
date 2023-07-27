import open3d
import numpy as np
import torch
import pickle

from pointcloud_recon import PointCloudAE

model = PointCloudAE(normal_channel=False)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/weights/AE_weights_only_one_chamfer(4)"))

with open('/home/baothach/shape_servo_data/data_for_training_AE', 'rb') as handle:
    data = pickle.load(handle)["point clouds"]

pc_1 = data[3]
pc_2 = data[5]
points_1 = np.swapaxes(pc_1,0,1)  
points_2 = np.swapaxes(pc_2,0,1)
delta = points_2 - points_1
print(sum(delta[0]))
print(sum(delta[1]))
print(sum(delta[2]))

# # reconstructed_points = data[0]

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(np.array(pc_1))  
pcd.paint_uniform_color([0, 1, 0])

pcd_2 = open3d.geometry.PointCloud()
pcd_2.points = open3d.utility.Vector3dVector(np.array(pc_2))  
pcd_2.paint_uniform_color([1, 0, 0])

delta = np.swapaxes(delta,0,1)
print(np.where(delta>0.000001))
# print(np.linalg.norm(delta[0]))
# print(np.linalg.norm(delta[1]))
# print(np.linalg.norm(delta[2]))
# pcd_3 = open3d.geometry.PointCloud()
# pcd_3.points = open3d.utility.Vector3dVector(delta)  
# pcd_3.paint_uniform_color([1, 0, 0])

# open3d.visualization.draw_geometries([pcd, pcd_2, pcd_3])


# pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=8))   
# normals = np.asarray(pcd.normals)
# processed_points = np.concatenate((points, normals), axis = 1)
# processed_points = points
# # print(processed_points.shape)

# reconstructed_points = model(torch.from_numpy(np.swapaxes(processed_points,0,1)).unsqueeze(0).float())
# # print(reconstructed_points.shape)
# reconstructed_points = np.swapaxes(reconstructed_points.squeeze().detach().numpy(),0,1)
# reconstructed_points = reconstructed_points[:,:3]
# print(reconstructed_points.shape)
# pcd2 = open3d.geometry.PointCloud()
# pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points)) 
# open3d.visualization.draw_geometries([pcd2])  