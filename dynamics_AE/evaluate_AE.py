import torch
import os
from os.path import join, exists
import pickle
import open3d
import numpy as np
from models import *

device = torch.device('cuda:0')
folder = "/home/baothach/med_robot_course/learned_dynamics/autoencoder/run1"
checkpoint = torch.load(join(folder, 'checkpoint'), map_location=device)
encoder = checkpoint['encoder'].to(device)
decoder = checkpoint['decoder'].to(device)
trans = checkpoint['trans'].to(device)

with open('/home/baothach/shape_servo_data/RL_shapeservo/box/processed_data/processed sample 0.pickle', 'rb') as handle:
    obs_current = pickle.load(handle)["partial pcs"][0]

z, saved_points = encoder(torch.from_numpy(obs_current).unsqueeze(0).to(device), decode=True)
a = (torch.FloatTensor([0.01, 0.01, 0.01])*1000).unsqueeze(0).to(device)
# print(z.shape)
z_next = trans(z, a)
obs_next = decoder(saved_points).squeeze().permute(1,0).detach().cpu().numpy()



# reconstructed_points = data[0]

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(np.swapaxes(obs_current,0,1))   
pcd.paint_uniform_color([0, 0, 0])


pcd2 = open3d.geometry.PointCloud()
pcd2.points = open3d.utility.Vector3dVector(obs_next) 
pcd2.paint_uniform_color([1, 0, 0])

# open3d.visualization.draw_geometries([pcd, pcd2])  
open3d.visualization.draw_geometries([pcd2]) 
