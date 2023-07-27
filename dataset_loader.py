import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle
import open3d
import sklearn

                       
# class PointNetShapeServoDataset(Dataset):
#     """Shape servo dataset."""

#     def __init__(self, train = True, percentage = 0.8):
#         """
#         Args:

#         """
#         with open('/home/baothach/shape_servo_data/multi_grasps/one_grasp', 'rb') as handle:
#             self.data = pickle.load(handle)
#             # self.data = self.data       
        
#         self.pc = self.data["point clouds"]
#         self.position = self.data["positions"]
#         # self.pc, self.position = sklearn.utils.shuffle(self.pc, self.position)
#         # self.get_normal_channels()
#         if train:
#             idx_for_train = int(percentage*len(self.pc))
#             self.pc = self.pc[:idx_for_train]    
#             self.position = self.position[:idx_for_train] 
           
#         else:
#             idx_for_test = int((1-percentage)*len(self.pc))
#             self.pc = self.pc[idx_for_test:]
#             self.position = self.position[idx_for_test:]       

#     def __len__(self):
#         return len(self.pc)

#     def __getitem__(self, idx):
#         # pc = torch.tensor(np.swapaxes(self.pc[0],0,1)).float()
#         # pc_goal = torch.tensor(np.swapaxes(self.pc[idx],0,1)).float()
#         # print(torch.tensor(self.pc[idx][0]).shape)
#         pc = torch.tensor(self.pc[idx][0]).permute(1,0).float()
#         pc_goal = torch.tensor(self.pc[idx][1]).permute(1,0).float()

#         position = (torch.tensor(self.position[idx])*1000).float()
#         # print(position.shape)
#         # print("okcnaicnem", pc.shape)
      
#         # print("shape: ", input.shape, target.shape)
#         sample = {'input': (pc, pc_goal), 'target': position}
#         return sample

#     def get_normal_channels(self):
#         processed_data = []
#         for point_cloud in self.data:
#             pcd = open3d.geometry.PointCloud()
#             pcd.points = open3d.utility.Vector3dVector(np.array(point_cloud))    
#             pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=8))   
#             normals = np.asarray(pcd.normals)
#             processed_data.append(np.concatenate((point_cloud, normals), axis = 1))
        
#         self.data = processed_data


# class PointNetShapeServoDataset(Dataset):
#     """Shape servo dataset."""
#     '''
#     Multi grasps
#     '''

#     def __init__(self, train = True, percentage = 0.8):
#         """
#         Args:

#         """ 
#         with open('/home/baothach/shape_servo_data/multi_grasps/batch1_processed', 'rb') as handle:
#             self.dataset = pickle.load(handle)
#             # self.data = self.data       
        
#         # self.pc = self.data["point clouds"]
#         # self.position = self.data["positions"]
#         # self.pc, self.position = sklearn.utils.shuffle(self.pc, self.position)
#         # self.get_normal_channels()
#         if train:
#             idx_for_train = int(percentage*len(self.dataset))
#             self.dataset = self.dataset[:idx_for_train]
           
#         else:
#             idx_for_test = int((1-percentage)*len(self.dataset))
#             self.dataset = self.dataset[idx_for_test:]     

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         # pc = torch.tensor(np.swapaxes(self.pc[0],0,1)).float()
#         # pc_goal = torch.tensor(np.swapaxes(self.pc[idx],0,1)).float()
#         # print(torch.tensor(self.pc[idx][0]).shape)
#         # pc = torch.tensor(self.pc[idx][0]).permute(1,0).float()
#         # pc_goal = torch.tensor(self.pc[idx][1]).permute(1,0).float()

#         # position = (torch.tensor(self.position[idx])*1000).float()

#         pc = torch.tensor(self.dataset[idx]["point clouds"][0]).float()
#         pc_goal = torch.tensor(self.dataset[idx]["point clouds"][1]).float()
#         # print(self.dataset[idx]["grasp pose"][0])
#         # print(self.dataset[idx]["grasp pose"])
#         grasp_pose = torch.tensor(list(self.dataset[idx]["grasp pose"][0])).float()
#         position = torch.tensor(self.dataset[idx]["positions"]*1000).float()

        
#         sample = {"grasp pose": grasp_pose, "point clouds": (pc, pc_goal), "positions": position}
#         return sample

# class PointNetShapeServoManiDataset(Dataset):
#     """Regression for manipulation point"""
#     '''
#     with 200 keypoints and paritial point cloud
#     '''

#     def __init__(self, percentage = 1.0):
#         """
#         Args:

#         """ 
#         self.dataset_path = "/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/batch4_ori/processed"
#         # with open(data_processed_path, 'rb') as handle:
#         #     self.dataset = pickle.load(handle)
#         self.filenames = os.listdir(self.dataset_path)
        
#         if percentage != 1.0:
#             self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]


    
#     def load_pickle_data(self, filename):
#         with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
#             return pickle.load(handle)            

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
        
#         sample = self.load_pickle_data(self.filenames[idx])
        
#         # pc = torch.tensor(sample["keypoints"][0]).float()     # keypoints cloud [:,:64]
#         # pc_goal = torch.tensor(sample["keypoints"][1]).float()

#         # print("shape:", pc.shape, pc_goal.shape)

#         pc = torch.tensor(sample["point clouds"][0]).float()    # original partial point cloud
#         pc_goal = torch.tensor(sample["point clouds"][1]).float()        
 
#         # grasp_pose = (torch.tensor(list(self.dataset[idx]["grasp pose"][0]))*1000).float()
#         grasp_pose = (torch.tensor(list(sample["grasp_pose"][0]))*1000).float()
#         # grasp_pose = (torch.tensor(sample["positions"])*1000).float()
        
#         sample = {"pcs": (pc, pc_goal), "grasp_pose": grasp_pose}        

#         return sample  


class PointNetShapeServoKpDataset(Dataset):
    """Shape servo dataset."""
    '''
    with 200 keypoints
    '''

    def __init__(self, percentage = 1.0):
        """
        Args:

        """ 
        # self.dataset_path = "/home/baothach/shape_servo_data/new_task/batch_1/data"
        self.dataset_path = "/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/batch3_original_partial_pc/processed_2"

        # with open(data_processed_path, 'rb') as handle:
        #     self.dataset = pickle.load(handle)
        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]



        # if train:
        #     idx_for_train = int(percentage*len(self.dataset))
        #     self.dataset = self.dataset[:idx_for_train]
           
        # else:
        #     idx_for_test = int((1-percentage)*len(self.dataset))
        #     self.dataset = self.dataset[idx_for_test:]     

    
    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        sample = self.load_pickle_data(self.filenames[idx])
        
        # pc = torch.tensor(sample["keypoints"][0]).float()     # keypoints cloud
        # pc_goal = torch.tensor(sample["keypoints"][1]).float()

        pc = torch.tensor(sample["point clouds"][0]).float()    # original partial point cloud
        pc_goal = torch.tensor(sample["point clouds"][1]).float()        

        # pc = torch.tensor(sample["point clouds"][0]).permute(1,0).float()    # original partial point cloud
        # pc_goal = torch.tensor(sample["point clouds"][1]).permute(1,0).float()   
        # print("shape", pc.shape, pc_goal.shape)    

        # grasp_pose = (torch.tensor(list(self.dataset[idx]["grasp pose"][0]))*1000).float()
        position = (torch.tensor(sample["positions"])*1000).float()
        
        sample = {"keypoints": (pc, pc_goal), "positions": position}        

        return sample        