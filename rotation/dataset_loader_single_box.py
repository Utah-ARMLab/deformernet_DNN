import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle
# import open3d
# import sklearn



class SingleBoxDataset(Dataset):
    """Shape servo dataset."""
    '''
    Dataset for surgical setup task
    '''

    def __init__(self, percentage = 1.0, use_mp_input=True, dataset_path=None):
        """
        Args:

        """ 

        # self.dataset_path = "/home/baothach/shape_servo_data/rotation_extension/box/processed_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/rotation_extension/box/mp_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/rotation_extension/box/mp_data_smaller"
        # self.dataset_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/processed_data_w_mp_twist"
        # self.dataset_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/processed_data_w_gripper_eulers"
        # self.dataset_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa_2/processed_data_w_mp"
        # self.dataset_path = "/home/baothach/shape_servo_data/rotation_extension/multi_hemis_1000Pa_2/processed_data_w_mp"
        # self.dataset_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinders_1000Pa/processed_data_w_mp"
        # self.dataset_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{obj_category}/processed_data_w_mp"
        self.dataset_path = dataset_path

        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]
 
        self.use_mp_input = use_mp_input

    
    def load_pickle_data(self, filename):
        if os.path.getsize(os.path.join(self.dataset_path, filename)) == 0: 
            print(filename)
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        sample = self.load_pickle_data(self.filenames[idx])
        
        if self.use_mp_input:
            pc = torch.tensor(sample["partial pcs"][0]).float()   # original partial point cloud with MP input
        else:
            pc = torch.tensor(sample["partial pcs"][0][:3,:]).float()   # no MP input
        
        pc_goal = torch.tensor(sample["partial pcs"][1]).float()      
        # print("pc.shape:", pc.shape)

        position = (torch.tensor(sample["pos"])*1000).squeeze().float()
        rot_mat = torch.tensor(sample["rot"]).float()
        sample = {"pcs": (pc, pc_goal), "pos": position, "rot": rot_mat}   

        # # twist = (torch.tensor(sample["twist"])*1000).squeeze().float()
        # twist = (torch.tensor(sample["twist"]).squeeze() * torch.tensor([1000,1000,1000,100,100,100])).float()
        # sample = {"pcs": (pc, pc_goal), "twist": twist}   

        # position = (torch.tensor(sample["pos"])*1000).squeeze().float()
        # rot_mat = torch.tensor(sample["rot"]).float()
        # gripper_eulers = torch.tensor(sample["gripper_eulers"]).float()
        # sample = {"pcs": (pc, pc_goal), "pos": position, "rot": rot_mat, "gripper_eulers": gripper_eulers}   

        return sample    



        