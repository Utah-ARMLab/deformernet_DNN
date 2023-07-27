import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle
import open3d
import sklearn

                       



class SurgicalSetUpDataset(Dataset):
    """Shape servo dataset."""
    '''
    Dataset for surgical setup task
    '''

    def __init__(self, percentage = 1.0):
        """
        Args:

        """ 

        self.dataset_path = "/home/baothach/shape_servo_data/generalization/surgical_setup/data_on_ground/"


        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]
 

    
    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        sample = self.load_pickle_data(self.filenames[idx])
        pc = torch.tensor(sample["full pcs"][0]).permute(1,0).float()   # original partial point cloud
        pc_goal = torch.tensor(sample["full pcs"][1]).permute(1,0).float()      

        
        # if torch.isnan(pc).any()  or torch.isnan(pc_goal).any():
        #     print("nan file name:", self.filenames[idx])  
        #     print(np.isnan(sample["full pcs"][0]).any(), np.isnan(sample["full pcs"][1]).any())
          
        # print(pc_goal.shape)
        # pc = torch.tensor(sample["point clouds"][0]).permute(1,0).float()    # original partial point cloud
        # pc_goal = torch.tensor(sample["point clouds"][1]).permute(1,0).float()   
        # print("shape", pc.shape, pc_goal.shape)    

        # grasp_pose = (torch.tensor(list(self.dataset[idx]["grasp pose"][0]))*1000).float()
        position = (torch.tensor(sample["positions"])*1000).float()
        # print(position.shape)
        sample = {"pcs": (pc, pc_goal), "positions": position}        

        return sample    

class SurgicalSetUpManiPointDataset(Dataset):
    """Shape servo dataset."""
    '''
    Dataset for train Mani Point with regression
    '''

    def __init__(self, percentage = 1.0):
        """
        Args:

        """ 

        self.dataset_path = "/home/baothach/shape_servo_data/generalization/surgical_setup/data_on_ground/"


        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]
 

    
    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        sample = self.load_pickle_data(self.filenames[idx])
        pc = torch.tensor(sample["full pcs"][0]).permute(1,0).float()   # original partial point cloud
        pc_goal = torch.tensor(sample["full pcs"][1]).permute(1,0).float()      
         

        grasp_pose = (torch.tensor(list(sample["grasp_pose"][0]))*1000).float()
        # position = (torch.tensor(sample["positions"])*1000).float()

        sample = {"pcs": (pc, pc_goal), "positions": grasp_pose}        

        return sample               

class BoxDataset(Dataset):
    """Shape servo dataset."""
    '''
    Dataset for surgical setup task
    '''

    def __init__(self, percentage = 1.0):
        """
        Args:

        """ 


        # self.dataset_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_1000Pa/processed_data_partial"
        # self.dataset_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_5kPa/processed_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_10kPa/processed_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/generalization/multi_hemis_5kPa/processed_data"
        self.dataset_path = "/home/baothach/shape_servo_data/generalization/multi_hemis_10kPa/processed_data"

        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]
 

    
    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        sample = self.load_pickle_data(self.filenames[idx])


        pc = torch.tensor(sample["full pcs"][0]).float()   # original partial point cloud
        pc_goal = torch.tensor(sample["full pcs"][1]).float()         

        # pc = torch.tensor(sample["partial pcs"][0]).float()   # original partial point cloud
        # pc_goal = torch.tensor(sample["partial pcs"][1]).float()              
        

        position = (torch.tensor(sample["positions"])*1000).float()
        # print(position.shape)
        sample = {"pcs": (pc, pc_goal), "positions": position}        

        return sample         