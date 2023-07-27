import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle
import open3d
import sklearn

                       



class TubeConnectDataset(Dataset):
    """Shape servo dataset."""
    '''
    Dataset for surgical setup task
    '''

    def __init__(self, percentage = 1.0):

        # self.dataset_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot/processed_data"
        self.dataset_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/processed_data"


        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]
 

    
    def load_pickle_data(self, filename):
        if os.path.getsize(os.path.join(self.dataset_path, filename)) == 0: 
            print(filename)
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        sample = self.load_pickle_data(self.filenames[idx])
        
        pc = torch.tensor(sample["partial pcs"][0]).float()   # original partial point cloud
        pc_goal = torch.tensor(sample["partial pcs"][1]).float()      

        position_1 = torch.tensor(sample["pos_1"]).squeeze().float()
        rot_mat_1 = torch.tensor(sample["rot_1"]).float()
        position_2 = torch.tensor(sample["pos_2"]).squeeze().float()
        rot_mat_2 = torch.tensor(sample["rot_2"]).float()
        sample = {"pcs": (pc, pc_goal), "pos_1": position_1, "rot_1": rot_mat_1,
                  "pos_2": position_2, "rot_2": rot_mat_2}   



        return sample    



        