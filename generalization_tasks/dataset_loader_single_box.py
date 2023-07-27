import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle
import open3d
import sklearn

                       



class SingleBoxDataset(Dataset):
    """Shape servo dataset."""
    '''
    Dataset for surgical setup task
    '''

    def __init__(self, percentage = 1.0):
        """
        Args:

        """ 

        self.dataset_path = "/home/baothach/shape_servo_data/RL_shapeservo/box/processed_data/"


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
        pc = torch.tensor(sample["partial pcs"][0]).float()   # original partial point cloud
        pc_goal = torch.tensor(sample["partial pcs"][1]).float()      

        position = (torch.tensor(sample["positions"])*1000).float()
        # print(position.shape)
        sample = {"pcs": (pc, pc_goal), "positions": position}        

        return sample    



        