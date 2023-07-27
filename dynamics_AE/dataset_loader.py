import torch.utils.data as data
import pickle
import torch
import numpy as np
import pickle as pkl
import os

class DynamicsDataset(data.Dataset):
    """
    Dataset that returns current state / next state image pairs
    """

    def __init__(self, percentage = 1.0, dataset_path = "/home/baothach/shape_servo_data/RL_shapeservo/box/processed_data/"):
        """
        Args:

        """ 

        self.dataset_path = dataset_path


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
        obs = torch.tensor(sample["partial pcs"][0]).float()   # original partial point cloud
        obs_next = torch.tensor(sample["partial pcs"][1]).float()      

        action = (torch.tensor(sample["positions"])*1000).float()   



        return obs, obs_next, action