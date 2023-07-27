import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle
import open3d
import sklearn

                       



class PlaneSDFDataset(Dataset):
    """Shape servo dataset."""
    '''
    Dataset for surgical setup task
    '''

    def __init__(self, percentage = 1.0):
        """
        Args:

        """ 

        self.dataset_path = "/home/baothach/shape_servo_data/RL_shapeservo/box/plane_data/"
        


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
        
        state = torch.tensor(sample["embedding"]).float()   
        plane = torch.from_numpy(sample["plane"]).float()
        percent_passed = 100*(torch.tensor(sample["percent passed"])).unsqueeze(0).float() 
        
        sample = {"state": state, "plane": plane, "percent passed": percent_passed}        

        return sample    



        