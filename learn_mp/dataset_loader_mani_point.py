import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle
import open3d
import sklearn
from farthest_point_sampling import *

                       



class SingleBoxDataset(Dataset):
    """Shape servo dataset."""
    '''
    Dataset for surgical setup task
    '''

    def __init__(self, percentage = 1.0, model_type='classifier', dataset_path=None):
        """
        Args:

        """ 

        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/box/processed_mp_classifer_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/processed_mp_classifer_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa_2/processed_classifer_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/multi_hemis_5kPa/processed_classifer_data"
        self.dataset_path = dataset_path

        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]
 
        self.model_type = model_type
    
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
        
        if self.model_type == 'classifier':
            label = torch.tensor(sample["label"]).long()
            sample = {"pcs": (pc, pc_goal), "label": label}   
        elif self.model_type == 'regressor':
            chamfer = (torch.tensor(sample["chamfer"])*1000).unsqueeze(-1).float()
            sample = {"pcs": (pc, pc_goal), "chamfer": chamfer}  
        
        return sample    



class DensePredictorDataset(Dataset):
    
    """
    Dataset for dense predictor training. Predict manipulation point using segmentation network.
    """


    def __init__(self, percentage = 1.0, dataset_path=None):
        """
        Args:

        """ 

        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/processed_seg_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/bimanual/multi_boxes_1000Pa/processed_seg_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa_2/processed_seg_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/multi_hemis_5kPa/processed_seg_data"
        self.dataset_path = dataset_path

        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]
 

        # ######
        # def down_sampling(pc):
        #     farthest_indices,_ = farthest_point_sampling(pc, 1024)
        #     pc = pc[farthest_indices.squeeze()]  
        #     return pc
        # pcd = open3d.io.read_point_cloud("/home/baothach/shape_servo_data/manipulation_points/box/init_box_pc.pcd")    
        # pc = down_sampling(np.asarray(pcd.points))
        # self.pc_tensor = torch.from_numpy(pc).permute(1,0).float()

    def load_pickle_data(self, filename):
        if os.path.getsize(os.path.join(self.dataset_path, filename)) == 0: 
            print(filename)
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):        
        sample = self.load_pickle_data(self.filenames[idx])

        pc = torch.tensor(sample["partial pcs"][0]).float() 
        pc_goal = torch.tensor(sample["partial pcs"][1]).float()    
        
        label = torch.tensor(sample["mp_labels"]).long()

        
        sample = {"pcs": (pc, pc_goal), "label": label}   

        
        return sample          



        