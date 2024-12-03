import voxelmorph2d as vm2d
import voxelmorph3d as vm3d
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
from skimage.transform import resize
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
import wandb
import pickle

use_gpu = torch.cuda.is_available()

wandb.login(key='cddbb81e657d85514600791c422ff35c68117a53')

class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, input_dims, is_2d=False, use_gpu=True):
        self.dims = input_dims
        if is_2d:
            self.vm = vm2d
            self.voxelmorph = vm2d.VoxelMorph2d(input_dims[0] * 2, use_gpu)
        else:
            self.vm = vm3d
            self.voxelmorph = vm3d.VoxelMorph3d(input_dims[0] * 2, use_gpu)
            
        self.optimizer = optim.SGD(self.voxelmorph.parameters(), lr=1e-4, momentum=0.99)
        # self.optimizer = optim.Adam(
        #     self.voxelmorph.parameters(), lr=1e-3)
        self.params = {'batch_size': 3,
                       'shuffle': True,
                       'num_workers': 6,
                       'worker_init_fn': np.random.seed(42)
                       }
        self.device = torch.device("cuda:1" if use_gpu else "cpu")
        print(self.device)

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        self.check_dims(x)
        return self.voxelmorph(x)

    def calculate_loss(self, y, ytrue, n=9, lamda=0.01, is_training=True):
        loss = self.vm.vox_morph_loss(y, ytrue, n, lamda)
        return loss
    
    def ss_grid_gen(self,V):
        y_, x_ = np.meshgrid(np.arange(0, 256), np.arange(0, 256), indexing='ij')
        y_, x_ = 2.0*y_/255.0 - 1.0, 2.0*x_/255.0 - 1.0
        XY = torch.tensor(np.stack([x_,y_],axis=2),dtype=torch.float32).to(self.device)
        with torch.no_grad():
            normv2 = V[:,:,:,0]**2 + V[:,:,:,1]**2+1e-10
            m = torch.sqrt(torch.max(normv2))
            n = torch.ceil(torch.log2(100.0*(XY.shape[1]+XY.shape[2])*m)).type(torch.int64)
            # avoid null values
            if n<0:
                n = 0    
                
        # Scale it (so it's close to 0)
        V = V / (2**n)
        # V = V.permute(0, 2, 3, 1) # (B, H, W, 2)
        # print(V.shape)
        # print(XY.shape)

        for itr in range(n):
            Vx = F.grid_sample(V[:,:,:,0:1].permute(0,3,1,2), 
                            XY+V, padding_mode='reflection',align_corners=True).permute(0,2,3,1)
            Vy = F.grid_sample(V[:,:,:,1:2].permute(0,3,1,2), 
                            XY+V, padding_mode='reflection',align_corners=True).permute(0,2,3,1)
            V = V + torch.cat([Vx,Vy],dim=3)
        
        return XY+V, XY
    
    def normalize_dense_field(self, dense_field):
        """
        Normalize the dense field to have values between -1 and 1.
        
        Args:
            dense_field (torch.Tensor): The input dense field of shape (1, H, W, 2).
        
        Returns:
            torch.Tensor: The normalized dense field with values between -1 and 1.
        """
        # Find the min and max values for normalization
        min_val = dense_field.min()
        max_val = dense_field.max()
        
        # Normalize to the range [0, 1]
        normalized_field = (dense_field - min_val) / (max_val - min_val)
        
        # Scale to the range [-1, 1]
        normalized_field = normalized_field * 2 - 1
        
        return normalized_field


    def train_model(self, batch_moving, batch_fixed, n=9, lamda=0.01, return_metric_score=True):
        self.optimizer.zero_grad()
        batch_fixed, batch_moving = batch_fixed.to(
            self.device), batch_moving.to(self.device)
        registered_image, _ = self.voxelmorph(batch_moving, batch_fixed)
        train_loss = self.calculate_loss(
            registered_image, batch_fixed, n, lamda)
        train_loss.backward()
        self.optimizer.step()
        if return_metric_score:
            train_dice_score = self.vm.dice_score(
                registered_image, batch_fixed)
            return train_loss, train_dice_score
        return train_loss

    def get_test_loss(self, batch_moving, batch_fixed, epoch, count, n=9, lamda=0.01):
        with torch.set_grad_enabled(False):
            batch_fixed, batch_moving = batch_fixed.to(
                self.device), batch_moving.to(self.device)
            registered_image, flow = self.voxelmorph(batch_moving, batch_fixed)
            xyd, xy_ = self.ss_grid_gen(flow)
            xyd_ = self.normalize_dense_field(xyd)
            val_loss = self.vm.vox_morph_loss(
                registered_image, batch_fixed, n, lamda)
            val_dice_score = self.vm.dice_score(registered_image, batch_fixed)
            if count == 0:
                self.vm.visualise_results(
                batch_fixed[0], batch_moving[0], registered_image[0], xy_, xyd_[0], epoch)
            return val_loss, val_dice_score


class Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        fixed_image = torch.Tensor(
            resize(io.imread('./fire-fundus-image-registration-dataset/' + ID + '_1.jpg'), (256, 256, 3)))
        moving_image = torch.Tensor(
            resize(io.imread('./fire-fundus-image-registration-dataset/' + ID + '_2.jpg'), (256, 256, 3)))
        return fixed_image, moving_image


def main():
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    vm = VoxelMorph(
        (3, 256, 256), is_2d=True, use_gpu=True)# Object of the higher level class
    DATA_PATH = './fire-fundus-image-registration-dataset/'
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 4,
              'worker_init_fn': np.random.seed(42)
              }

    max_epochs = 10
    # filename = list(set([x.split('_')[0]
    #                      for x in os.listdir('./fire-fundus-image-registration-dataset/')]))
    # partition = {}
    # partition['train'], partition['validation'] = train_test_split(
    #     filename, test_size=0.33, random_state=42)

    with open('../partition.pkl', 'rb') as f:
        partition = pickle.load(f)
    # Generators
    training_set = Dataset(partition['train'])
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'])
    validation_generator = data.DataLoader(validation_set, batch_size=1, shuffle=False)

    print(len(training_set), len(validation_set))
    train_config = {'lr' :1e-4,
        'momentum' :0.99,
        'epochs' :10,}    
    
    # Loop over epochs
    with wandb.init(mode='disabled',project="Neural-B-spline-Image-Registration", config=train_config, name=f"Voxel-Image-Registration") as run:
        min_val_loss = float('inf')
        for epoch in range(max_epochs):
            start_time = time.time()
            train_loss = 0
            train_dice_score = 0
            val_loss = 0
            val_dice_score = 0
            for batch_fixed, batch_moving in training_generator:
                # print(batch_fixed.shape, batch_moving.shape)
                loss, dice = vm.train_model(batch_moving, batch_fixed)
                train_dice_score += dice.data
                train_loss += loss.data
            print('After', epoch + 1, 'epochs, the Average training loss is ', train_loss *
                params['batch_size'] / len(training_set), 'and average DICE score is', train_dice_score.data * params['batch_size'] / len(training_set))
            # Testing time
            start_time = time.time()
            count = 0
            for batch_fixed, batch_moving in validation_generator:
                # Transfer to GPU
                loss, dice = vm.get_test_loss(batch_moving, batch_fixed, epoch, count)
                val_dice_score += dice.data
                val_loss += loss.data
                count += 1
            print('After', epoch + 1, 'epochs, the Average validations loss is ', val_loss *
                1/ len(validation_set), 'and average DICE score is', val_dice_score.data * 1 / len(validation_set))
            
            avg_val_loss = val_loss * 1 / len(validation_set)
            
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                # torch.save(vm.voxelmorph.state_dict(), 'best_model-voxel.pth')
            
            wandb.log({"Train Loss": train_loss * params['batch_size'] / len(training_set),
                       "Train Dice Score": train_dice_score.data * params['batch_size'] / len(training_set),
                       "Validation Loss": val_loss * 1 / len(validation_set),
                       "Validation Dice Score": val_dice_score.data * 1 / len(validation_set)})
if __name__ == "__main__":
    main()
