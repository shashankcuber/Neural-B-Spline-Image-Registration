import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage import io
import numpy as np
import os
import time
import NeuralBspline as nb
import wandb
import matplotlib.pyplot as plt
import pickle

wandb.login(key = '<PASTE YOUR WANDB KEY>')

class Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, data_type="segmented"):
        'Initialization'
        self.list_IDs = list_IDs
        self.data_type = data_type

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        seg_path = './FUNDUS_seg_output/'
        fixed_image = torch.Tensor( resize(io.imread(seg_path + ID + '_1.png'), (256, 256, 1)))
        moving_image = torch.Tensor( resize(io.imread(seg_path + ID + '_2.png'), (256, 256, 1)))


        if self.data_type != "segmented":
            fixed_image = torch.Tensor(
                resize(io.imread('./Fire/' + ID + '_1.jpg'), (256, 256, 3)))
            moving_image = torch.Tensor(
                resize(io.imread('./Fire/' + ID + '_2.jpg'), (256, 256, 3)))
        
        return fixed_image, moving_image, ID
    

class Registration_Net():
    def __init__(self, input_channels, output_channels, train_config):
        self.nb = nb
        self.train_config = train_config
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = nb.IRnet(image_size=(1, 256, 256), num_points=train_config['num_points'], num_patches=train_config['num_patches'], device=self.device).to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=train_config['lr'], momentum=train_config['momentum'])
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_config['lr'])
        self.params = {'batch_size': 4,
                        'shuffle': True,
                        'num_workers': 1,
                        'worker_init_fn': np.random.seed(42)
                        }

    def forward(self, x):
        return self.model(x)
    
    def calculate_loss(self, y, ytrue):
        loss =  self.nb.vox_morph_loss(y, ytrue)
        return loss
    
    def ss_grid_gen(self,V):
        y_, x_ = np.meshgrid(np.arange(0, 256), np.arange(0, 256), indexing='ij')
        y_, x_ = 2.0*y_/255.0 - 1.0, 2.0*x_/255.0 - 1.0
        XY = torch.tensor(np.stack([x_,y_],axis=2),dtype=torch.float32).to(self.device)
        XY = XY.expand(V.shape[0],-1,-1,-1)
        with torch.no_grad():
            normv2 = V[:,:,:,0]**2 + V[:,:,:,1]**2+1e-10
            m = torch.sqrt(torch.max(normv2))
            n = torch.ceil(torch.log2(100.0*(XY.shape[1]+XY.shape[2])*m)).type(torch.int64)
            # avoid null values
            if n<0:
                n = 0    
                
        # Scale it (so it's close to 0)
        V = V / (2**n)
        V = V.permute(0, 2, 3, 1) # (B, H, W, 2)
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

    def train_model(self, batch_moving, batch_fixed, lamda = 0.01):
        self.optimizer.zero_grad()
        batch_fixed, batch_moving = batch_fixed.to(self.device), batch_moving.to(self.device)
        dense_field = self.model(batch_moving, batch_fixed)
        
        xyd, _ = self.ss_grid_gen(dense_field)
        xyd_ = self.normalize_dense_field(xyd)
        registered_image = F.grid_sample(batch_moving.permute(0, 3, 1, 2), xyd_, mode='bilinear', padding_mode='reflection', align_corners=True)
        
        train_loss = self.calculate_loss(registered_image.permute(0, 2, 3, 1), batch_fixed)
        
        train_loss.backward()
        self.optimizer.step()
        
        dice_score = self.nb.dice_score(registered_image.permute(0, 2, 3, 1), batch_fixed)
        
        return train_loss, dice_score
    
    def get_test_loss(self, batch_moving, batch_fixed, epoch, count, ID):
        with torch.no_grad():
            batch_moving, batch_fixed = batch_moving.to(self.device), batch_fixed.to(self.device)
            dense_field = self.model(batch_moving, batch_fixed)
            
            xyd, xy = self.ss_grid_gen(dense_field)
            xyd_ = self.normalize_dense_field(xyd)
            # xy_ = self.normalize_dense_field(xy)

            registered_image = F.grid_sample(batch_moving.permute(0, 3, 1, 2), xyd_, mode='bilinear', padding_mode='reflection', align_corners=True)
            
            val_loss = self.calculate_loss(registered_image.permute(0, 2, 3, 1), batch_fixed)

            val_dice_score = self.nb.dice_score(registered_image.permute(0, 2, 3, 1), batch_fixed)
            
            if count == 1 and epoch % 10 == 0:
                nb.visualise_results(batch_fixed[0], batch_moving[0], registered_image[0].permute(1, 2, 0), xy, xyd_[0], epoch, self.train_config['num_points'])
            
            # nb.visualise_results(batch_fixed[0], batch_moving[0], registered_image[0], xy_, dense_field[0], epoch)
            return val_loss, val_dice_score


def main():

    train_config = {
        "epochs":100,
        "batch_size": 4,
        "lr": 1e-5,
        "momentum": 0.99,
        "num_points": 50,
        "num_patches": 32,
        "device": "cuda:1",
        "train_size": len(training_generator),
        "val_size": len(validation_generator),
        "data_type": "segmented",
    }

    params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 2,
                'worker_init_fn': np.random.seed(42)
                }
    
    filename = list(set([x.split('_')[0] for x in os.listdir('./Fire-segmented/')]))

    print(len(filename))

    partition = {}

    # Storing the train and validation data file names for reproducibility
    partition['train'], partition['validation'] = train_test_split( filename, test_size=0.1, random_state=42)

    with open('partition.pkl', 'wb') as f:
        pickle.dump(partition, f)
    
    with open('partition.pkl', 'rb') as f:
        partition = pickle.load(f)

    print(len(partition['train']))
    print(len(partition['validation']))
    # Generators
    training_set = Dataset(partition['train'], data_type="segmented")
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'], data_type="segmented")
    validation_generator = data.DataLoader(validation_set, batch_size=1, shuffle=False)

    rnet = Registration_Net(input_channels=3, output_channels=2, train_config=train_config)

    # Turn it to True to test the model
    test = False
    if test == True:
        train_config["epochs"] = 1

    with wandb.init(project="Neural-B-spline-Image-Registration", config=train_config, name=f"Neural-Bspline-{train_config['num_points']}-num_points") as run:
        min_val_loss = float('inf')
        for epoch in range(train_config["epochs"]):
            if test == False:
                start_time = time.time()
                train_loss = 0
                train_dice_score = 0
                val_loss = 0
                val_dice_score = 0
                for batch_fixed, batch_moving, ID in training_generator:
                    loss, dice = rnet.train_model(batch_moving, batch_fixed)
                    train_dice_score += dice.data
                    train_loss += loss.data
                avg_train_loss = train_loss / len(training_generator)
                avg_train_dice_score = train_dice_score / len(training_generator)
                print("Epoch: ", epoch, "Train Loss: ", avg_train_loss, "Train Dice Score: ", avg_train_dice_score)

            start_time = time.time()
            count = 0
            for batch_fixed, batch_moving, ID in validation_generator:
                loss, dice = rnet.get_test_loss(batch_moving, batch_fixed, epoch, count, ID)
                count += 1
                val_dice_score += dice.data
                val_loss += loss.data
            
            avg_val_loss = val_loss / len(validation_set)
            avg_val_dice_score = val_dice_score / len(validation_set)
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                # torch.save(rnet.model.state_dict(), f"./best_model/best_model_{train_config['num_points']}_num_points.pth")

            print("Epoch: ", epoch, "Validation Loss: ", avg_val_loss, "Validation Dice Score: ", avg_val_dice_score)
            wandb.log(
                {
                    "Train Loss": avg_train_loss,
                    "Train Dice Score": avg_train_dice_score,
                    "Validation Loss": avg_val_loss,
                    "Validation Dice Score": avg_val_dice_score
                }
            )

if __name__ == '__main__':
    main()