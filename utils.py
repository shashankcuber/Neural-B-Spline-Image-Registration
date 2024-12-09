import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.transform import resize
from matplotlib.collections import LineCollection

random.seed(56)
torch.manual_seed(56)
np.random.seed(56)
# class PairedImageDataset(Dataset):
#     def __init__(self, root_dir, image_size , transform=False):
#         """
#         Args:
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_pairs = self._load_image_pairs()
#         self.image_size = image_size

#     def _load_image_pairs(self):
#         """
#         Load all image pairs from the root directory. Assumes that each fixed image has 
#         the name format 'X_1.jpg' and the corresponding moving image has the name 'X_2.jpg'.
#         """
#         pairs = []
#         for filename in os.listdir(self.root_dir):
#             if filename.endswith('_1.jpg'):
#                 base_name = filename[:-6]  # Remove '_1.jpg' to get the base name
#                 fixed_image_path = os.path.join(self.root_dir, f"{base_name}_1.jpg")
#                 moving_image_path = os.path.join(self.root_dir, f"{base_name}_2.jpg")
                
#                 if os.path.exists(fixed_image_path) and os.path.exists(moving_image_path):
#                     pairs.append((fixed_image_path, moving_image_path))
#                 else:
#                     print(f"Warning: Missing pair for {base_name}")
#             # print(pairs)
#         return pairs[:5]

#     def __len__(self):
#         return len(self.image_pairs)

#     def __getitem__(self, idx):
#         """
#         Fetches a fixed-moving image pair and applies transformations.
#         """
#         fixed_image_path, moving_image_path = self.image_pairs[idx]
        
#         # Load images
#         fixed_image = Image.open(fixed_image_path).convert('L')  # Assuming grayscale images
#         moving_image = Image.open(moving_image_path).convert('L')
        
#         transform = transforms.Compose([
#                 transforms.Resize(self.image_size),  # Resize images to model input size
#                 transforms.ToTensor()           # Convert to tensor
#             ])
#         # Apply transformations if provided
#         if self.transform:
#             fixed_image = transform(fixed_image)
#             moving_image = transform(moving_image)
        
#         return {'fixed': fixed_image, 'moving': moving_image, 'fixed_path': fixed_image_path, 'moving_path': moving_image_path}

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PairedImageDataset(Dataset):
    def __init__(self, root_dir, image_size, transform=False, normalize=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_size (tuple): Size to resize the images to.
            transform (bool, optional): Whether to apply transforms.
            normalize (bool, optional): Whether to normalize images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize
        self.image_pairs = self._load_image_pairs()
        self.image_size = image_size

        # Define normalization parameters
        self.mean = 0.2553
        self.std = 0.1918

    def _load_image_pairs(self):
        """
        Load all image pairs from the root directory. Assumes that each fixed image has 
        the name format 'X_1.jpg' and the corresponding moving image has the name 'X_2.jpg'.
        """
        pairs = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith('_1.jpg'):
                base_name = filename[:-6]  # Remove '_1.jpg' to get the base name
                fixed_image_path = os.path.join(self.root_dir, f"{base_name}_1.jpg")
                moving_image_path = os.path.join(self.root_dir, f"{base_name}_2.jpg")
                
                if os.path.exists(fixed_image_path) and os.path.exists(moving_image_path):
                    pairs.append((fixed_image_path, moving_image_path))
                else:
                    print(f"Warning: Missing pair for {base_name}")
        return pairs[5:11]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Fetches a fixed-moving image pair and applies transformations.
        """
        fixed_image_path, moving_image_path = self.image_pairs[idx]
        
        # Load images using cv2
        fixed_image = cv2.imread(fixed_image_path, cv2.IMREAD_COLOR) 
        moving_image = cv2.imread(moving_image_path, cv2.IMREAD_COLOR) 

        fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
        moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY)
        # Resize images
        fixed_image = cv2.resize(fixed_image, self.image_size)
        moving_image = cv2.resize(moving_image, self.image_size)

        fixed_image = fixed_image.astype('float32') / 255.0
        moving_image = moving_image.astype('float32') / 255.0

        # Convert to tensor
        # fixed_image = torch.tensor(fixed_image, dtype=torch.float32).unsqueeze(0)
        # moving_image = torch.tensor(moving_image, dtype=torch.float32).unsqueeze(0)

        return {'fixed': transforms.ToTensor() (fixed_image), 'moving': transforms.ToTensor() (moving_image), 'fixed_path': fixed_image_path, 'moving_path': moving_image_path}


def create_dataloaders(train_config):
    # Load the dataset
    dataset = PairedImageDataset(root_dir=train_config['root_dir'], image_size= train_config['image_size'], transform=True)
    print(len(dataset))
    # Create DataLoader
    # dataset = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0)

    train_size = int(train_config['split-ratio'] * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader

def warp_image(image, flow, DEVICE):
    N, C, H, W = image.size()
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack((x, y), 2).float().to(DEVICE) 
    # grid = create_mesh_grid(H, W).unsqueeze(0).to(image.device)  # (1, H, W, 2)
    # make grid of size (N, H, W, 2)
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)
    # print(grid.shape)
    # print(flow.shape)
    # print(flow.permute(0, 2, 3, 1).shape)
    grid = grid + flow.permute(0, 2, 3, 1)  # Add offsets to the grid
    grid[:, :, :, 0] = grid[:, :, :, 0] / (W - 1) * 2 - 1 
    grid[:, :, :, 1] = grid[:, :, :, 1] / (H - 1) * 2 - 1  
    warped_image = F.grid_sample(image, grid, align_corners=True, mode='bilinear', padding_mode='reflection')
    return warped_image, x, y

def plot_grid(x,y, ax=None, **kwargs):
    # x = x.cpu().numpy()
    # y = y.cpu().numpy()
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

def make_identity_flow(H, W):
    h_ = H
    w_ = W
    y_, x_ = np.meshgrid(np.arange(0,h_), np.arange(0,w_),indexing='ij')
    y_, x_ = 2.0*y_/(h_-1) - 1.0, 2.0*x_/(w_-1) - 1.0

    xy_ = torch.tensor(np.stack([x_,y_],2),dtype=torch.float32).unsqueeze(0)
    
    return xy_


# def plot_figures(predicted_img_np, low_img_np, high_img_np,  predicted_flow_np, epoch, train_config, grid_x=None, grid_y=None):
    #  Plot images
    # predicted_img_np = predicted_img_np.detach().cpu() * 255.0
    # low_img_np = low_img_np.detach().cpu() * 255.0
    # high_img_np = high_img_np.detach().cpu() * 255.0

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(low_img_np, cmap='gray')
    axs[0].set_title('Moving_image')
    axs[0].axis('off')

    axs[1].imshow(high_img_np, cmap='gray')
    axs[1].set_title('Fixed Image')
    axs[1].axis('off')

    axs[2].imshow(predicted_img_np, cmap='gray')
    axs[2].set_title('Registered Image')
    axs[2].axis('off')

    # Professor Method
    #quiver plot
    df = 5
    xy_ = make_identity_flow(predicted_flow_np.shape[1], predicted_flow_np.shape[2])
    xy_ = xy_.cpu().numpy()
    d_ = (predicted_flow_np - xy_).squeeze()
    axs[3].quiver(d_.cpu().data[::df,::df,0], d_.cpu().data[::df,::df,1],color='r')
    axs[3].axis('equal')
    axs[3].set_title('Optical Flow Field')

    # deformable grid
    down_factor= 0.125
    h_ = predicted_flow_np.shape[2]
    w_ = predicted_flow_np.shape[3]
    h_resize = int(down_factor*h_)
    w_resize = int(down_factor*w_)

    grid_x = resize(xy_.cpu()[:,:,:,0].squeeze().numpy(),(h_resize,w_resize))
    grid_y = resize(xy_.cpu()[:,:,:,1].squeeze().numpy(),(h_resize,w_resize))
    distx = resize(predicted_flow_np.cpu()[:,:,:,0].squeeze().detach().numpy(),(h_resize,w_resize))
    disty = resize(predicted_flow_np.cpu()[:,:,:,1].squeeze().detach().numpy(),(h_resize,w_resize))
    plot_grid(grid_x,grid_y, ax=axs[4],  color="lightgrey")
    plot_grid(distx, disty, ax=axs[4], color="C0")

    plt.savefig(f'./plots/test/test-{epoch}.png', dpi=100)
    #### MY METHOD ####
    # grid = torch.stack((grid_x, grid_y)).squeeze().cpu().numpy()
    # print(grid.shape)
    # # print(predicted_flow_np.shape)

    # # print(predicted_flow_np)
    # grid_x = grid_x.cpu().numpy()
    # grid_y = grid_y.cpu().numpy()
    # step = 5
    # grid_x = grid_x[::step, ::step]
    # grid_y = grid_y[::step, ::step]
    
    # # predicted_flow_np = predicted_flow_np / (IMAGE_SIZE - 1) * 2 - 1
    # U = predicted_flow_np[0, :, :]
    # V = predicted_flow_np[1, :, :]

    # U_ = U[::step, ::step]
    # V_ = V[::step, ::step]

    # U_ = U_ * 2
    # V_ = V_ * 2
    # U_ = U_.cpu().numpy()
    # V_ = V_.cpu().numpy()
    # axs[3].quiver(grid_x, grid_y, U_, V_, angles='xy', scale_units='xy', color='C0')
    # axs[3].set_title('Optical Flow Field')


    # # Normalize grid and predicted flow for visualization in range -1 to 1 for plot_grid
    # grid_x = grid_x / (256 - 1) * 2 - 1
    # grid_y = grid_y / (256 - 1) * 2 - 1

    

    # w_ = predicted_flow_np.shape[2] 
    # h_ = predicted_flow_np.shape[1]
    # y_, x_ = np.meshgrid(np.arange(0,h_), np.arange(0,w_),indexing='ij')
    # y_, x_ = 2.0*y_/(h_-1) - 1.0, 2.0*x_/(w_-1) - 1.0

    # xy_ = torch.tensor(np.stack([x_,y_],2),dtype=torch.float32).unsqueeze(0)

    # down_factor=0.125
    # h_resize = int(down_factor*h_)
    # w_resize = int(down_factor*w_)

    # grid_x = resize(xy_.cpu()[:,:,:,0].squeeze().numpy(),(h_resize,w_resize))
    # grid_y = resize(xy_.cpu()[:,:,:,1].squeeze().numpy(),(h_resize,w_resize))
    # # distx = resize(xyd.cpu()[:,:,:,0].squeeze().detach().numpy(),(h_resize,w_resize))
    # # disty = resize(xyd.cpu()[:,:,:,1].squeeze().detach().numpy(),(h_resize,w_resize))
    # U = U.cpu().numpy()
    # V = V.cpu().numpy()
    # U = resize(U,(h_resize,w_resize))
    # V = resize(V,(h_resize,w_resize))

    # U = grid_x + U
    # V = grid_y + V


    # # U = 2.0 * U / (w_ - 1)
    # # V = 2.0 * V / (h_ - 1)   

    # # grid_x = grid_x.cpu().numpy()
    # # grid_y = grid_y.cpu().numpy()

    # # Normalize between -1 and 1 U and V
    # plot_grid(grid_x, grid_y, ax=axs[4], color='lightgrey')
    # plot_grid(U, V, ax=axs[4], color='C0')
    # plot_grid(predicted_flow_np[0, :, :], predicted_flow_np[0, :, :], ax=axs[4], color='C0')

    # plt.savefig(f'./plots/{train_config["num_control_points"]}/res_{train_config["image_size"]}_basis_{train_config["basis"]},_sigma_{train_config["sigma"]}_epoch={epoch}.png', dpi=400)
    # plt.savefig(f'./plots/test-{epoch}.png', dpi=100)
    # plt.savefig(f'./plots/{train_config["num_control_points"]}/res_{train_config["image_size"]}_basis_{train_config['basis']},_sigma_{train_config["sigma"]}_epoch={epoch}.png', dpi=400)


def plot_figures(registered_img_np, moving_img_np, fixed_img_np,  predicted_flow_np, grid_x, grid_y, epoch, device, train_config):
    # Plot images
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(moving_img_np.astype(np.uint8), cmap='gray')
    axs[0].set_title('Moving Image')
    axs[0].axis('off')

    axs[1].imshow(fixed_img_np.astype(np.uint8), cmap='gray')
    axs[1].set_title('Fixed Image')
    axs[1].axis('off')

    axs[2].imshow(registered_img_np.astype(np.uint8), cmap='gray')
    axs[2].set_title('Predicted Registered Image')
    axs[2].axis('off')
    # grid = torch.stack((grid_x, grid_y)).squeeze().cpu().numpy()
    # print(grid.shape)
    # print(predicted_flow_np.shape)

    # print(predicted_flow_np)
    grid_x = grid_x.cpu().numpy()
    grid_y = grid_y.cpu().numpy()
    step = 5
    grid_x = grid_x[::step, ::step]
    grid_y = grid_y[::step, ::step]
    
    # predicted_flow_np = predicted_flow_np / (IMAGE_SIZE - 1) * 2 - 1
    U = predicted_flow_np[0, :, :]
    V = predicted_flow_np[1, :, :]

    U_ = U[::step, ::step]
    V_ = V[::step, ::step]

    U_ = U_ * 2
    V_ = V_ * 2

    axs[3].quiver(grid_x, grid_y, U_, V_, angles='xy', scale_units='xy', color='C0')
    axs[3].set_title('Optical Flow Field')


    # Normalize grid and predicted flow for visualization in range -1 to 1 for plot_grid
    # grid_x = grid_x / (IMAGE_SIZE - 1) * 2 - 1
    # grid_y = grid_y / (IMAGE_SIZE - 1) * 2 - 1

    

    w_ = predicted_flow_np.shape[2] 
    h_ = predicted_flow_np.shape[1]
    y_, x_ = np.meshgrid(np.arange(0,h_), np.arange(0,w_),indexing='ij')
    y_, x_ = 2.0*y_/(h_-1) - 1.0, 2.0*x_/(w_-1) - 1.0

    xy_ = torch.tensor(np.stack([x_,y_],2),dtype=torch.float32).unsqueeze(0).to(device) 

    down_factor=0.125
    h_resize = int(down_factor*h_)
    w_resize = int(down_factor*w_)

    grid_x = resize(xy_.cpu()[:,:,:,0].squeeze().numpy(),(h_resize,w_resize))
    grid_y = resize(xy_.cpu()[:,:,:,1].squeeze().numpy(),(h_resize,w_resize))
    # distx = resize(xyd.cpu()[:,:,:,0].squeeze().detach().numpy(),(h_resize,w_resize))
    # disty = resize(xyd.cpu()[:,:,:,1].squeeze().detach().numpy(),(h_resize,w_resize))
    U = resize(U,(h_resize,w_resize))
    V = resize(V,(h_resize,w_resize))

    U = grid_x + U
    V = grid_y + V


    # U = 2.0 * U / (w_ - 1)
    # V = 2.0 * V / (h_ - 1)   

    
    # Normalize between -1 and 1 U and V
    plot_grid(grid_x, grid_y, ax=axs[4], color='lightgrey')
    plot_grid(U, V, ax=axs[4], color='C0')
    # plot_grid(predicted_flow_np[0, :, :], predicted_flow_np[0, :, :], ax=axs[4], color='C0')

    plt.savefig(f'./plots/test/test-{epoch}.png', dpi=100)
    
# Load the dataset
# root_dir = './FIRE/Images'
# dataset = PairedImageDataset(root_dir=root_dir, transform=True)

# # Create DataLoader
# batch_size = 4
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# train_config = {
#     'root_dir': './FIRE/Images',
#     'batch_size': 4,
#     'split-ratio': 0.9,
#     'epochs': 1,
#     'lr': 1e-4,
#     'weight_decay': 1e-5
# }
# train_loader, val_loader = create_dataloaders(train_config)
# # Example usage
# for batch in train_loader:
#     fixed_images = batch['fixed']
#     moving_images = batch['moving']
#     fixed_paths = batch['fixed_path']
#     moving_paths = batch['moving_path']
#     print(type(fixed_images[0]))
#     print(fixed_images.shape)
#     print(moving_images.shape)
#     print(fixed_paths[0])
#     print(moving_paths[0])
#     I_f = fixed_images[0].permute(1, 2, 0)
#     I_m = moving_images[0].permute(1, 2, 0)
#     f, ax = plt.subplots(1, 2)
#     ax[0].imshow(I_f, cmap='gray')
#     ax[1].imshow(I_m, cmap='gray')
#     plt.show()  # Should print [batch_size, 1, 256, 256]
#     break