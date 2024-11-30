import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torchvision.utils import make_grid
from skimage.transform import resize
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from VoxelMorph.voxelmorph2d import SpatialTransformation
from b_spline_interpolate import b_spline_interpolation
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def sample_control_points(image_size, num_points):
    """
    Uniformly sample control points from patches across the image.
    Args:
        image_size (tuple): (Batch size, Height, Width)
        num_points (int): Total number of control points to sample.
    Returns:
        Tensor: Control points of shape (B, num_points, 2)
    """
    B, H, W = image_size
    num_patches = int(num_points ** 0.5)  # Determine grid size for patches
    patch_size = H // num_patches, W // num_patches

    control_points = []
    for i in range(num_patches):
        for j in range(num_patches):
            x_start, y_start = i * patch_size[0], j * patch_size[1]
            x_end, y_end = min(x_start + patch_size[0], H), min(y_start + patch_size[1], W)

            # Randomly sample one point in the patch
            x_coord = torch.randint(x_start, x_end, (B, 1))
            y_coord = torch.randint(y_start, y_end, (B, 1))

            control_points.append(torch.cat((x_coord, y_coord), dim=-1))

    control_points = torch.stack(control_points, dim=1)  # (B, N, 2)
    return control_points[:, :num_points, :]  # Keep only the required number of points


def normalize_control_points(control_points, image_size):
    """
    Normalize control points to the range [-1, 1].
    """
    _, H, W = image_size
    normalized_points = control_points.clone().float()
    normalized_points[..., 0] = normalized_points[..., 0] / H * 2 - 1  # Normalize x to [-1, 1]
    normalized_points[..., 1] = normalized_points[..., 1] / W * 2 - 1  # Normalize y to [-1, 1]
    return normalized_points

class DisplacementNet(nn.Module):
    def __init__(self, input_dim=2):
        super(DisplacementNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 2)
    
    def forward(self, control_points):
        # x = F.tanh(self.fc1(control_points))
        # x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))
        # displacement = self.fc4(x)
        # return displacement
        x = F.relu(self.fc1(control_points))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        displacement = self.fc8(x)
        return displacement

def interpolate_dense_field(control_points, displacements, image_size, method='nearest'):
    """
    Interpolate scattered control points to a dense deformation field.
    """
    B, N, _ = control_points.shape
    B, H, W = image_size

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1, 1, H, device=control_points.device),
        torch.linspace(-1, 1, W, device=control_points.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
    # print(grid.shape)
    # print(displacements.shape)
    # Use scattered interpolation (e.g., bilinear)
    dense_field = F.grid_sample(
        displacements.unsqueeze(0).permute(0, 3, 1, 2),
        grid.unsqueeze(0).permute(0, 1, 2, 3),
        mode=method,
        align_corners=True
    )

    return dense_field

class IRnet(nn.Module):
    def __init__(self, image_size, num_points, num_patches, device):
        super(IRnet, self).__init__()
        self.image_size = image_size
        self.num_points = num_points
        self.num_patches = num_patches
        # self.spatial_transform = SpatialTransformation(use_gpu=True).to(device)
        self.device = device
        self.dnet = DisplacementNet(input_dim=2).to(self.device)
    def forward(self, moving_image):
        control_points = sample_control_points(self.image_size, self.num_points).to(self.device)
        norm_cp = normalize_control_points(control_points, self.image_size).to(self.device)
        displacement = self.dnet(norm_cp)
        # dense_field = interpolate_dense_field(control_points, displacement, self.image_size)
        dense_field = b_spline_interpolation(control_points, displacement, self.image_size)
        dense_field = dense_field.permute(0, 2, 3, 1)  # (B, H, W, 2)
        # print(moving_image.shape)
        # registered_image = self.spatial_transform(moving_image, dense_field)
        return dense_field

def dice_score(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch

    """
    top = 2 *  torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    #print("Dice score", dice)
    return dice

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d/2.0

def vox_morph_loss(y, ytrue, lamda=0.01):
    mse = F.mse_loss(y, ytrue)
    # cc = cross_correlation_loss(y, ytrue, n)
    sm = smooothing_loss(y)
    #print("CC Loss", cc, "Gradient Loss", sm)
    # loss = -1.0 * cc + lamda * sm
    loss = mse + lamda * sm
    return loss

def visualise_results(fixed, moving, pred, xy_, xyd, epoch, num_points):
    """
    Visualize the results of registration.
    """
    fixed = fixed.cpu().detach().numpy()
    moving = moving.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()

    fixed = (fixed * 255.0).astype(np.uint8)
    moving = (moving * 255.0).astype(np.uint8)
    pred = (pred * 255.0).astype(np.uint8)

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(fixed)
    axs[0].set_title('Fixed Image')
    axs[1].imshow(moving)
    axs[1].set_title('Moving Image')
    axs[2].imshow(pred)
    axs[2].set_title('Registered Image')

    d_ = xyd
    d_ = d_ * 5.0
    df= 8
    axs[3].quiver(d_.cpu().data[::df, ::df, 0], d_.cpu().data[::df, ::df, 1], scale=1, scale_units='xy', color='r')
    # def plot_grid(x,y, ax=None, **kwargs):
    #     # ax = ax or plt.gca()
    #     segs1 = np.stack((x,y), axis=2)
    #     segs2 = segs1.transpose(1,0,2)
    #     ax.add_collection(LineCollection(segs1, **kwargs))
    #     ax.add_collection(LineCollection(segs2, **kwargs))
    #     ax.autoscale()

    # down_factor=0.125
    # h_ = fixed.shape[0]
    # w_ = fixed.shape[1]
    # h_resize = int(down_factor*h_)
    # w_resize = int(down_factor*w_)

    
    # # print(xy_.shape)    
    # # print(xyd.shape)

    # grid_x = resize(xy_.cpu()[:,:,0].numpy(),(h_resize,w_resize))
    # grid_y = resize(xy_.cpu()[:,:,1].numpy(),(h_resize,w_resize))
    # distx = resize(xyd.cpu()[:,:,0].detach().numpy(),(h_resize,w_resize))
    # disty = resize(xyd.cpu()[:,:,1].detach().numpy(),(h_resize,w_resize))

    # # print(grid_x.shape)
    # # print(grid_y.shape)
    # # print(distx.shape)
    # # print(disty.shape)


    # # fig, ax = plt.subplots()
    # plot_grid(grid_x,grid_y, ax=axs[3],  color="lightgrey")
    # plot_grid(distx, disty, ax=axs[3], color="C0")

    plt.savefig(f'./plots/test-1/{num_points}/Neural-Bspline-{epoch}-num_points-{num_points}.png')