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
import matplotlib.cm as cm
from VoxelMorph.voxelmorph2d import SpatialTransformation
from b_spline_interpolate import b_spline_interpolation
# import neurite as ne
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

def sample_control_points(image, num_points):
    """
    Uniformly sample control points from patches across the image.
    Args:
        image_size (tuple): (Batch size, Channels, Height, Width)
        num_points (int): Total number of control points to sample per image in the batch.
    Returns:
        Tensor: Control points of shape (B, num_points, 2)
    """
    B, H, W, C = image.shape

    num_patches = int(num_points ** 0.5)  # Determine grid size for patches
    patch_size = H // num_patches, W // num_patches

    control_points_batch = []

    for k in range(B):
        control_points = []
        for i in range(num_patches):
            for j in range(num_patches):
                x_start, y_start = i * patch_size[0], j * patch_size[1]
                x_end, y_end = min(x_start + patch_size[0], H), min(y_start + patch_size[1], W)

                # Randomly sample one point in the patch
                x_coord = torch.randint(x_start, x_end, (1,))
                y_coord = torch.randint(y_start, y_end, (1,))

                control_points.append(torch.tensor([x_coord.item(), y_coord.item()]))  # Collect as a list of tensors

        control_points = torch.stack(control_points, dim=0)  # (num_patches**2, 2)
        control_points = control_points[:num_points]  # Keep only the required number of points
        control_points_batch.append(control_points)

    control_points_batch = torch.stack(control_points_batch, dim=0)  # (B, num_points, 2)
    return control_points_batch


def normalize_control_points(control_points, image_size):
    """
    Normalize control points to the range [-1, 1].
    """
    _, H, W = image_size
    normalized_points = control_points.clone().float()
    normalized_points[..., 0] = normalized_points[..., 0] / H * 2 - 1  # Normalize x to [-1, 1]
    normalized_points[..., 1] = normalized_points[..., 1] / W * 2 - 1  # Normalize y to [-1, 1]
    return normalized_points

class ControlPointEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Input: BxNx2 (B: batch size, N: number of control points, 2: x,y coordinates)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: BxNx2
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Shape: BxNxhidden_dim
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
class DisplacementNet(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # Output: x,y displacement
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

def visualize_control_points(images, control_points, image_size, batch_idx=0):
        """
        Visualize control points over an image using Matplotlib.
        
        Args:
            images: Tensor of shape (B, C, H, W), batch of images.
            control_points: Tensor of shape (B, num_control_points, 2), control points normalized to [0, 1].
            image_size: Tuple (H, W), size of the images.
            batch_idx: Index of the batch element to visualize.
        """
        # print(image_size)
        C, H, W = image_size[0], image_size[1], image_size[2]

        # Get the specific image and control points from the batch
        image = images[batch_idx].cpu().numpy()  # Convert to numpy for visualization
        image = (image*255.0).astype(np.uint8)  # Convert to 8-bit for visualization
        points = control_points[batch_idx].cpu().numpy()  # Shape: (num_control_points, 2)

        # Denormalize control points
        # points[:, 0] *= W  # Scale x-coordinates
        # points[:, 1] *= H  # Scale y-coordinates

        # Plot the image
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.scatter(points[:, 0], points[:, 1], c='red', marker='o')  # Overlay control points
        # plt.title("Control Points Visualization")
        plt.axis("off")
        # plt.show()
        plt.savefig(f'./plots/test-1/control_points.png')

class IRnet(nn.Module):
    def __init__(self, image_size, num_points, num_patches, device):
        super(IRnet, self).__init__()
        self.image_size = image_size
        self.num_points = num_points
        self.num_patches = num_patches
        # self.spatial_transform = SpatialTransformation(use_gpu=True).to(device)
        self.device = device
        self.hidden_dim = 64
        # self.dnet = DisplacementNet(input_dim=2).to(self.device)
        self.coord_embed_net = ControlPointEncoder(hidden_dim=self.hidden_dim).to(self.device)
        self.feature_extractor = FeatureExtractor(in_ch = 2*1, out_ch=self.hidden_dim).to(self.device)
        self.dnet = DisplacementNet(2*self.hidden_dim).to(self.device)

    def forward(self, moving_image, fixed_image):
        control_points = sample_control_points(moving_image, self.num_points).to(self.device)
        # visualize_control_points(moving_image, control_points, self.image_size)
        norm_cp = normalize_control_points(control_points, self.image_size).to(self.device)
        
        moving_image = moving_image.permute(0, 3, 1, 2)
        fixed_image = fixed_image.permute(0, 3, 1, 2)
        combined_image = torch.cat((moving_image, fixed_image), dim=1)

        coord_embed = self.coord_embed_net(norm_cp)
        image_encode = self.feature_extractor(combined_image)
        B, N = control_points.shape[:2]
        sample_points = norm_cp.view(B, N, 1, 2)
        sampled_features = F.grid_sample(
            image_encode, 
            sample_points,
            mode='bicubic',
            padding_mode='reflection',
            align_corners=True
        ).squeeze(-1).transpose(1, 2) 
        combined_features = torch.cat((coord_embed, sampled_features), dim=-1)

        displacement = self.dnet(combined_features)

        dense_field = b_spline_interpolation(control_points, displacement, self.image_size)

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
    # sm = smooothing_loss(y)
    #print("CC Loss", cc, "Gradient Loss", sm)
    # loss = -1.0 * cc + lamda * sm
    # loss = mse + lamda * sm
    loss = mse
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
    axs[0].imshow(fixed, cmap='gray')
    axs[0].set_title('Fixed Image')
    axs[1].imshow(moving, cmap='gray')
    axs[1].set_title('Moving Image')
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title('Registered Image')
    d_ = xyd
    d_ = d_
    df= 6
    axs[3].quiver(d_.cpu().data[::df, ::df, 0], d_.cpu().data[::df, ::df, 1], scale=2, scale_units='xy', color='blue')
    # def plot_grid(x,y, ax=None, **kwargs):
    #     # ax = ax or plt.gca()
    #     segs1 = np.stack((x,y), axis=2)
    #     segs2 = segs1.transpose(1,0,2)
    #     ax.add_collection(LineCollection(segs1, **kwargs))
    #     ax.add_collection(LineCollection(segs2, **kwargs))
    #     ax.autoscale()

    # down_factor= 1/16.0
    # h_ = fixed.shape[0]
    # w_ = fixed.shape[1]
    # h_resize = int(down_factor*h_)
    # w_resize = int(down_factor*w_)

    
    # # print(xy_.shape) 
    # xy_ = xy_.squeeze(0)   
    # # print(xyd.shape)
    # grid_x = resize(xy_.cpu()[:,:,0].numpy(),(h_resize,w_resize))
    # grid_y = resize(xy_.cpu()[:,:,1].numpy(),(h_resize,w_resize))
    # distx = resize(xyd.cpu()[:,:,0].detach().numpy(),(h_resize,w_resize))
    # disty = resize(xyd.cpu()[:,:,1].detach().numpy(),(h_resize,w_resize))

    # distx += grid_x
    # disty += grid_y
    # # print(grid_x.shape)
    # # print(grid_y.shape)
    # # print(distx.shape)
    # # print(disty.shape)


    # # fig, ax = plt.subplots()
    # plot_grid(grid_x,grid_y, ax=axs[3],  color="lightgrey")
    # plot_grid(distx, disty, ax=axs[3], color="C0")

    # plt.savefig(f'./plots/test-1/{num_points}/Neural-Bspline-{epoch}-num_points-{num_points}.png')