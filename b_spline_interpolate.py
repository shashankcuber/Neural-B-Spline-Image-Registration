import scipy.interpolate
import torch
import torch.nn.functional as F
import scipy 
import numpy as np


def b_spline_basis(x, order):
    """
    Compute the p-th order B-spline basis function recursively.
    Args:
        x: Tensor of distances between grid points and control points.
        order: The order of the B-spline basis (p).
    Returns:
        Tensor of p-th order B-spline weights.
    """
    if order == 0:
        # Zero-order basis (piecewise constant)
        return (x >= 0) & (x < 1)  # 1 if 0 <= x < 1, else 0

    # Recursive definition of p-th order basis
    b_p_minus_1_x = b_spline_basis(x, order - 1)
    b_p_minus_1_x_minus_1 = b_spline_basis(x - 1, order - 1)
    
    return (x / order) * b_p_minus_1_x + ((order + 1 - x) / order) * b_p_minus_1_x_minus_1

def b_spline_interpolation(control_points, displacements, image_size=(3, 256, 256)):
    """
    Perform B-spline interpolation to generate a dense deformation field from sparse displacements.
    
    Parameters:
    - control_points: Tensor of shape (B, num_control_points, 2), where each point is (x, y).
    - displacements: Tensor of shape (B, num_control_points, 2), with (dx, dy) for each control point.
    - image_size: Tuple, the spatial dimensions of the output deformation field (height, width).
    
    Returns:
    - dense_field: Dense deformation field of shape (B, 2, H, W).
    """
    B, num_control_points, _ = control_points.shape
    C, H, W = image_size
    # denormalize control points
    # control_points = control_points * torch.tensor([H], device=control_points.device)
    
    # Initialize dense deformation field (dx, dy) for each pixel
    dense_field = torch.zeros((B, 2, H, W), device=control_points.device)
    weight_sum = torch.zeros((B, 1, H, W), device=control_points.device)
    # Create a meshgrid for the pixel coordinates
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=control_points.device),
                                    torch.arange(W, device=control_points.device))
    grid = torch.stack((x_grid, y_grid), dim=-1).float()  # Shape: (H, W, 2)

    # Perform B-spline interpolation for each batch
    for b in range(B):
        for i in range(num_control_points):
            # Extract the control point and its displacement
            cp = control_points[b, i]  # (x, y)
            disp = displacements[b, i]  # (dx, dy)

            # Compute distances from the control point to all pixels
            distances_x = grid[..., 0] - cp[0]  # Difference in x-coordinates
            distances_y = grid[..., 1] - cp[1]  # Difference in y-coordinates

            # Apply the cubic B-spline basis to compute weights
            weights_x = b_spline_basis(distances_x, order=3)
            weights_y = b_spline_basis(distances_y, order=3)
            weights = weights_x + weights_y  # Combined weight (outer product)

            # Accumulate the weighted displacements into the dense field
            dense_field[b, 0] += weights * disp[0]  # dx contribution
            dense_field[b, 1] += weights * disp[1]  # dy contribution
            weight_sum[b, 0] += weights  # Sum of weights for normalization

    # Normalize the dense field by the sum of weights
    dense_field[:, 0] /= (weight_sum[:, 0] + 1e-6)  # Normalize dx
    dense_field[:, 1] /= (weight_sum[:, 0] + 1e-6)  # Normalize dy


    return dense_field  # Output shape: (B, 2, H, W)
