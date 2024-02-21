import numpy as np
import torch
from torch import nn

def _mesh_grid(B:int, H:int, W:int, D:int)->np.array:
    # batches not implented
    x = torch.arange(H)
    y = torch.arange(W)
    z = torch.arange(D)
    mesh = torch.stack(torch.meshgrid(x, y, z)[::-1], 0) 

    mesh = mesh.unsqueeze(0)
    return mesh.repeat([B,1,1,1,1])

def _norm_grid(v_grid:np.array)->np.array:
    """scale grid to [-1,1]"""
    _, _, H, W, D = v_grid.size()
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :, :] = (2.0 * v_grid[:, 0, :, :, :] / (D - 1)) - 1.0 
    v_grid_norm[:, 1, :, :, :] = (2.0 * v_grid[:, 1, :, :, :] / (W - 1)) - 1.0
    v_grid_norm[:, 2, :, :, :] = (2.0 * v_grid[:, 2, :, :, :] / (H - 1)) - 1.0 
    
    return v_grid_norm.permute(0, 2, 3, 4, 1)

def warp(image:np.array, flow:np.array, warping_borders_pad:str, warping_interp_mode:str) -> np.array: 
    flow = np.rollaxis(flow,-1)
    flow = torch.tensor(flow)
    flow = torch.unsqueeze(flow,0)
    image = torch.tensor(image)
    image = torch.unsqueeze(torch.unsqueeze(image,0),0)

    B, _, H, W, D = flow.size()
    flow = torch.flip(flow, [1]) # flow is now z, y, x
    base_grid = _mesh_grid(B, H, W, D).type_as(image)  # B2HW
    grid_plus_flow = base_grid + flow
    v_grid = _norm_grid(grid_plus_flow)  # BHW2
    image_warped = nn.functional.grid_sample(image, v_grid, mode=warping_interp_mode, padding_mode=warping_borders_pad, align_corners=False)

    return image_warped[0,0,:,:,:].cpu().numpy()