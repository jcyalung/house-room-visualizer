"""
Lightweight NeRF Implementation
A simplified, fast-training NeRF for quick experimentation.

Based on the minimal NeRF architecture but optimized for speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LightweightNeRF(nn.Module):
    """Lightweight NeRF model with fewer parameters."""
    
    def __init__(self, D=4, W=128, input_ch=3, input_ch_views=3, output_ch=4, skips=[2]):
        """
        D: depth (number of layers)
        W: width (number of neurons per layer)
        input_ch: input channels for 3D position (x, y, z)
        input_ch_views: input channels for viewing direction
        output_ch: output channels (RGB + density)
        skips: layers to add skip connections
        """
        super(LightweightNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        
        # Position encoding dimensions (reduced for speed)
        self.embed_fn = None
        self.embeddirs_fn = None
        
        # Main network (smaller than standard NeRF)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) 
             for i in range(D-1)]
        )
        
        # View-dependent network (smaller)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
        # Output layers
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        alpha = self.alpha_linear(h)
        
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)
        
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        
        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)
        
        return outputs


def get_rays(H, W, K, c2w):
    """Get ray origins and directions."""
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), 
                          torch.linspace(0, H-1, H), indexing='xy')
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    dirs = dirs.unsqueeze(-2)  # Add dimension for broadcasting
    rays_d = torch.sum(dirs * c2w[:3,:3].unsqueeze(0).unsqueeze(0), -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def render_rays(rays_o, rays_d, near, far, N_samples, model, chunk=1024*32):
    """Render rays through the NeRF model."""
    # Sample points along rays
    t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    
    # Expand dimensions for broadcasting
    rays_o_expanded = rays_o.unsqueeze(-2)  # [N_rays, 1, 3]
    rays_d_expanded = rays_d.unsqueeze(-2)  # [N_rays, 1, 3]
    z_vals_expanded = z_vals.unsqueeze(0)   # [1, N_samples]
    
    pts = rays_o_expanded + rays_d_expanded * z_vals_expanded.unsqueeze(-1)
    
    # Flatten for network
    pts_flat = torch.reshape(pts, [-1, 3])
    
    # Get viewing directions (normalize)
    dirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-10)
    dirs_flat = dirs.unsqueeze(-2).expand(pts.shape).reshape(-1, 3)
    
    # Run network in chunks
    raw = []
    for i in range(0, pts_flat.shape[0], chunk):
        pts_chunk = pts_flat[i:i+chunk]
        dirs_chunk = dirs_flat[i:i+chunk]
        x = torch.cat([pts_chunk, dirs_chunk], -1)
        raw_chunk = model(x)
        raw.append(raw_chunk)
    raw = torch.cat(raw, 0)
    
    # Reshape
    raw = torch.reshape(raw, list(pts.shape[:-1]) + [4])
    
    # Volume rendering
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)
    
    return rgb_map, disp_map, acc_map


def raw2outputs(raw, z_vals, rays_d):
    """Convert raw NeRF output to RGB image."""
    rgb = torch.sigmoid(raw[...,:3])
    alpha = 1. - torch.exp(-F.relu(raw[...,3]))
    
    # Transmittance
    ones = torch.ones((alpha.shape[0], 1), device=alpha.device)
    transmittance = torch.cat([ones, 1.-alpha + 1e-10], -1)
    transmittance = torch.cumprod(transmittance, -1)[:, :-1]
    
    weights = alpha * transmittance
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (acc_map + 1e-10))
    
    return rgb_map, disp_map, acc_map, weights, depth_map


def load_data(data_dir):
    """Load NeRF data from transforms files."""
    data_dir = Path(data_dir)
    
    with open(data_dir / "transforms_train.json", 'r') as f:
        meta = json.load(f)
    
    images = []
    poses = []
    
    for frame in meta['frames']:
        fname = data_dir / frame['file_path'].replace('./', '')
        if fname.exists():
            img = imageio.imread(fname)
            images.append(img)
            poses.append(np.array(frame['transform_matrix']))
    
    images = np.array(images).astype(np.float32) / 255.0
    poses = np.array(poses).astype(np.float32)
    
    H, W = images[0].shape[:2]
    camera_angle_x = meta['camera_angle_x']
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    return images, poses, [H, W, focal]


def train(model, images, poses, hwf, N_iters=10000, lr=5e-4):
    """Train lightweight NeRF."""
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ], dtype=np.float32)
    K = torch.Tensor(K).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    images_t = torch.Tensor(images).to(device)
    poses_t = torch.Tensor(poses).to(device)
    
    # Training loop
    pbar = tqdm(range(N_iters), desc="Training")
    for i in pbar:
        # Random image
        img_i = np.random.randint(images.shape[0])
        target = images_t[img_i]
        pose = poses_t[img_i]
        
        # Get rays
        c2w = pose
        rays_o, rays_d = get_rays(H, W, K, c2w)
        
        # Sample random rays
        N_rand = 1024
        select_inds = np.random.choice(H * W, size=[N_rand], replace=False)
        rays_o_flat = rays_o.reshape(-1, 3)[select_inds]
        rays_d_flat = rays_d.reshape(-1, 3)[select_inds]
        target_s = target.reshape(-1, 3)[select_inds]
        
        # Render
        rgb, _, _ = render_rays(rays_o_flat, rays_d_flat, 2., 6., 64, model)
        
        # Loss
        loss = F.mse_loss(rgb, target_s)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        if i % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="nerf_data")
    parser.add_argument("--N_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    
    args = parser.parse_args()
    
    print("Loading data...")
    images, poses, hwf = load_data(args.data_dir)
    
    print("Creating model...")
    model = LightweightNeRF(D=4, W=128).to(device)
    
    print("Training...")
    model = train(model, images, poses, hwf, args.N_iters, args.lr)
    
    print("Done!")

