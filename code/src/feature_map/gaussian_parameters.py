import torch
import torch.nn as nn
from itertools import product
import numpy as np
import math

from src.utils.block import MLP
from .base import map_base
from .direct_map import get_coord, is_perfect_square # A tool function to repeat DirectMap

class WatermarkMap(nn.Module, map_base):
    def __init__(self, hidden_dim, gaussian_per_patch, out_patch_size, out_image_size, image_size, window_size=2, gs_fix=False):
        """
        Description of parameters:
        window_size: Defaults to 2, allowing Gaussians to move across a wider grid neighborhood than DirectMap.
        """
        super().__init__()

        # Initialize storage variables
        self.color = None
        self.cholesky = None
        self.offset = None
        self.opacity = None # Add: Storage transparency

        self.window_size = window_size
        self.gaussian_per_patch = gaussian_per_patch
        self.hidden_dim = hidden_dim
        self.gs_fix = gs_fix
        self.out_patch_size = out_patch_size
        self.out_image_size = out_image_size
        self.image_size = image_size

        assert is_perfect_square(gaussian_per_patch), 'must be square number'
        self.sqrt_gaussian = int(math.sqrt(gaussian_per_patch))

        # =====================================================================
        # Codebook (repeating DirectMap logic to ensure mathematical compatibility)
        # =====================================================================
        # Keep the codebook compatible with the model device.
        cho1 = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36])
        cho2 = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54])
        cho3 = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23])

        self.gau_dict = torch.tensor(list(product(cho1, cho2, cho3)))
        self.gau_dict = torch.cat((self.gau_dict, torch.zeros(1,3)), dim=0)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.cholesky_linear = nn.Linear(hidden_dim, hidden_dim)

        # Note: This output dimension matches the length of the gau_dict
        self.mlp_gaudict = MLP(
            in_dim=3,
            out_dim=hidden_dim,
            hidden_list=[hidden_dim//2, hidden_dim, hidden_dim, hidden_dim]
        )

        # =====================================================================
        # 2. Color MLP (same as usual)
        # =====================================================================
        self.color_mlp = MLP(
            in_dim=hidden_dim,
            out_dim=3,
            hidden_list= [hidden_dim,hidden_dim,hidden_dim,hidden_dim//2]
        )

        # =====================================================================
        # 3. Location MLP (maintain structure, but window_size becomes larger)
        # =====================================================================
        self.mlp_xy = MLP(
            in_dim=hidden_dim,
            out_dim=2,
            hidden_list=[hidden_dim,hidden_dim,hidden_dim//2]
        )

        # =====================================================================
        # 4. [NEW] Transparency MLP (Opacity)
        # =====================================================================
        # Predict a sparse opacity value for each watermark Gaussian.
        self.opacity_mlp = MLP(
            in_dim=hidden_dim,
            out_dim=1, # Output single channel alpha
            hidden_list=[hidden_dim, hidden_dim, hidden_dim//2]
        )

        # =====================================================================
        # 5. [NEW] Scaled factor (High-Frequency Constraining)
        # =====================================================================
        # Optional learnable factor for reducing the scale of watermark Gaussians.
        # A small initial value encourages compact, high-frequency Gaussians.
        # self.scale_suppression = nn.Parameter(torch.tensor(0.5))

    def map(self, feat, bs):
        # 1. Get Color
        color_feat = feat.reshape(bs, -1, self.gaussian_per_patch, self.hidden_dim)
        self.color = self.color_mlp(color_feat) # Output not activated, follow sigmoid in rasterizer or get_iter?
        # The color_mlp output of the original code DirectMap is not activated, but stored in the color variable.
        # Usually rasterizers do sigmoids, or do them here. I see the original code get_iter returns directly to the color_color_.
        # Assuming the original rasterizer internal processing colour range.

        # 2. Predict opacity.
        opacity_feat = feat.reshape(bs, -1, self.gaussian_per_patch, self.hidden_dim)
        raw_opacity = self.opacity_mlp(opacity_feat)
        self.opacity = torch.sigmoid(raw_opacity) # Limit to [0, 1]

        # 3. Get Cholesky (Covariance)
        cholesky_fea = feat.reshape(bs, -1, self.gaussian_per_patch, self.hidden_dim)
        cholesky_fea = self.leaky_relu(cholesky_fea)
        cholesky_fea = self.cholesky_linear(cholesky_fea)

        # Codebook Attention
        vector = self.mlp_gaudict(self.gau_dict.to(cholesky_fea.device))
        vector = vector.permute(1,0)
        cholesky_weight = cholesky_fea @ vector
        cholesky_weight = torch.softmax(cholesky_weight, dim=-1)
        self.cholesky = cholesky_weight @ self.gau_dict.to(cholesky_weight.device) # B 256 20 3
        # # Get basic shape parameters
        # base_cholesky = cholesky_weight @ self.gau_dict.to(cholesky_weight.device) # [B, N, M, 3]

        # # [KEY] Force down shape
        # # Multiply by a factor below 1 to encourage compact high-frequency Gaussians.
        # # abs ensures that the scale factor remains positive.
        # self.cholesky = base_cholesky * torch.clamp(self.scale_suppression.abs(), max=0.8)

        # 4. Get XY (Offset)
        xy_feat = feat.reshape(bs, -1, self.gaussian_per_patch, self.hidden_dim)
        offset = self.mlp_xy(xy_feat)
        self.offset = torch.tanh(offset) # Range: [-1, 1]

    def get_iter(self, i):
        # Extract data for batch item i.
        offset_ = self.offset[i,:,:,:].squeeze(0) # [Num_Patches, Gaussians_Per_Patch, 2]
        color_ = self.color[i, :, :].squeeze(0)    # [Num_Patches, Gaussians_Per_Patch, 3]
        para_ = self.cholesky[i, :, :].squeeze(0)  # [Num_Patches, Gaussians_Per_Patch, 3]

        # [NEW] Transparency in extraction
        opacity_ = self.opacity[i, :, :].squeeze(0) # [Num_Patches, Gaussians_Per_Patch, 1]

        # Generate grid-center coordinates on the input device.
        device = offset_.device
        get_xyz = torch.tensor(get_coord(self.sqrt_gaussian, self.sqrt_gaussian), device=device).reshape(self.sqrt_gaussian, self.sqrt_gaussian, 2)
        get_xyz = get_xyz.reshape(-1,2) # [Gaussians_Per_Patch, 2]

        if self.gs_fix:
            patch_n = offset_.shape[0]
            get_xyz = get_xyz.unsqueeze(0).repeat(patch_n,1,1)
        else:
            # Calculate Final Coordinates
            # Note: Self.window_size is used here. Watermarks can travel across the grid if they are bigger init (e. g. two or three).
            xyz1 = get_xyz[:,0:1] + 2*self.window_size*offset_[:,:,0:1]/self.out_patch_size[1] - 1/self.out_patch_size[1]
            xyz2 = get_xyz[:,1:2] + 2*self.window_size*offset_[:,:,1:2]/self.out_patch_size[0] - 1/self.out_patch_size[0]
            get_xyz = torch.cat((xyz1, xyz2), dim = -1)

        # Adjust Cholesky Scale
        weighted_cholesky = para_*(self.out_image_size[0]/self.image_size[0])

        # Returns modified opacity, not torch.ones
        return get_xyz, weighted_cholesky, color_, opacity_
