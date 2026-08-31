import torch
import torch.nn as nn


class FrequencyLoss(nn.Module):
    """Frequency loss.

    Modified from: 
    `<https://github.com/EndlessSora/focal-frequency-loss/blob/master/focal_frequency_loss/focal_frequency_loss.py>`_.

    Args:
        loss_gamma (float): the exponent to control the sharpness of the frequency distance. Defaults to 1.
        matrix_gamma (float): the scaling factor of the spectrum weight matrix for flexibility. Defaults to 1.
        patch_factor (int): the factor to crop image patches for patch-based frequency loss. Defaults to 1.
        ave_spectrum (bool): whether to use minibatch average spectrum. Defaults to False.
        with_matrix (bool): whether to use the spectrum weight matrix. Defaults to False.
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Defaults to False.
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Defaults to False.
    """

    def __init__(self,
                 loss_gamma=1.,
                 matrix_gamma=1.,
                 patch_factor=16,
                 ave_spectrum=False,
                 with_matrix=False,
                 log_matrix=False,
                 batch_matrix=False,
                 max_freq_radius=None):
        super(FrequencyLoss, self).__init__()
        self.loss_gamma = loss_gamma
        self.matrix_gamma = matrix_gamma
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.with_matrix = with_matrix
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.max_freq_radius = max_freq_radius  # None means all frequencies, otherwise 0.0-1.0

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1).float()  # NxPxCxHxW

        # perform 2D FFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm='ortho')
        # shift low frequency to the center
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        # stack the real and imaginary parts along the last dimension
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq
    def create_freq_mask(self, freq_shape, device, dtype):
        """
        Create a frequency mask that only keeps low/mid frequencies.
        
        Args:
            freq_shape: Shape of frequency tensor (..., H, W, 2)
            device: Device to create mask on
            dtype: Data type of mask
            
        Returns:
            mask: Frequency mask with shape (..., H, W) where 1.0 = keep, 0.0 = mask out
        """
        if self.max_freq_radius is None:
            # No frequency constraint, return all ones
            # freq_shape is (..., H, W, 2), we need (..., H, W)
            spatial_shape = freq_shape[:-1]
            return torch.ones(spatial_shape, device=device, dtype=dtype)
        
        # Get spatial dimensions (H, W)
        H, W = freq_shape[-3], freq_shape[-2]
        
        # Create coordinate grids
        center_h, center_w = H // 2, W // 2
        y = torch.arange(H, device=device, dtype=dtype).view(-1, 1) - center_h
        x = torch.arange(W, device=device, dtype=dtype).view(1, -1) - center_w
        
        # Calculate distance from center (normalized by max possible distance)
        max_dist = torch.sqrt(torch.tensor(H/2.0, device=device, dtype=dtype)**2 + 
                             torch.tensor(W/2.0, device=device, dtype=dtype)**2)
        dist = torch.sqrt(x**2 + y**2) / max_dist  # Normalized to [0, 1]
        
        # Create mask: 1.0 for frequencies within max_freq_radius, 0.0 otherwise
        mask = (dist <= self.max_freq_radius).float()
        
        # Expand to match frequency tensor shape (but without the last dimension for real/imag)
        # freq_shape is (..., H, W, 2), we need to expand mask to (..., H, W)
        # The mask already has shape (H, W), we need to expand it to match batch dimensions
        if len(freq_shape) > 3:
            # Expand for batch, patch, channel dimensions
            for _ in range(len(freq_shape) - 3):
                mask = mask.unsqueeze(0)
        
        return mask
        
    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        loss = torch.sqrt(tmp[..., 0] + tmp[..., 1] + 1e-12) ** self.loss_gamma
        if self.with_matrix:
            # spectrum weight matrix
            if matrix is not None:
                # if the matrix is predefined
                weight_matrix = matrix.detach()
            else:
                # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
                matrix_tmp = (recon_freq - real_freq) ** 2
                matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.matrix_gamma

                # whether to adjust the spectrum weight matrix by logarithm
                if self.log_matrix:
                    matrix_tmp = torch.log(matrix_tmp + 1.0)

                # whether to calculate the spectrum weight matrix using batch-based statistics
                if self.batch_matrix:
                    matrix_tmp = matrix_tmp / matrix_tmp.max()
                else:
                    matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

                matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
                matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
                weight_matrix = matrix_tmp.clone().detach()

            assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                'The values of spectrum weight matrix should be in the range [0, 1], '
                'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))
            # dynamic spectrum weighting (Hadamard product)
            loss = weight_matrix * loss
        
        # Apply frequency mask to only constrain low/mid frequencies
        if self.max_freq_radius is not None:
            freq_mask = self.create_freq_mask(recon_freq.shape, loss.device, loss.dtype)
            loss = loss * freq_mask
        return loss

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate frequency loss.

        Args:
            pred (torch.Tensor): Predicted tensor with shape (N, C, H, W).
            target (torch.Tensor): Target tensor with shape (N, C, H, W).
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Defaults to None.
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix)
