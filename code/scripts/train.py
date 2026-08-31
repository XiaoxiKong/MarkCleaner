import sys
import os
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from mmengine import Config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lpips import LPIPS
import math
import cv2
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.distributed import init_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LambdaLR
from transformers import get_constant_schedule_with_warmup
from torchvision.utils import save_image

from src.datasets.dataset import RatioDistributedSampler, inpaint_dataset_nomarked
from src.datasets import mask_gen
from src.model import MarkCleanerModel
from src.loss.discriminator import weights_init, NLayerDiscriminator as ldm_D, gradient_penalty_loss
from src.loss.frequency_loss import FrequencyLoss
from src.utils import util
from src.dino import create_model as create_dino_model, get_feature as get_dino_feature


def ddp_setup(rank: int, world_size: int):
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ.setdefault("MASTER_PORT", "13372")
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cosine_similarity(feature1, feature2):
    assert feature1.shape == feature2.shape

    f1 = F.normalize(feature1, dim=-1)
    f2 = F.normalize(feature2, dim=-1)

    cosine_sim = -1*(f1*f2).sum(dim=-1)
    return cosine_sim.mean()


def delay_lambda(current_step):
    return 0.0


def fft_low_freq_loss(pred, gt, max_radius=128, min_radius=60):
    """
    Constraint the FFT low frequency area with GT
    Args:
        pred: predicted image [B, C, H, W]
        gt: Real Images [B, C, H, W]
        max_radius: Maximum radius (watermarks mainly concentrated area)
        Min_radius: Minimum radius (optional, for a more precise ring mask)
    Returns:
        Lost: FFT Low Frequency Loss
    """
    # Convert to Greyscale (if RGB)
    if pred.shape[1] == 3:
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        gt_gray = 0.299 * gt[:, 0:1] + 0.587 * gt[:, 1:2] + 0.114 * gt[:, 2:3]
    else:
        pred_gray = pred
        gt_gray = gt

    # Compute FFT
    pred_fft = torch.fft.fft2(pred_gray)
    pred_fft = torch.fft.fftshift(pred_fft)

    gt_fft = torch.fft.fft2(gt_gray)
    gt_fft = torch.fft.fftshift(gt_fft)

    B, C, H, W = pred_gray.shape
    cy, cx = H // 2, W // 2

    # Create a circular mask for the frequency range where the watermarks are located
    y = torch.arange(H, device=pred.device, dtype=pred.dtype).view(-1, 1) - cy
    x = torch.arange(W, device=pred.device, dtype=pred.dtype).view(1, -1) - cx
    dist = torch.sqrt(x**2 + y**2)

    # Create mask: you can choose a circle or a circle
    if min_radius > 0:
        ring_mask = ((dist >= min_radius) & (dist <= max_radius)).float()
    else:
        ring_mask = (dist <= max_radius).float()

    # Calculated loss (costed separately for each channel)
    loss = torch.abs(pred_fft - gt_fft) * ring_mask
    return loss.mean()


def wavelet_hh_loss_conv(pred, gt):
    """
    Compute a differentiable approximation of the wavelet HH subband.
    This constrains high-frequency diagonal details.
    Args:
        pred: predicted image [B, C, H, W]
        gt: Ground-truth images [B, C, H, W]
    Returns:
        Wavelet HH reconstruction loss
    """
    # Haar HH filter for diagonal high-frequency changes.
    hh_kernel = torch.tensor([[ 1, -1],
                               [-1,  1]], dtype=pred.dtype, device=pred.device).view(1, 1, 2, 2) / 2.0

    # Each channel is treated separately.
    total_loss = 0
    for c in range(pred.shape[1]):
        hh_kernel_c = hh_kernel.repeat(1, 1, 1, 1)
        hh_pred = F.conv2d(pred[:, c:c+1], hh_kernel_c, stride=2, padding=0)
        hh_gt = F.conv2d(gt[:, c:c+1], hh_kernel_c, stride=2, padding=0)
        total_loss += F.l1_loss(hh_pred, hh_gt)

    return total_loss / pred.shape[1]


def multi_scale_freq_loss(pred, gt, scales=[1, 2, 4]):
    """
    Multiscale frequency consistency constraint
    Args:
        pred: predicted image [B, C, H, W]
        gt: Real Images [B, C, H, W]
        scales: multiscale list
    Returns:
        Loss: Loss of multiscale frequency
    """
    total_loss = 0
    for s in scales:
        if s > 1:
            pred_s = F.avg_pool2d(pred, s)
            gt_s = F.avg_pool2d(gt, s)
        else:
            pred_s, gt_s = pred, gt

        # FFT range binding on each scale
        pred_fft = torch.fft.fft2(pred_s)
        gt_fft = torch.fft.fft2(gt_s)

        pred_mag = torch.abs(pred_fft)
        gt_mag = torch.abs(gt_fft)

        total_loss += F.l1_loss(pred_mag, gt_mag)

    return total_loss / len(scales)


class Trainer:
    def __init__(self, cfg, device='cuda', dtype='fp32', rank=0, world_size=1):
        self.device = torch.device(device)
        self.dtype = util.to_torch_dtype(dtype)
        self.rank = rank
        self.world_size = world_size
        self.model_type = cfg.get('model_type', 'ae')
        self.model_condition_type = cfg['model_param']['condition_type']

        train_param = cfg['train_param']
        self.model_param = cfg['model_param']

        # init tensorboard
        self.tensorboard = False
        self.writer = None
        if train_param['wandb'] and self.rank==0:
            self.tensorboard = True
            log_dir = os.path.join(train_param.get('tensorboard_dir', 'runs'), train_param['wandb_name'])
            self.writer = SummaryWriter(log_dir=log_dir)

        self.enable_mask = train_param['mask']
        self.enable_freq_mask = train_param.get('enable_freq_mask', True)
        self.freq_mask_max_prob = train_param.get('freq_mask_max_prob', 0.6)
        self.freq_mask_warmup_steps = train_param.get('freq_mask_warmup_steps', 200000)
        if not 0.0 <= self.freq_mask_max_prob <= 1.0:
            raise ValueError('freq_mask_max_prob must be between 0 and 1')
        if self.freq_mask_warmup_steps <= 0:
            raise ValueError('freq_mask_warmup_steps must be positive')


        # mask ratio
        self.use_multi_mask_ratio = train_param['use_multi_mask_ratio']
        self.ratio_interp = train_param['ratio_interp']
        self.min_ratio = train_param['min_ratio']
        self.max_ratio = train_param['max_ratio']

        # load dataset - support both original format and MFM format
        dataset_param = cfg['data_param']
        data_path = dataset_param.get('path', dataset_param.get('data_path', ''))
        num_workers = dataset_param.get('num_workers', 8)
        batch_size = dataset_param.get('bs', 128)
        self.gradient_accumulation_steps = dataset_param.get('gradient_accumulation_steps', 1)

        # For MFM format: mask type for spatial mask generation
        # Store mask_type from dataset_param for later use (must be defined before using it)
        self.mask_type_for_mfm = dataset_param.get('mask_type', [2, 4])

        # Check if using MFM format or original format
        use_mfm_format = dataset_param.get('use_mfm_format', False)  # Default to original format

        if use_mfm_format:
            # Use MFM data loader with cover_img and hidden_img pairs
            img_size = dataset_param.get('img_size', 256)
            mask_radius1 = dataset_param.get('mask_radius1', 16)
            mask_radius2 = dataset_param.get('mask_radius2', 999)
            sample_ratio = dataset_param.get('sample_ratio', 0.5)
            filter_type = dataset_param.get('filter_type', 'mfm')

            # Check if using txt files (like original format) or ImageFolder
            use_txt_files = dataset_param.get('use_txt_files', True)  # Default to txt files for cover/hidden pairs

            if use_txt_files:
                # Use txt files to load cover_img and hidden_img pairs (like original format)
                # N.B. The data gathering ensures that:
                # - hidden_img remains unchanged (load directly from file, apply only Transform, do not shift)
                # -Mask remains unchanged (based on hidden_img generation, and therefore also unchanged)
                # - cover_img will be generated by translate_and_resize_tensor from hidden_img (will change)
                from torchvision.transforms import InterpolationMode
                from src.datasets.paired_dataset import MFMDataTransform, MFMPairedDataset, mfm_paired_collate_fn

                transform = MFMDataTransform(
                    img_size=img_size,
                    min_crop_scale=dataset_param.get('min_crop_scale', 0.2),
                    interpolation=InterpolationMode.BICUBIC,
                    filter_type=filter_type,
                    mask_radius1=mask_radius1,
                    mask_radius2=mask_radius2,
                    sample_ratio=sample_ratio,
                    mask_type=self.mask_type_for_mfm,
                    min_ratio=self.min_ratio,
                    max_ratio=self.max_ratio
                )

                # Retrieve shift parameters, use default if not specified
                translate_m_range = dataset_param.get('translate_m_range', (3, 20))  # Range of pixels to move to the left equal
                translate_n_range = dataset_param.get('translate_n_range', (3, 20))  # The range of pixels to move up equals
                generate_cover_from_hidden = dataset_param.get('generate_cover_from_hidden', True)  # Generate cover_img from hidden_img

                dataset = MFMPairedDataset(
                    data_path=data_path,
                    transform=transform,
                    is_train=dataset_param.get('is_train', True),
                    translate_m_range=translate_m_range,
                    translate_n_range=translate_n_range,
                    target_size=img_size,
                    generate_cover_from_hidden=generate_cover_from_hidden
                )
                print(f"[GPU{self.rank}] UseMFMFormat Datasets(Match Mode): Number of training images = {len(dataset)}")
                print(f"[GPU{self.rank}] Image Size Set to: {img_size}x{img_size}")
                print(f"[GPU{self.rank}] Move Parameter Range: m={translate_m_range}, n={translate_n_range}")
                print(f"[GPU{self.rank}] Fromhidden_imgGeneratecover_img: {generate_cover_from_hidden}")
                print(f"[GPU{self.rank}] Data transform: hidden_img and mask stay fixed; cover_img is translated and resized")

                if self.use_multi_mask_ratio:
                    sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                else:
                    sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

                self.dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    collate_fn=mfm_paired_collate_fn,
                    drop_last=True
                )
            else:
                # Use ImageFolder (single image mode)
                from torchvision.transforms import InterpolationMode
                from src.datasets.paired_dataset import MFMDataTransform, MFMSingleImageDataset, mfm_collate_fn

                transform = MFMDataTransform(
                    img_size=img_size,
                    min_crop_scale=dataset_param.get('min_crop_scale', 0.2),
                    interpolation=InterpolationMode.BICUBIC,
                    filter_type=filter_type,
                    mask_radius1=mask_radius1,
                    mask_radius2=mask_radius2,
                    sample_ratio=sample_ratio,
                    mask_type=self.mask_type_for_mfm,
                    min_ratio=self.min_ratio,
                    max_ratio=self.max_ratio
                )

                dataset = MFMSingleImageDataset(data_path, transform, is_train=dataset_param.get('is_train', True))
                print(f"[GPU{self.rank}] UseMFMFormat Datasets(Single image mode): Number of training images = {len(dataset)}")
                print(f"[GPU{self.rank}] Image Size Set to: {img_size}x{img_size}")

                if self.use_multi_mask_ratio:
                    sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                else:
                    sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

                self.dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    collate_fn=mfm_collate_fn,
                    drop_last=True
                )
            self.use_mfm_format = True
        else:
            # Use original dataset format with translate_and_resize_tensor
            # N.B. The data gathering ensures that:
            # - hidden_img remains unchanged (direct load from file, no conversion)
            # -Mask remains unchanged (based on hidden_img generation, and therefore also unchanged)
            # - cover_img will be generated by translate_and_resize_tensor from hidden_img (will change)
            shuffle = dataset_param.pop('shuffle', True)

            # Retrieve shift parameters, use default if not specified
            translate_m_range = dataset_param.pop('translate_m_range', (5, 20))  # Range of pixels to move to the left equal
            translate_n_range = dataset_param.pop('translate_n_range', (5, 20))  # The range of pixels to move up equals

            dataset = inpaint_dataset_nomarked(
                **dataset_param,
                min_ratio=self.min_ratio,
                max_ratio=self.max_ratio,
                translate_m_range=translate_m_range,
                translate_n_range=translate_n_range
            )

            if self.use_multi_mask_ratio:
                sampler = RatioDistributedSampler
            else:
                sampler = DistributedSampler

            self.dataloader = DataLoader(dataset, num_workers=num_workers, pin_memory=True, batch_size=batch_size, sampler=sampler(dataset))
            self.use_mfm_format = False
            print(f"[GPU{self.rank}] Use original data set format(Take it. translate_and_resize_tensor): Number of training images = {len(dataset)}")
            print(f"[GPU{self.rank}] Move Parameter Range: m={translate_m_range}, n={translate_n_range}")
            print(f"[GPU{self.rank}] Data transform: hidden_img and mask stay fixed; cover_img is translated and resized")

        self.data_iter = len(self.dataloader)


        self.use_dino_cls = self.model_param.get('use_dino_cls', False)
        self.use_dino_pred_loss = self.model_param.get('use_dino_pred_loss', False)

        if self.use_dino_cls:
            self.dino_model = create_dino_model().to(self.device, self.dtype)
            util.freeze(self.dino_model)
        else:
            self.dino_model=None

        # init model
        if self.model_type == 'ae':
            self.model = MarkCleanerModel(**cfg['model_param'], is_train=True, dino_model=self.dino_model).to(self.device, self.dtype)
        else:
            raise NotImplementedError


        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)


        # init param
        self.lr = train_param['lr']
        self.global_step = 0
        self.start_epoch = 0
        self.end_epoch = train_param['epoch']
        self.log_step = train_param['log_step']
        self.save_step = train_param['save_step']
        self.sample_step = train_param['sample_step']
        self.sample_dir = os.path.join(train_param['sample_dir'], train_param['exp_name'])
        self.output_dir = os.path.join(train_param['output_dir'], train_param['exp_name'])
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Generator trainable params: {util.format_numel(sum(p.numel() for p in self.model.parameters() if p.requires_grad))}")


        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=1e-3)
        self.scheduler = MultiStepLR(self.optimizer, milestones=train_param['milestones'], gamma=0.2, last_epoch=self.start_epoch-1)
        self.loss_type = train_param['loss_type']


        if self.loss_type.get('gan', None):
            self.discriminator = ldm_D(norm_type = 'gn').apply(weights_init).to(self.device, self.dtype)
            self.discriminator = DDP(self.discriminator, device_ids=[self.rank])
            self.gan_iter_step = self.loss_type['gan_iter_step']

            self.d_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=train_param['gan_lr'], weight_decay=1e-3)


            if train_param['d_warmup_step'] == 0 and  train_param['d_zero_step']==0:
                self.d_scheduler =  MultiStepLR(self.d_optimizer, milestones=train_param['milestones'], gamma=0.5, last_epoch=self.start_epoch-1)
                self.d_zero_step = -1
            else:
                delay_scheduler = LambdaLR(self.d_optimizer, lr_lambda=delay_lambda)
                warmup_scheduler = get_constant_schedule_with_warmup(
                    self.d_optimizer,
                    num_warmup_steps=train_param['d_warmup_step'],
                )
                self.d_scheduler = SequentialLR(
                    self.d_optimizer,
                    schedulers=[delay_scheduler, warmup_scheduler],
                    milestones=[train_param['d_zero_step']]
                )
                self.d_zero_step = train_param['d_zero_step']

            print(f"Discriminator trainable params: {util.format_numel(sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad))}")
        else:
            print('No Discriminator')


        # resume
        if train_param['resume']:
            checkpoint = train_param['model_path']
            assert checkpoint is not None
            self.load(checkpoint)


        assert (self.end_epoch-self.start_epoch) >0

        # loss related
        if self.loss_type['reconstruction']:
            if self.loss_type['reconstruction'].endswith('l1'):
                self.reconstruction_loss = nn.L1Loss()
            elif self.loss_type['reconstruction'].endswith('l2'):
                self.reconstruction_loss = nn.MSELoss()


        self.percep_model = None
        if self.loss_type['perceptual_loss'] == 'lpips':
            self.percep_model = LPIPS().to(self.device, self.dtype)
            util.freeze(self.percep_model)

        # Parameters associated with frequency domain loss
        self.use_freq_loss = self.loss_type.get('freq_loss', True)

        # Initialization frequency loss function
        if self.use_freq_loss:
            loss_gamma = self.loss_type.get('freq_loss_gamma', 1.0)
            patch_factor = self.loss_type.get('freq_patch_factor', 4)
            self.freq_criterion = FrequencyLoss(
                loss_gamma=loss_gamma,
                patch_factor=patch_factor
            ).to(self.device, self.dtype)

        # Dynamic freq_scale parameters
        self.freq_scale_start = self.loss_type.get('freq_scale_start', 0.0)  # Initial value
        self.freq_scale_end = self.loss_type.get('freq_scale', 0.5)  # Final value (using preq_scale in configuration)
        self.freq_scale_warmup_epochs = self.loss_type.get('freq_scale_warmup_epochs', None)  # Number of warmup epoch, None means the whole training process

        self.current_freq_scale = self.freq_scale_start  # Initialize Current Value

        # Precreate zerotensor to avoid creating each overlap
        self.zero_tensor = torch.zeros(1, device=self.device, dtype=self.dtype)

    def _generate_spatial_mask(self, img):
        """
        Generate space mask for MFM format (for inpainting)

        Args:
            img: [B, C, H, W] image tensor

        Returns:
            mask: [B, 1, H, W] Space mask(1Expression of reservations, 0Represents mask area)
        """
        batch_size = img.shape[0]
        masks = []

        for i in range(batch_size):
            # Convert tensor to numpy for mask generation
            img_np = img[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            img_np = (img_np * 255).astype(np.uint8)

            # Generate mask using mask_gen (same as original dataset)
            if self.use_multi_mask_ratio:
                ratio = self.min_ratio + 0.1*(self.epoch_id // self.ratio_interp)
                ratio = self.max_ratio if ratio > self.max_ratio else ratio
            else:
                ratio = None

            # Randomly select mask type
            if self.mask_type_for_mfm and len(self.mask_type_for_mfm) > 0:
                mask_type_index = random.randint(0, len(self.mask_type_for_mfm) - 1)
                mask_type = self.mask_type_for_mfm[mask_type_index]

                if mask_type == 0:
                    mask = mask_gen.center_mask(img_np, ratio=ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
                elif mask_type == 1:
                    mask = mask_gen.random_regular_mask(img_np, ratio=ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
                elif mask_type == 2:
                    mask = mask_gen.random_irregular_mask(img_np, ratio=ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
                elif mask_type == 4:
                    mask = mask_gen.random_freeform_mask(img_np, ratio=ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
                else:
                    mask = mask_gen.random_irregular_mask(img_np, ratio=ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
            else:
                mask = mask_gen.random_irregular_mask(img_np, ratio=ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

            # Convert back to tensor
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)  # [1, H, W]
            masks.append(mask_tensor)

        return torch.stack(masks)  # [B, 1, H, W]

    def _apply_freq_mask(self, img, freq_mask):
        """
        Apply frequency mask to image (apply in frequency field) - optimize version

        Args:
            img: [B, C, H, W] Original Image (on GPU)
            freq_mask: [B, 1, H, W] or [B, H, W] Frequency mask (on GPU)

        Returns:
            hidden_img: [B, C, H, W] Apply the image after frequency mask
        """
        # Ensure freq_mask has channel dimension: [B, H, W] -> [B, 1, H, W]
        if freq_mask.dim() == 3:
            freq_mask = freq_mask.unsqueeze(1)

        # Convert to frequency domain
        img_fft = torch.fft.fft2(img, norm='ortho')
        img_fft_shifted = torch.fft.fftshift(img_fft)

        # Apply mask in frequency domain (broadcasting is memory efficient)
        img_fft_masked = img_fft_shifted * freq_mask

        # Convert back to spatial domain
        img_fft_unshifted = torch.fft.ifftshift(img_fft_masked)
        hidden_img = torch.fft.ifft2(img_fft_unshifted, norm='ortho').real

        # Clamp to valid range
        return torch.clamp(hidden_img, 0, 1)


    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.start_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']

        self.model.module.load_state_dict(ckpt['model'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])

        if self.loss_type.get('gan', None):
            missing_keys, unexpected_keys = self.discriminator.module.load_state_dict(ckpt['d_model'], strict=False)
            if unexpected_keys:
                raise RuntimeError(f'Discriminator Unexpected keys:{unexpected_keys}')
            if missing_keys:
                print(f'Discriminator Missing keys:{missing_keys}')

            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.d_scheduler.load_state_dict(ckpt['d_scheduler'])


    def update_gan(self, pred, gt, dino_feature=None):
        util.unfreeze(self.discriminator)
        util.freeze(self.model)


        gt.requires_grad_(True).to(self.discriminator.device)

        for i in range(self.gan_iter_step):

            self.d_optimizer.zero_grad()

            # Real
            D_real = self.discriminator(gt)
            # fake
            D_fake = self.discriminator(pred)


            if self.loss_type['gan_gradient_penalty']:
                gradient_penalty = gradient_penalty_loss(self.discriminator, gt, pred) * (self.loss_type['r1_weight'] //2)
            else:
                gradient_penalty = torch.zeros(1).to(D_real.device)

            # loss for discriminator
            if self.loss_type.get('gan', None) == 'hinge':
                D_loss = F.relu(1-D_real).mean() + F.relu(1+D_fake).mean()
            elif self.loss_type.get('gan', None) == 'logistic':
                D_loss = (F.softplus(-D_real).mean() + F.softplus(D_fake).mean())*0.5
            else:
                raise NotImplementedError

            D_loss = D_loss + gradient_penalty

            D_loss.backward(retain_graph=((i < self.gan_iter_step - 1)))
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1)

            self.gradient_penalty = gradient_penalty
            self.D_loss = D_loss
            self.d_optimizer.step()


        util.freeze(self.discriminator)
        util.unfreeze(self.model)


    def get_gan_loss(self, pred, dino_feature=None):
        D_fake = self.discriminator(pred)

        if self.loss_type.get('gan', None) == 'hinge':
            gan_loss = -torch.mean(D_fake)
        elif self.loss_type.get('gan', None) == 'logistic':
            gan_loss = F.softplus(-D_fake).mean()
        else:
            raise NotImplementedError

        return gan_loss



    def update(self, pred, gt, mask, dino_feature=None, dino_pred_fea=None, dino_clean_feature=None):
        self.optimizer.zero_grad()

        self.dino_pred_loss = torch.tensor([0]).to(pred.device, pred.dtype)
        if self.use_dino_cls:
            assert (dino_pred_fea is not None) and (dino_clean_feature is not None)
            self.dino_pred_loss = cosine_similarity(dino_pred_fea, dino_clean_feature)
        self.dino_pred_loss = self.dino_pred_loss * self.loss_type['dino_pred_scale']


        self.gan_loss_n = torch.tensor([0]).to(pred.device, pred.dtype)
        if self.loss_type.get('gan', None) and (self.epoch_id>self.d_zero_step) and self.global_step>500:
            self.update_gan(pred.detach(), gt.clone(), dino_feature=dino_feature)
            self.gan_loss_n = self.get_gan_loss(pred, dino_feature=dino_feature)
        self.gan_loss_n = self.gan_loss_n*self.loss_type['gan_scale']


        self.recons_loss = torch.tensor([0]).to(pred.device, pred.dtype)
        if self.loss_type['reconstruction']:
            if self.loss_type['reconstruction'].startswith('full'):
                self.recons_loss = self.reconstruction_loss(pred, gt)
            elif self.loss_type['reconstruction'].startswith('mask'):
                self.recons_loss = self.reconstruction_loss(pred*(1-mask), gt*(1-mask))
        self.recons_loss = self.recons_loss * self.loss_type['recons_scale']


        self.perceptual_loss = torch.tensor([0]).to(pred.device, pred.dtype)
        if self.loss_type['perceptual_loss']:
            self.perceptual_loss = self.percep_model(pred, gt).mean()
        self.perceptual_loss = self.perceptual_loss * self.loss_type['perceptual_scale']

        # Frequency area loss
        self.freq_loss = torch.tensor([0]).to(pred.device, pred.dtype)
        if self.use_freq_loss:
            # Calculate frequency domain loss using FrequencyLoss
            freq_loss_value = self.freq_criterion(pred, gt)
            # FrequencyLoss returns the average amount of loss per patch required
            self.freq_loss = freq_loss_value.mean()

        # Dynamic calculation freq_scale: from start linear to end
        elapsed_epochs = self.epoch_id - self.start_epoch
        # Determines the warmup cycle: if the warmup_epochs are specified, use it, otherwise use the total training cycle
        warmup_period = self.freq_scale_warmup_epochs if self.freq_scale_warmup_epochs is not None else (self.end_epoch - self.start_epoch)
        # Calculate progress
        progress = min(elapsed_epochs / max(warmup_period, 1), 1.0)
        # Linear Plug-In
        self.current_freq_scale = self.freq_scale_start + (self.freq_scale_end - self.freq_scale_start) * progress
        # Apply weight to loss
        self.freq_loss = self.freq_loss * self.current_freq_scale

        self.loss = self.gan_loss_n + self.recons_loss + self.perceptual_loss + self.dino_pred_loss + self.freq_loss
        self.loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

        self.optimizer.step()



    def sample(self, preds, masks, gts, hidden_imgs, img_paths, epoch):
        bs = preds.shape[0]
        savedir = os.path.join(self.sample_dir, f'Epoch{epoch}_Step{self.global_step}')
        os.makedirs(savedir, exist_ok=True)
        preds = torch.clamp(preds, 0, 1)

        for b in range(bs):
            pred = (preds[b] * 255).to(dtype=torch.uint8)
            gt = (gts[b] * 255).to(dtype=torch.uint8)
            hidden_img = (hidden_imgs[b] * 255).to(dtype=torch.uint8)
            mask = masks[b]
            # Show the actual input: watermarked image with mask applied
            mask_input = (hidden_img * mask).to(dtype=torch.uint8)

            img_path = img_paths[b]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            savep = os.path.join(savedir, f'sample_{img_name}.jpg')

            if self.enable_mask:
                out_img = torch.cat([pred, mask_input, gt], dim=-1)  # [C, H, W*3]
            else:
                out_img = torch.cat([pred, gt], dim=-1)  # [C, H, W*2]

            # Save tansor directly using torchvision.save_image
            # Save_image Expected Format: [C,H,W], Value Range [0,1] or [0,255] (uint8)
            # Current out_img is [C,H,W] format, value range [0,255] (uint8)
            # Requires conversion to [0,1] range float32, or uses nonmalize=False
            out_img_float = out_img.to(dtype=torch.float32) / 255.0  # Convert to range [0,1]

            # Ensures continuous memory on CPU
            out_img_float = out_img_float.cpu()
            if not out_img_float.is_contiguous():
                out_img_float = out_img_float.contiguous()

            # Save with torchvision, direct with tensor, without having to convert to numbery
            save_image(out_img_float, savep, normalize=False)



    def train(self):
        log_loss = 0
        print(f'Training from epoch {self.start_epoch} to epoch {self.end_epoch}, each with {self.data_iter//self.world_size} iterations')
        for epoch_id in range(self.start_epoch, self.end_epoch+1):
            self.epoch_id = epoch_id
            print(f"[GPU{self.rank}] Epoch {epoch_id} | Steps: {len(self.dataloader)}")
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch_id}')
            self.dataloader.sampler.set_epoch(epoch_id)

            if self.use_multi_mask_ratio:
                ratio = self.min_ratio + 0.1*(epoch_id // self.ratio_interp)
                ratio = self.max_ratio if ratio>self.max_ratio else ratio

                print(f'current mask ratio:', ratio)
                if hasattr(self.dataloader.sampler, 'set_ratio'):
                    self.dataloader.sampler.set_ratio(ratio, self.min_ratio, self.max_ratio)
                # Update dataset's transform ratio for MFM format
                if self.use_mfm_format and hasattr(self.dataloader.dataset, 'set_ratio'):
                    self.dataloader.dataset.set_ratio(ratio)


            for iter_id, batch in enumerate(pbar):

                # Move all tensors to GPU with non_blocking for better pipeline
                cover_img = batch['cover_img'].to(self.device, self.dtype, non_blocking=True)  # [B3 HW] - Clean image read from file
                hidden_img = batch['hidden_img'].to(self.device, self.dtype, non_blocking=True)  # [B3 HW] - Watermark image read from file
                mask = batch['spatial_mask'].to(self.device, self.dtype, non_blocking=True)  # [B, 1, H, W] - spatial mask
                img_path = batch['cover_img_path']  # list of length B
                hidden_img_path = batch['hidden_img_path']  # list of length B

                # Gradually increase frequency-mask augmentation to its configured maximum.
                if self.enable_freq_mask:
                    if 'freq_mask' not in batch:
                        raise KeyError(
                            'Frequency-mask augmentation is enabled, but the dataset batch '
                            'does not contain freq_mask.'
                        )
                    prob = self.freq_mask_max_prob * min(
                        1.0, self.global_step / self.freq_mask_warmup_steps
                    )
                    if random.random() < prob:
                        freq_mask = batch['freq_mask'].to(
                            self.device, self.dtype, non_blocking=True
                        )
                        hidden_img = self._apply_freq_mask(hidden_img, freq_mask)


                dino_feature = None
                dino_clean_feature = None
                if self.enable_mask:
                    # Apply spatial mask to hidden_img for input
                    input = hidden_img * mask
                    input_sets = (input, mask)

                    if self.use_dino_cls:
                        dino_feature = get_dino_feature(input, self.dino_model)
                        if self.use_dino_pred_loss:
                            dino_clean_feature = get_dino_feature(hidden_img, self.dino_model)

                else:
                    input_sets = (hidden_img,)


                dino_pred_fea=None
                pred, dino_pred_fea = self.model(*input_sets, dino_cls_token=dino_feature)

                if isinstance(pred, torch.Tensor):
                    pred = torch.clamp(pred, 0, 1)
                elif isinstance(pred, list):
                    pred = [torch.clamp(_,0,1) for _ in pred]


                self.update(pred, cover_img, mask, dino_feature=dino_feature, dino_pred_fea=dino_pred_fea, dino_clean_feature=dino_clean_feature)
                log_loss += self.loss.item()


                pbar.set_postfix({
                        "loss":self.loss.item(),
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'd_lr':self.d_optimizer.param_groups[0]['lr'] if hasattr(self, 'd_optimizer') else 0,
                        "recons_loss":self.recons_loss.item(),
                        "gan_loss":self.gan_loss_n.item() if self.loss_type.get('gan', None) else 0,
                        "gradient_penalty":self.gradient_penalty.item() if hasattr(self, 'gradient_penalty') else 0,
                        "dino_pred_loss": self.dino_pred_loss.item() if self.use_dino_pred_loss else 0,
                        "perceptual_loss":self.perceptual_loss.item(),
                        "freq_loss":self.freq_loss.item() if self.use_freq_loss else 0,
                        "freq_scale":self.current_freq_scale if self.use_freq_loss else 0,
                        "D_loss":self.D_loss.item() if hasattr(self, 'D_loss') else 0,
                        })

                if self.rank == 0:
                    if self.global_step%self.log_step == 0:
                        avg_loss = log_loss / self.log_step
                        log_loss = 0
                        if self.tensorboard and self.writer is not None:
                            self.writer.add_scalar('Loss/total_loss', self.loss.item(), self.global_step)
                            self.writer.add_scalar('Loss/dino_pred_loss', self.dino_pred_loss.item(), self.global_step)
                            self.writer.add_scalar('Loss/gan_loss', self.gan_loss_n.item(), self.global_step)
                            self.writer.add_scalar('Loss/recons_loss', self.recons_loss.item(), self.global_step)
                            self.writer.add_scalar('Loss/D_loss', self.D_loss.item() if hasattr(self, 'D_loss') else 0, self.global_step)
                            self.writer.add_scalar('Loss/perceptual_loss', self.perceptual_loss.item(), self.global_step)
                            self.writer.add_scalar('Loss/freq_loss', self.freq_loss.item() if self.use_freq_loss else 0, self.global_step)
                            self.writer.add_scalar('Loss/freq_scale', self.current_freq_scale if self.use_freq_loss else 0, self.global_step)
                            self.writer.add_scalar('Loss/gradient_penalty', self.gradient_penalty.item() if hasattr(self, 'gradient_penalty') else 0, self.global_step)
                            self.writer.add_scalar('Loss/avg_loss', avg_loss, self.global_step)
                            self.writer.add_scalar('LearningRate/generator_lr', self.optimizer.param_groups[0]["lr"], self.global_step)
                            self.writer.add_scalar('LearningRate/discriminator_lr', self.d_optimizer.param_groups[0]['lr'] if hasattr(self, 'd_optimizer') else 0, self.global_step)

                    if self.global_step % self.sample_step == 0:
                        self.sample(pred.detach().cpu(), mask.detach().cpu(), cover_img.detach().cpu(), hidden_img.detach().cpu(), img_path, epoch_id)

                    if self.global_step%self.save_step == 0:
                        ckpt_p = os.path.join(self.output_dir, f'epoch{epoch_id}_step{self.global_step}.pth')
                        ckpt_dict = dict(
                            epoch = epoch_id,
                            global_step = self.global_step,
                            model=self.model.module.state_dict(),
                            optimizer = self.optimizer.state_dict(),
                            scheduler = self.scheduler.state_dict(),
                            d_model = self.discriminator.module.state_dict() if self.loss_type.get('gan') else None,
                            d_optimizer = self.d_optimizer.state_dict() if self.loss_type.get('gan') else None,
                            d_scheduler = self.d_scheduler.state_dict() if self.loss_type.get('gan') else None,
                        )
                        torch.save(ckpt_dict, ckpt_p)

                self.global_step += 1

            self.scheduler.step()
            if hasattr(self, 'd_scheduler'):
                self.d_scheduler.step()

            pbar.close()

        # Close tensorboard writer
        if self.tensorboard and self.writer is not None:
            self.writer.close()



def train(cfg_path, **kwargs):
    cfg = Config.fromfile(cfg_path)
    pipe = Trainer(cfg, **kwargs)
    if cfg['model_type'] == 'ae':
        pipe.train()
    else:
        raise NotImplementedError


def mp_train(rank, world_size, cfg_path):
    ddp_setup(rank, world_size)
    train(cfg_path, rank=rank, world_size=world_size)


def mp_train_wrapper(cfg_path):
    world_size = torch.cuda.device_count()
    print(f'World size:{world_size}')
    if world_size >= 1:
        print(f'Train using {world_size} gpus')
        mp.spawn(mp_train, args=(world_size, cfg_path), nprocs=world_size)
    else:
        assert None, f'wrong world size:{world_size}'


import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='train_args')
    parser.add_argument('--config_idx', type=str, default='', help='config file index')
    parser.add_argument('--config_path', type=str, default='', help='config file path')
    return parser.parse_args()


if __name__ =='__main__':
    parser = get_parser()

    if parser.config_idx !='':
        cfg_path = f'configs/train_exp{parser.config_idx}_cfg.py'
    elif parser.config_path !='':
        cfg_path = parser.config_path
    print(f'Using config file {cfg_path}')

    mp_train_wrapper(cfg_path)
