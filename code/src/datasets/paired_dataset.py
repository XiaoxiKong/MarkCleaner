import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from PIL import Image
import os
import random
import math
import torch.nn.functional as F
# Try to import a random degradation change (if a deblur or denoise function is required)
try:
    from .random_degradations import RandomBlur, RandomNoise
except ImportError:
    RandomBlur, RandomNoise = None, None

from . import mask_gen


def convert_to_rgb(img):
    """
    Convert images to RGB formatting (boxle function, instead of lambda)
    """
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img


class FreqMaskGenerator:
    def __init__(self,
                 input_size=224,
                 mask_radius1=16,
                 mask_radius2=999,
                 sample_ratio=0.5):
        """
        Frequency mask generator

        Args:
            input_size: Enter image size (support 224, 256, 512, etc.)
            mask_radius1: inner circle radius (inclusive)
            mask_radius2: outer circle radius (not included)
            sample_ratio: Select the probability of low/ high
        """
        self.input_size = input_size
        self.mask_radius1 = mask_radius1
        self.mask_radius2 = mask_radius2
        self.sample_ratio = sample_ratio

        # Anticipated mask - Supports any input size
        self.mask = np.ones((self.input_size, self.input_size), dtype=int)
        for y in range(self.input_size):
            for x in range(self.input_size):
                # Calculate distance square to centre
                dist_sq = (x - self.input_size // 2) ** 2 + (y - self.input_size // 2) ** 2
                if self.mask_radius1 ** 2 <= dist_sq < self.mask_radius2 ** 2:
                    self.mask[y, x] = 0

    def __call__(self):
        """
        Generate frequency mask

        Returns:
            mask: Frequency mask (1 reservation, 0 mask)
        """
        rnd = torch.bernoulli(torch.tensor(self.sample_ratio, dtype=torch.float)).item()
        if rnd == 0:  # High access mask (maintain high frequency)
            return 1 - self.mask
        elif rnd == 1:  # Low traffic mask (maintain low frequency)
            return self.mask
        else:
            raise ValueError("Random value must be0or1")


class MFMDataTransform:
    def __init__(self,
                 img_size=224,  # In support of 224, 256, 512, any size.
                 min_crop_scale=0.2,
                 interpolation=InterpolationMode.BICUBIC,
                 filter_type='mfm',
                 # MFM parameters
                 mask_radius1=16,
                 mask_radius2=999,
                 sample_ratio=0.5,
                 # Fuzzy Arguments
                 blur_params=None,
                 # Noise Parameter
                 noise_params=None,
                 # Space mask parameters
                 mask_type=[2, 4],
                 min_ratio=0.0,
                 max_ratio=0.2,
                 ratio=None):
        """
        MFM Data Conversion Class

        Args:
            img_size: Output image size (supports any square size, e. g. 224, 256, 512).
            Min_crop_scale: Minimum zoom ratio for random crop
            Interpolation: Plug-in
            filter_type: Filter Type ('mfm', 'deblur', 'denoise')
            mask_radius1: MFM inner radius
            mask_radius2: MFM outer radius
            sample_ratio: MFM sampling ratio
            blur_params: Fuzzy Arguments(Only if...filter_type='deblur'_Other Organiser)
            noise_params: Noise Parameter(Only if...filter_type='denoise'_Other Organiser)
            mask_type: List of Space Mask Types [0, 1, 2, 4]
            Min_ratio: Minimum space mask ratio
            max_ratio: Maximum space mask ratio
            ratio: Space mask ratio (random if None)
        """
        # Basic image enhancement - support arbitrary img_size
        # Replace lambda with a pickle function to support multi-process data loading
        # Resize to at least img_size size (maintains long edge) and then randomly crop to img_sizeximg_size
        self.transform_img = T.Compose([
            T.Lambda(convert_to_rgb),  # Use a pickle function
            # T. Resize (size=int (img_size * 1.1), Internationalation=internation), #resize to a slightly larger size before ensuring sufficient areas to be trimmed
            T.RandomCrop(img_size),  # Cuts directly randomly to img_size×img_size, without scaling
            T.RandomHorizontalFlip(),
        ])

        self.filter_type = filter_type
        self.img_size = img_size
        self.mask_type = mask_type
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.ratio = ratio

        # Initialize the corresponding changes by filter type
        if filter_type == 'deblur':
            if RandomBlur is None:
                raise ImportError("RandomBlur is unavailable in random_degradations")
            self.degrade_transform = RandomBlur(params=blur_params or {})
        elif filter_type == 'denoise':
            if RandomNoise is None:
                raise ImportError("RandomNoise is unavailable in random_degradations")
            self.degrade_transform = RandomNoise(params=noise_params or {})
        elif filter_type == 'mfm':
            # Frequency mask generator - supports any img_size
            self.freq_mask_generator = FreqMaskGenerator(
                input_size=img_size,
                mask_radius1=mask_radius1,
                mask_radius2=mask_radius2,
                sample_ratio=sample_ratio
            )
        else:
            raise NotImplementedError(f"Unsupported filter type: {filter_type}")

    def set_ratio(self, ratio):
        """
        Dynamically update the space mask’s ratio

        Args:
            ratio: New ratio value
        """
        self.ratio = ratio

    def _apply_freq_mask(self, img, freq_mask):
        """
        Apply frequency mask to image (apply in frequency field)

        Args:
            img: [C, H, W] original image tensor
            freq_mask: [H, W] Frequency mask(numpyArray, 1Expression of reservations, 0Mask)

        Returns:
            hidden_img: [C, H, W] Apply the image after frequency mask
        """
        # Convert to tensor if needed
        if isinstance(freq_mask, np.ndarray):
            freq_mask = torch.tensor(freq_mask, dtype=torch.float32)

        # Add batch dimension for FFT
        img_batch = img.unsqueeze(0)  # [1, C, H, W]
        freq_mask_batch = freq_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Convert to frequency domain
        img_fft = torch.fft.fft2(img_batch, norm='ortho')
        img_fft_shifted = torch.fft.fftshift(img_fft)

        # Apply mask in frequency domain
        # freq_mask: 1 means keep, 0 means mask out
        # Expand mask to match number of channels
        mask_expanded = freq_mask_batch.expand_as(img_fft_shifted)  # [1, C, H, W]
        img_fft_masked = img_fft_shifted * mask_expanded

        # Convert back to spatial domain
        img_fft_unshifted = torch.fft.ifftshift(img_fft_masked)
        hidden_img = torch.fft.ifft2(img_fft_unshifted, norm='ortho')
        hidden_img = hidden_img.real  # Take real part

        # Remove batch dimension
        hidden_img = hidden_img.squeeze(0)  # [C, H, W]

        # Clamp to valid range
        hidden_img = torch.clamp(hidden_img, 0, 1)

        return hidden_img

    def _generate_spatial_mask(self, img):
        """
        Generate space mask (for inpainting)

        Args:
            img: [C, H, W] image tensor

        Returns:
            mask: [1, H, W] Space mask(1Expression of reservations, 0Represents mask area)
        """
        if mask_gen is None:
            raise ImportError("mask_gen is required to generate spatial masks")

        # The mask_gen function accepts tensor input in [C, H, W]
        # Randomly select mask type
        if self.mask_type and len(self.mask_type) > 0:
            mask_type_index = random.randint(0, len(self.mask_type) - 1)
            mask_type = self.mask_type[mask_type_index]

            if mask_type == 0:
                mask = mask_gen.center_mask(img,
                                           ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
            elif mask_type == 1:
                mask = mask_gen.random_regular_mask(img,
                                                   ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
            elif mask_type == 2:
                mask = mask_gen.random_irregular_mask(img,
                                                     ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
            elif mask_type == 4:
                mask = mask_gen.random_freefrom_mask(img,
                                                    ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
            else:
                mask = mask_gen.random_irregular_mask(img,
                                                     ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)
        else:
            mask = mask_gen.random_irregular_mask(img,
                                                 ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        return mask  # [1, H, W]

    def _generate_corner_mask(self, img, corner_size=64):
        """
        Generate 4-angle mask (for simultaneous application of space mask while applying frequency mask)

        Args:
            img: [C, H, W] image tensor
            Corner_size: Size of each angle mask (default 64x64)

        Returns:
            corner_mask: [C, H, W] 4It’s an angle.mask(0OrganisationmaskRegional, 1Express reserved area)
        """
        C, H, W = img.shape
        device = img.device
        dtype = img.dtype

        # Create full 1 mask (indicating that all areas are reserved)
        corner_mask = torch.ones(C, H, W, device=device, dtype=dtype)

        # Create mask area at 4 angles (set at 0)
        # Top left corner
        corner_mask[:, :corner_size, :corner_size] = 0
        # Top right corner
        corner_mask[:, :corner_size, -corner_size:] = 0
        # Bottom Left
        corner_mask[:, -corner_size:, :corner_size] = 0
        # Bottom right corner
        corner_mask[:, -corner_size:, -corner_size:] = 0

        return corner_mask

    def __call__(self, img, apply_freq_mask=True, generate_spatial_mask=True):
        """
        Apply data conversion

        Args:
            img: PIL image
            Apply_freq_mask: Whether to use frequency mask to generate hidden_img (single image format only)
            Generate_spatial_mask: Whether to create a space mask

        Returns:
            img: original image tensor (CxHxW, 0-1)
            hidden_img: Apply the image after frequency mask (CxHxW, 0-1), If...apply_freq_mask=True
            img_lq: Low-quality image tensor (CxHxW, 0-1), populated only in deblur/denoise mode
            freq_mask: Frequency mask only in mfm mode
            spatial_mask: Space mask, If...generate_spatial_mask=True
        """
        img = self.transform_img(img)  # PIL Image (HxWxC, 0-255)

        img_lq = None
        if self.filter_type in ['deblur', 'denoise']:
            # Application of degradation transformation
            img_lq = np.array(img).astype(np.float32) / 255.
            img_lq = self.degrade_transform(img_lq)
            img_lq = torch.tensor(img_lq.transpose(2, 0, 1), dtype=torch.float32)

        # Convert to a CxHxW tensor in [0, 1] without torchvision ToTensor.
        img_array = np.array(img, dtype=np.uint8)
        img = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1) / 255.0

        freq_mask = None
        hidden_img = None
        spatial_mask = None

        if self.filter_type == 'mfm':
            # Ensure that freq_mask_generator exists
            if not hasattr(self, 'freq_mask_generator'):
                raise AttributeError(f"freq_mask_generator not initialized. filter_type={self.filter_type}")
            freq_mask = self.freq_mask_generator()

            # Apply frequency mask to generate hidden_img
            # if apply_freq_mask:
            #     hidden_img = self._apply_freq_mask(img, freq_mask)

            # # 50% probability of applying space at 4 angles at the same time mask
            #     if random.random() < 0.5:
            #         corner_mask = self._generate_corner_mask(hidden_img, corner_size=64)
            #         hidden_img = hidden_img * corner_mask

            # Generate space mask
            if generate_spatial_mask:
                # Use hidden_img (if generated) or original img to generate space mask
                img_for_mask = hidden_img if hidden_img is not None else img
                spatial_mask = self._generate_spatial_mask(img_for_mask)

        return img, hidden_img, img_lq, freq_mask, spatial_mask


class MFMSingleImageDataset(Dataset):
    """
    Single image data set in MFM format for ImageFolder mode
    Apply frequency mask to generate hidden_img in pre-treatment and create space mask
    """
    def __init__(self, data_path, transform, is_train=True):
        """
        Args:
            Data_path: data path (dir in ImageFolder format)
            Examples of MFMDataTransform
            is_training:
        """
        from torchvision.datasets import ImageFolder
        self.imagefolder = ImageFolder(data_path)
        self.transform = transform
        self.is_train = is_train

    def set_ratio(self, ratio):
        """
        Dynamicly update the transform ’rateio

        Args:
            ratio: New ratio value
        """
        if hasattr(self.transform, 'set_ratio'):
            self.transform.set_ratio(ratio)

    def __len__(self):
        return len(self.imagefolder)

    def __getitem__(self, index):
        img, label = self.imagefolder[index]

        # Apply Transform (data enhancement, etc.)
        # Transform returns: (img, hidden_img, img_lq, freq_mask, spatial_mask)
        # For single image formats, use frequency mask to generate hidden_img and create space mask
        img_tensor, hidden_img, _, freq_mask, spatial_mask = self.transform(
            img, apply_freq_mask=True, generate_spatial_mask=True
        )

        return {
            'img': img_tensor,
            'hidden_img': hidden_img,
            'freq_mask': freq_mask,  # Frequency mask (numpy array)
            'spatial_mask': spatial_mask  # Space mask (tensor)
        }


class MFMPairedDataset(Dataset):
    """
    Matching data sets in MFM format, supported by reading cover_img and hidden_img, respectively
    Similar to inpaint_dataset, but using MFM transform
    Support the generation of cover_img from hidden_img (through migration and resize)
    """
    def __init__(self, data_path, transform, is_train=True,
                 translate_m_range=(5, 20), translate_n_range=(5, 20),
                 target_size=256, generate_cover_from_hidden=False):
        """
        Args:
            data_path: data path (including directory for cover_train.txt and Hidden_train.txt or only hidden_train.txt)
            Examples of MFMDataTransform
            is_training:
            translate_m_range: Range of pixels to move to the left equal (min, max), For use fromhidden_imgGeneratecover_img
            translate_n_range: The range of pixels to move up equals (min, max), For use fromhidden_imgGeneratecover_img
            Target size after target_size: resize (default 256)
            generate_cover_from_hidden: Whether to generate cover_img from hidden_img (Cover_training.txt is not required for True)
        """
        self.transform = transform
        self.is_train = is_train
        self.translate_m_range = translate_m_range
        self.translate_n_range = translate_n_range
        self.target_size = target_size
        self.generate_cover_from_hidden = generate_cover_from_hidden

        if os.path.isdir(data_path):
            if is_train:
                hidden_txt_path = os.path.join(data_path, 'hidden_train.txt')
                if not generate_cover_from_hidden:
                    cover_txt_path = os.path.join(data_path, 'cover_train.txt')
            else:
                hidden_txt_path = os.path.join(data_path, 'hidden_test.txt')
                if not generate_cover_from_hidden:
                    cover_txt_path = os.path.join(data_path, 'cover_test.txt')
        else:
            raise ValueError(f"Data path does not exist: {data_path}")

        with open(hidden_txt_path, 'r') as f:
            self.hidden_imgs = [_.rstrip('\n') for _ in f.readlines()]

        if not generate_cover_from_hidden:
            with open(cover_txt_path, 'r') as f:
                self.cover_imgs = [_.rstrip('\n') for _ in f.readlines()]
            assert len(self.cover_imgs) == len(self.hidden_imgs), \
                f"coverandhiddenNumber of images does not match: {len(self.cover_imgs)} vs {len(self.hidden_imgs)}"
        else:
            # Could not close temporary folder: %s
            self.cover_imgs = None

    def set_ratio(self, ratio):
        """
        Dynamicly update the transform ’rateio

        Args:
            ratio: New ratio value
        """
        if hasattr(self.transform, 'set_ratio'):
            self.transform.set_ratio(ratio)

    def translate_and_resize_tensor(self, img_tensor, m, n, target_size=256):
        """
        Conduct efficient migration and resize operations on tensor

        Args:
            img_tensor: torch.Tensor [C, H, W] Format, Value Range [0, 1]
            m: Number of pixels moving to the left
            n: Number of pixels moving up equal
            Target size after target_size: resize (default 256)

        Returns:
            transformed_tensor: Move andresizeIn the back. tensor [C, target_size, target_size]
        """
        import torch.nn.functional as F

        C, H, W = img_tensor.shape

        # To ensure that the volume of transfers is effective
        m = max(0, min(m, W - 1))
        n = max(0, min(n, H - 1))

        # Returns the original drawing (if the size matches) or resize if there is no lateral shift
        if m == 0 and n == 0:
            if H == target_size and W == target_size:
                return img_tensor
            else:
                # Only resizing is required.
                transformed = img_tensor.unsqueeze(0)  # [1, C, H, W]
                transformed = F.interpolate(
                    transformed,
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )
                return transformed.squeeze(0)  # [C, target_size, target_size]

        # Move left sidem, move upwards n: directly crop active areas without adding black edges
        # Crop the original map’s [n:H, m:W] section, tansor of (H-n) x (W-m)
        crop_h = H - n
        crop_w = W - m

        # Make sure the crop area is valid
        if crop_h <= 0 or crop_w <= 0:
            # Return to the original map (resize to target size) if the volume is too large
            transformed = img_tensor.unsqueeze(0)
            transformed = F.interpolate(
                transformed,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
            return transformed.squeeze(0)

        # Direct cropping: From (n, m), size (crop_h, crop_w)
        transformed = img_tensor[:, n:n+crop_h, m:m+crop_w]  # [C, crop_h, crop_w]

        # Resize Area Back to Target_size x Target_size
        transformed = transformed.unsqueeze(0)  # [1, C, crop_h, crop_w]
        transformed = F.interpolate(
            transformed,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
        transformed = transformed.squeeze(0)  # [C, target_size, target_size]

        return transformed

    def rotant_translate_and_resize_tensor(self, img_batch, m_batch, n_batch, target_size=256, angle_range=(0,5)):
        """
        Batch Rotation, Shifting, Scale Change

        Process: Rotate & Fill Black Edge & Crop Equal & Scale to Target_size

        Args:
            img_batch: torch.Tensor [B, C, H, W] Image Batch, Value Range [0, 1]
            m_batch: torch.Tensor [B] Horizontal Volume(Crop Left m Columns, Move Left)
            n_batch: torch.Tensor [B] Vertical Shift(Crop Up n Okay., Move Up)
            Target_size: int output image size (default 256)
            angle_range: tuple Rotate angle range (min_deg, max_deg), Default (0, 5)

        Returns:
            result: torch.Tensor [B, C, target_size, target_size] Convert Images

        Example:
            256×256 Image, m=10, n=10 → A valid area after cropping 246×246 → Scale Back 256×256
        """

        img_batch = img_batch.unsqueeze(0)
        B, C, H, W = img_batch.shape
        device = img_batch.device
        dtype = img_batch.dtype
        m_batch = torch.full((B,),m_batch,dtype=dtype,device=device)
        n_batch = torch.full((B,),n_batch,dtype=dtype,device=device)
        # == sync, corrected by elderman == @elder_man
        angle_min, angle_max = angle_range
        angles = torch.rand(B, device=device, dtype=dtype) * (angle_max - angle_min) + angle_min
        angles_rad = angles * (math.pi / 180.0)
        cos_a = torch.cos(angles_rad)
        sin_a = torch.sin(angles_rad)

        # Rotation transition matrix (rotation around image centre)
        theta_rot = torch.zeros(B, 2, 3, dtype=dtype, device=device)
        theta_rot[:, 0, 0] = cos_a
        theta_rot[:, 0, 1] = -sin_a
        theta_rot[:, 1, 0] = sin_a
        theta_rot[:, 1, 1] = cos_a
        # [:, : 2,] = 0 indicates rotation around centre, no extra shifts

        grid_rot = F.affine_grid(theta_rot, [B, C, H, W], align_corners=False)

        # Rotated image (with black edges)
        rotated_img = F.grid_sample(
            img_batch,
            grid_rot,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, C, H, W]

        # == sync, corrected by elderman == @elder_man
        ones = torch.ones(B, 1, H, W, dtype=dtype, device=device)
        valid_mask = F.grid_sample(
            ones,
            grid_rot,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, 1, H, W]

        # Mixed: Rotation charts for rotated areas and original charts for black-side areas
        filled_img = rotated_img * valid_mask + img_batch * (1.0 - valid_mask)

        # == sync, corrected by elderman == @elder_man
        m = m_batch.float().clamp(0, W - 1)
        n = n_batch.float().clamp(0, H - 1)

        # The size of a valid area after cropping
        crop_w = W - m  # Example: 256 - 10 = 246
        crop_h = H - n  # Example: 256 - 10 = 246

        # Calculate the zoom and centre of the crop area
        scale_x = crop_w / W
        scale_y = crop_h / H
        center_x = (m + crop_w / 2) / W * 2 - 1
        center_y = (n + crop_h / 2) / H * 2 - 1

        # Crop + Scale Change Matrix
        theta_crop = torch.zeros(B, 2, 3, dtype=dtype, device=device)
        theta_crop[:, 0, 0] = scale_x
        theta_crop[:, 0, 2] = center_x
        theta_crop[:, 1, 1] = scale_y
        theta_crop[:, 1, 2] = center_y

        grid_crop = F.affine_grid(
            theta_crop,
            [B, C, target_size, target_size],
            align_corners=False
        )

        # Crop & Scale From Filled Images
        result = F.grid_sample(
            filled_img,
            grid_crop,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, C, target_size, target_size]

        return result.squeeze(0)

    def __len__(self):
        return len(self.hidden_imgs)

    def __getitem__(self, index):
        hidden_img_path = self.hidden_imgs[index]

        # =====================================================================
        # Step 1: Load hidden_img and apply transform to get hidden_img_tensor
        # =====================================================================
        # Load hidden_img
        hidden_img = Image.open(hidden_img_path).convert('RGB')

        # Application of transform to hidden_img (resize, etc.)
        hidden_img_transformed = self.transform.transform_img(hidden_img)  # PIL Image (HxWxC, 0-255)
        # Convert to a tensor without torchvision ToTensor.
        hidden_array = np.array(hidden_img_transformed, dtype=np.uint8)
        hidden_img_tensor = torch.tensor(hidden_array, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # =====================================================================
        # Step 2: Generate cover_img
        # =====================================================================
        if self.generate_cover_from_hidden:
            # Generate cover_img from hidden_img: application migration and resize
            # Note: Ensure that hidden_img and mask remain unchanged and only cover_img changes

            # Step 2.1: Randomly generate shift parameters m and n
            m = random.randint(self.translate_m_range[0], self.translate_m_range[1])
            n = random.randint(self.translate_n_range[0], self.translate_n_range[1])

            # Step 2.2: Generate cover_img_tensor using translate_and_resize_tensor from hidden_img_tensor
            # The translation and resize create a new tensor and do not modify hidden_img_tensor.
            cover_img_tensor = self.rotant_translate_and_resize_tensor(hidden_img_tensor, m, n, target_size=self.target_size, angle_range=(0,5))
            cover_img_path = hidden_img_path  # cover_img is generated from hidden_img using the same path
        else:
            # Read cover_img from file.
            cover_img_path = self.cover_imgs[index]
            cover_img = Image.open(cover_img_path).convert('RGB')

            # Ensure that both images are subject to the same random changes (maintaining a consistent pre-processing)
            # Save Random Status
            rng_state = torch.get_rng_state()
            np_rng_state = np.random.get_state()
            random_state = random.getstate()

            # Application of transform to cover_img (no frequency mask, no space mask)
            cover_img_transformed = self.transform.transform_img(cover_img)  # PIL Image (HxWxC, 0-255)
            # Convert to a tensor without torchvision ToTensor.
            cover_array = np.array(cover_img_transformed, dtype=np.uint8)
            cover_img_tensor = torch.tensor(cover_array, dtype=torch.float32).permute(2, 0, 1) / 255.0

            # Resume random state and make sure that hidden_img uses the same random changes
            torch.set_rng_state(rng_state)
            np.random.set_state(np_rng_state)
            random.setstate(random_state)

            # Reapply the same transform for hidden_img (since random state has been restored)
            hidden_img_transformed = self.transform.transform_img(hidden_img)  # PIL Image (HxWxC, 0-255)
            hidden_array = np.array(hidden_img_transformed, dtype=np.uint8)
            hidden_img_tensor = torch.tensor(hidden_array, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # =====================================================================
        # Step 3: Generate frequency mask and space mask (based on hidden_img_tensor)
        # =====================================================================
        # Generate masks from hidden_img_tensor without modifying the source tensor.
        # Apply frequency mask to hidden_img (cover_img without frequency mask)
        freq_mask = None
        spatial_mask = None
        if self.transform.filter_type == 'mfm':
            # Generate frequency mask
            freq_mask = self.transform.freq_mask_generator()
            # Apply frequency mask to hidden_img (optional, currently annotated)
            # hidden_img_tensor = self.transform._apply_freq_mask(hidden_img_tensor, freq_mask)

            # Step 3.1: Generate spatial_mask based on hidden_img_tensor
            # This ensures that mask is consistent with hidden_img and will not change due to changes in cover_img
            spatial_mask = self.transform._generate_spatial_mask(hidden_img_tensor)

            # Step 3.2: Apply spatial_mask to hidden_img_tensor
            # Note: If we apply it here, we don’t need to do it again.
            # If not applied when data is loaded, apply at training according to the train_param [’mask’] parameter
            # The current default is not applied on data loading, maintaining consistency with the training script
            # Uncomment the following line to apply the spatial mask during loading:
            # hidden_img_tensor = hidden_img_tensor * spatial_mask

        return {
            'cover_img': cover_img_tensor,
            'hidden_img': hidden_img_tensor,
            'cover_img_path': cover_img_path,
            'hidden_img_path': hidden_img_path,
            'freq_mask': freq_mask,  # Frequency mask (numpy array)
            'spatial_mask': spatial_mask  # Space mask (tensor)
        }


def mfm_paired_collate_fn(batch):
    """
    Collate function for matching data sets
    """
    cover_imgs = []
    hidden_imgs = []
    cover_img_paths = []
    hidden_img_paths = []
    freq_masks = []
    spatial_masks = []

    for item in batch:
        cover_imgs.append(item['cover_img'])
        hidden_imgs.append(item['hidden_img'])
        cover_img_paths.append(item['cover_img_path'])
        hidden_img_paths.append(item['hidden_img_path'])
        if item['freq_mask'] is not None:
            freq_masks.append(item['freq_mask'])
        if item.get('spatial_mask') is not None:
            spatial_masks.append(item['spatial_mask'])

    result = {
        'cover_img': torch.stack(cover_imgs),
        'hidden_img': torch.stack(hidden_imgs),
        'cover_img_path': cover_img_paths,
        'hidden_img_path': hidden_img_paths
    }

    if freq_masks:
        result['freq_mask'] = torch.stack([torch.tensor(m, dtype=torch.float32) for m in freq_masks])

    if spatial_masks:
        result['spatial_mask'] = torch.stack(spatial_masks)

    return result


def mfm_collate_fn(batch):
    """
    Data Merge Functions

    Args:
        Match: Data batch (from MFMSingleImageDataset in dictionaries list)

    Returns:
        Merged Data Dictionary
    """
    img_list = []
    hidden_img_list = []
    img_lq_list = []
    freq_mask_list = []
    spatial_mask_list = []

    # Check if batch is list of dicts (from MFMSingleImageDataset) or tuples (old format)
    if isinstance(batch[0], dict):
        # New format: list of dicts
        for item in batch:
            img_list.append(item['img'])
            if 'hidden_img' in item and item['hidden_img'] is not None:
                hidden_img_list.append(item['hidden_img'])
            if 'freq_mask' in item and item['freq_mask'] is not None:
                freq_mask_list.append(item['freq_mask'])
            if 'spatial_mask' in item and item['spatial_mask'] is not None:
                spatial_mask_list.append(item['spatial_mask'])
    else:
        # Old format: list of tuples (img, hidden_img, img_lq, freq_mask, spatial_mask)
        for img, hidden_img, img_lq, freq_mask, spatial_mask in batch:
            img_list.append(img)
            if hidden_img is not None:
                hidden_img_list.append(hidden_img)
            if img_lq is not None:
                img_lq_list.append(img_lq)
            if freq_mask is not None:
                freq_mask_list.append(freq_mask)
            if spatial_mask is not None:
                spatial_mask_list.append(spatial_mask)

    result = {
        'img': torch.stack(img_list)
    }

    if hidden_img_list:
        result['hidden_img'] = torch.stack(hidden_img_list)

    if img_lq_list:
        result['img_lq'] = torch.stack(img_lq_list)

    if freq_mask_list:
        result['freq_mask'] = torch.stack([torch.tensor(m, dtype=torch.float32) if isinstance(m, np.ndarray) else m for m in freq_mask_list])

    if spatial_mask_list:
        result['spatial_mask'] = torch.stack(spatial_mask_list)

    return result


def create_mfm_dataloader(data_path,
                          batch_size=128,
                          img_size=224,  # Support 224, 256, 512.
                          min_crop_scale=0.2,
                          interpolation=InterpolationMode.BICUBIC,
                          filter_type='mfm',
                          mask_radius1=16,
                          mask_radius2=999,
                          sample_ratio=0.5,
                          blur_params=None,
                          noise_params=None,
                          num_workers=8,
                          pin_memory=True,
                          drop_last=True):
    """
    Create MFM Data Loader

    Args:
        Data_path: Data path
        Match_size: Batch Size
        img_size: image size (supports any square size, e.g. 224, 256, 512).
        Min_crop_scale: Minimum crop scale
        Interpolation: Plug-in
        filter_type:
        mask_radius1: MFM inner radius
        mask_radius2: MFM outer radius
        sample_ratio: MFM sampling ratio
        blur_params: Fuzzy parameters
        Noise_params: Noise parameters
        Num_workers: Number of work lines
        Pin_morary: Whether or not to fix RAM
        Drop_last: Whether to discard the last incomplete batch

    Returns:
        Dataloader
    """
    # Initialize data conversion
    transform = MFMDataTransform(
        img_size=img_size,
        min_crop_scale=min_crop_scale,
        interpolation=interpolation,
        filter_type=filter_type,
        mask_radius1=mask_radius1,
        mask_radius2=mask_radius2,
        sample_ratio=sample_ratio,
        blur_params=blur_params,
        noise_params=noise_params
    )

    # Create a data set
    dataset = ImageFolder(data_path, transform)
    print(f"Build Datasets: Number of training images = {len(dataset)}")
    print(f"Image Size Set to: {img_size}x{img_size}")

    # Create Data Loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=mfm_collate_fn
    )

    return dataloader


# Use Example
if __name__ == "__main__":
    # Example 1: Use 256x256 image size
    dataloader_256 = create_mfm_dataloader(
        data_path="path/to/your/data",
        batch_size=64,
        img_size=256,  # Set to 256x256
        filter_type='mfm'
    )

    # Example 2: Use 512x512 image size
    dataloader_512 = create_mfm_dataloader(
        data_path="path/to/your/data",
        batch_size=32,  # Decrease batch size to fit bigger images
        img_size=512,  # Set to 512x512
        filter_type='mfm'
    )

    print("Data loader created successfully！")
