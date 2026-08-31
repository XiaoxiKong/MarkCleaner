import os
import random
import cv2
import numpy as np
from PIL import Image, ImageFile

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from . import mask_gen

class inpaint_dataset_mask_test(Dataset):
    def __init__(self, path, mask_type, mask_path=None, micro=True, is_train=True, min_ratio=None, max_ratio=None, ratio=None):
        self.mask_type = mask_type
        self.mask_path = mask_path
        self.target_size = (256,256)
        self.ratio = ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        if os.path.isdir(path):
            if is_train:
                # txt_path = os.path.join(path, 'cover_train.txt')
                cover_txt_path = os.path.join(path, 'cover_train.txt')
                hidden_txt_path = os.path.join(path, 'hidden_train.txt')
            else:
                cover_txt_path = os.path.join(path, 'cover_test.txt')
                hidden_txt_path = os.path.join(path, 'hidden_test.txt')
                mask_path = os.path.join(path, 'mask_test.txt')
        # else:
        #     txt_path = path

        with open(cover_txt_path, 'r') as f:
            self.cover_imgs = [_.rstrip('\n') for _ in f.readlines()]
        with open(hidden_txt_path, 'r') as f:
            self.hidden_imgs = [_.rstrip('\n') for _ in f.readlines()]
        with open(mask_path, 'r') as f:
            self.mask = [_.rstrip('\n') for _ in f.readlines()]


    def __len__(self):
        return len(self.cover_imgs)

    def load_img(self, index):
        # img_path = self.imgs[index]

        # img = Image.open(img_path).convert('RGB')
        # img = img.resize((256, 256), Image.Resampling.LANCZOS)
        # img = transforms.ToTensor()(img)

        cover_img_path = self.cover_imgs[index]
        hidden_img_path = self.hidden_imgs[index]
        mask_path = self.mask[index]

        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((256,256), Image.Resampling.NEAREST)
        mask = transforms.ToTensor()(mask)


        cover_img = Image.open(cover_img_path).convert('RGB')
        hidden_img = Image.open(hidden_img_path).convert('RGB')
        cover_img = cover_img.resize((256, 256), Image.Resampling.LANCZOS)
        # Direct tensor processing, avoid numbery:PIL image->tensor
        cover_img = torch.tensor(list(cover_img.getdata()), dtype=torch.uint8)
        cover_img = cover_img.view(256, 256, 3).permute(2, 0, 1).float() / 255.0

        hidden_img = hidden_img.resize((256, 256), Image.Resampling.LANCZOS)
        hidden_img = torch.tensor(list(hidden_img.getdata()), dtype=torch.uint8)
        hidden_img = hidden_img.view(256, 256, 3).permute(2, 0, 1).float() / 255.0

        return cover_img, hidden_img, cover_img_path, hidden_img_path, mask

        # return img, img_path

    def load_mask(self, img):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mask_type) - 1)
        mask_type = self.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return mask_gen.center_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random regular mask
        if mask_type == 1:
            return mask_gen.random_regular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random irregular mask
        if mask_type == 2:
            return mask_gen.random_irregular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # from gated convolution (iccv 2019)
        if mask_type == 4:
            return mask_gen.random_freefrom_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

    def Load_mask_test(self, img, img_path=None):
        """Load different mask types for training and testing"""
        # Load mask from file if mask_path is provided
        if self.mask_path is not None:
            if os.path.isdir(self.mask_path):
                # assume mask has same filename as image
                fname = os.path.basename(img_path)
                mask_file = os.path.join(self.mask_path, fname)
                if os.path.exists(mask_file):
                    mask = Image.open(mask_file).convert('L')
                    mask = mask.resize((256,256), Image.Resampling.NEAREST)
                    mask = transforms.ToTensor()(mask)
                    # mask in this project: 1 is valid, 0 is masked
                    mask = (mask > 0.5).float()
                    return mask
            elif os.path.isfile(self.mask_path):
                # use single mask file for all images
                mask = Image.open(self.mask_path).convert('L')
                mask = mask.resize((256, 256), Image.Resampling.NEAREST)
                mask = transforms.ToTensor()(mask)
                mask = (mask > 0.5).float()
                return mask
        # no mask file found or mask_path is None: return full-visible mask [1,256,256] so collate works
        # return torch.ones(1, 256, 256)

    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio=ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __getitem__(self, index):
        cover_img, hidden_img, cover_img_path, hidden_img_path, mask = self.load_img(index)
        # mask should be generated based on hidden_img since we apply mask to the watermarked image
        # mask = self.load_mask(hidden_img)#for training
        # mask = self.Load_mask_test(hidden_img, user_mask_path)
        return {
            'cover_img': cover_img,
            'hidden_img': hidden_img,
            'cover_img_path': cover_img_path,
            'hidden_img_path': hidden_img_path,
            'mask': mask
        }

class inpaint_dataset(Dataset):
    def __init__(self, path, mask_type, micro=True, is_train=True, min_ratio=None, max_ratio=None, ratio=None):
        self.mask_type = mask_type
        self.target_size = (256,256)
        self.ratio = ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        if os.path.isdir(path):
            if is_train:
                # txt_path = os.path.join(path, 'cover_train.txt')
                cover_txt_path = os.path.join(path, 'cover_train.txt')
                hidden_txt_path = os.path.join(path, 'hidden_train.txt')
            else:
                cover_txt_path = os.path.join(path, 'cover_test.txt')
                hidden_txt_path = os.path.join(path, 'hidden_test.txt')
        # else:
        #     txt_path = path

        with open(cover_txt_path, 'r') as f:
            self.cover_imgs = [_.rstrip('\n') for _ in f.readlines()]
        with open(hidden_txt_path, 'r') as f:
            self.hidden_imgs = [_.rstrip('\n') for _ in f.readlines()]


    def __len__(self):
        return len(self.cover_imgs)

    def load_img(self, index):
        # img_path = self.imgs[index]

        # img = Image.open(img_path).convert('RGB')
        # img = img.resize((256, 256), Image.Resampling.LANCZOS)
        # img = transforms.ToTensor()(img)

        cover_img_path = self.cover_imgs[index]
        hidden_img_path = self.hidden_imgs[index]

        cover_img = Image.open(cover_img_path).convert('RGB')
        hidden_img = Image.open(hidden_img_path).convert('RGB')
        cover_img = cover_img.resize((256, 256), Image.Resampling.LANCZOS)
        # Direct tensor processing, avoid numbery:PIL image->tensor
        cover_img = torch.tensor(list(cover_img.getdata()), dtype=torch.uint8)
        cover_img = cover_img.view(256, 256, 3).permute(2, 0, 1).float() / 255.0

        hidden_img = hidden_img.resize((256, 256), Image.Resampling.LANCZOS)
        hidden_img = torch.tensor(list(hidden_img.getdata()), dtype=torch.uint8)
        hidden_img = hidden_img.view(256, 256, 3).permute(2, 0, 1).float() / 255.0

        return cover_img, hidden_img, cover_img_path, hidden_img_path

        # return img, img_path

    def load_mask(self, img):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mask_type) - 1)
        mask_type = self.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return mask_gen.center_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random regular mask
        if mask_type == 1:
            return mask_gen.random_regular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random irregular mask
        if mask_type == 2:
            return mask_gen.random_irregular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # from gated convolution (iccv 2019)
        if mask_type == 4:
            return mask_gen.random_freefrom_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)


    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio=ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __getitem__(self, index):
        cover_img, hidden_img, cover_img_path, hidden_img_path = self.load_img(index)
        # mask should be generated based on hidden_img since we apply mask to the watermarked image
        mask = self.load_mask(hidden_img)#for training
        # mask = self.Load_mask_test(hidden_img, hidden_img_path) #for visable watermark removal
        return {
            'cover_img': cover_img,
            'hidden_img': hidden_img,
            'cover_img_path': cover_img_path,
            'hidden_img_path': hidden_img_path,
            'mask': mask
        }
class inpaint_dataset512(Dataset):
    def __init__(self, path, mask_type, micro=True, is_train=True, min_ratio=None, max_ratio=None, ratio=None):
        self.mask_type = mask_type
        self.target_size = (512,512)
        self.ratio = ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        if os.path.isdir(path):
            if is_train:
                # txt_path = os.path.join(path, 'cover_train.txt')
                cover_txt_path = os.path.join(path, 'cover_train.txt')
                hidden_txt_path = os.path.join(path, 'hidden_train.txt')
            else:
                cover_txt_path = os.path.join(path, 'cover_test.txt')
                hidden_txt_path = os.path.join(path, 'hidden_test.txt')
        # else:
        #     txt_path = path

        with open(cover_txt_path, 'r') as f:
            self.cover_imgs = [_.rstrip('\n') for _ in f.readlines()]
        with open(hidden_txt_path, 'r') as f:
            self.hidden_imgs = [_.rstrip('\n') for _ in f.readlines()]


    def __len__(self):
        return len(self.cover_imgs)

    def load_img(self, index):
        # img_path = self.imgs[index]

        # img = Image.open(img_path).convert('RGB')
        # img = img.resize((256, 256), Image.Resampling.LANCZOS)
        # img = transforms.ToTensor()(img)

        cover_img_path = self.cover_imgs[index]
        hidden_img_path = self.hidden_imgs[index]

        cover_img = Image.open(cover_img_path).convert('RGB')
        hidden_img = Image.open(hidden_img_path).convert('RGB')
        cover_img = cover_img.resize((512, 512), Image.Resampling.LANCZOS)
        # Direct tensor processing, avoid numbery:PIL image->tensor
        cover_img = torch.tensor(list(cover_img.getdata()), dtype=torch.uint8)
        cover_img = cover_img.view(512, 512, 3).permute(2, 0, 1).float() / 255.0

        hidden_img = hidden_img.resize((512, 512), Image.Resampling.LANCZOS)
        hidden_img = torch.tensor(list(hidden_img.getdata()), dtype=torch.uint8)
        hidden_img = hidden_img.view(512, 512, 3).permute(2, 0, 1).float() / 255.0

        return cover_img, hidden_img, cover_img_path, hidden_img_path

        # return img, img_path

    def load_mask(self, img):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mask_type) - 1)
        mask_type = self.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return mask_gen.center_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random regular mask
        if mask_type == 1:
            return mask_gen.random_regular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random irregular mask
        if mask_type == 2:
            return mask_gen.random_irregular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # from gated convolution (iccv 2019)
        if mask_type == 4:
            return mask_gen.random_freefrom_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)


    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio=ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __getitem__(self, index):
        cover_img, hidden_img, cover_img_path, hidden_img_path = self.load_img(index)
        # mask should be generated based on hidden_img since we apply mask to the watermarked image
        mask = self.load_mask(hidden_img)#for training
        # mask = self.Load_mask_test(hidden_img, hidden_img_path) #for visable watermark removal
        return {
            'cover_img': cover_img,
            'hidden_img': hidden_img,
            'cover_img_path': cover_img_path,
            'hidden_img_path': hidden_img_path,
            'mask': mask
        }

class inpaint_dataset_clean(Dataset):
    def __init__(self, path, mask_type, micro=True, is_train=True, min_ratio=None, max_ratio=None,
                 translate_m_range=(0, 20), translate_n_range=(0, 20)):
        """
        Args:
            translate_m_range: Range of pixels to move to the left equal (min, max)
            translate_n_range: The range of pixels to move up equals (min, max)
        """
        self.mask_type = mask_type
        self.target_size = (256,256)
        self.ratio = None
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.translate_m_range = translate_m_range
        self.translate_n_range = translate_n_range

        if os.path.isdir(path):
            if is_train:
                hidden_txt_path = os.path.join(path, 'hidden_train.txt')
            else:
                hidden_txt_path = os.path.join(path, 'hidden_test.txt')
        # else:
        #     txt_path = path

        # cover_imgs is no longer required because cover_img will be generated from hidden_img
        with open(hidden_txt_path, 'r') as f:
            self.hidden_imgs = [_.rstrip('\n') for _ in f.readlines()]


    def __len__(self):
        return len(self.hidden_imgs)

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

    def load_img(self, index):
        hidden_img_path = self.hidden_imgs[index]

        # Load hidden_img and convert to tansor (one-time completion, avoid multiple conversions)
        hidden_img = Image.open(hidden_img_path).convert('RGB')
        hidden_img = hidden_img.resize((256, 256), Image.Resampling.LANCZOS)
        # Use more efficient conversion methods
        hidden_img_tensor = transforms.ToTensor()(hidden_img)  # [C,H,W], range of values [0,1]

        # Randomly generate shift parameters m and n
        m = random.randint(self.translate_m_range[0], self.translate_m_range[1])
        n = random.randint(self.translate_n_range[0], self.translate_n_range[1])

        # Conduct efficient migration and resize operations on tensor
        cover_img = self.translate_and_resize_tensor(hidden_img_tensor, m, n, target_size=256)

        # cover_img_path uses hidden_img_path because cover_img is generated from hidden_img
        cover_img_path = hidden_img_path

        return cover_img, hidden_img_tensor, cover_img_path, hidden_img_path

    def load_mask(self, img):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mask_type) - 1)
        mask_type = self.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return mask_gen.center_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random regular mask
        if mask_type == 1:
            return mask_gen.random_regular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random irregular mask
        if mask_type == 2:
            return mask_gen.random_irregular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # from gated convolution (iccv 2019)
        if mask_type == 4:
            return mask_gen.random_freefrom_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)


    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio=ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __getitem__(self, index):
        cover_img, hidden_img, cover_img_path, hidden_img_path = self.load_img(index)
        # mask should be generated based on hidden_img since we apply mask to the watermarked image
        mask = self.load_mask(hidden_img)
        return {
            'cover_img': cover_img,
            'hidden_img': hidden_img,
            'cover_img_path': cover_img_path,
            'hidden_img_path': hidden_img_path,
            'mask': mask
        }


class inpaint_dataset_nomarked(Dataset):
    def __init__(self, path, mask_type, micro=True, is_train=True, min_ratio=None, max_ratio=None,
                 translate_m_range=(3, 20), translate_n_range=(3, 20)):
        """
        Args:
            translate_m_range: Range of pixels to move to the left equal (min, max)
            translate_n_range: The range of pixels to move up equals (min, max)
        """
        self.mask_type = mask_type
        self.target_size = (256,256)
        self.ratio = None
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.translate_m_range = translate_m_range
        self.translate_n_range = translate_n_range

        if os.path.isdir(path):
            if is_train:
                hidden_txt_path = os.path.join(path, 'cover_train.txt')
            else:
                hidden_txt_path = os.path.join(path, 'cover_test.txt')
        # else:
        #     txt_path = path

        # cover_imgs is no longer required because cover_img will be generated from hidden_img
        with open(hidden_txt_path, 'r') as f:
            self.hidden_imgs = [_.rstrip('\n') for _ in f.readlines()]


    def __len__(self):
        return len(self.hidden_imgs)

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

    def load_img(self, index):
        hidden_img_path = self.hidden_imgs[index]

        # Load hidden_img and convert to tansor (one-time completion, avoid multiple conversions)
        hidden_img = Image.open(hidden_img_path).convert('RGB')
        hidden_img = hidden_img.resize((256, 256), Image.Resampling.LANCZOS)
        # Use more efficient conversion methods
        hidden_img_tensor = transforms.ToTensor()(hidden_img)  # [C,H,W], range of values [0,1]

        # Randomly generate shift parameters m and n
        m = random.randint(self.translate_m_range[0], self.translate_m_range[1])
        n = random.randint(self.translate_n_range[0], self.translate_n_range[1])

        # Conduct efficient migration and resize operations on tensor
        cover_img = self.translate_and_resize_tensor(hidden_img_tensor, m, n, target_size=256)

        # cover_img_path uses hidden_img_path because cover_img is generated from hidden_img
        cover_img_path = hidden_img_path

        return cover_img, hidden_img_tensor, cover_img_path, hidden_img_path

    def load_mask(self, img):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mask_type) - 1)
        mask_type = self.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return mask_gen.center_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random regular mask
        if mask_type == 1:
            return mask_gen.random_regular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random irregular mask
        if mask_type == 2:
            return mask_gen.random_irregular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # from gated convolution (iccv 2019)
        if mask_type == 4:
            return mask_gen.random_freefrom_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)


    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio=ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __getitem__(self, index):
        cover_img, hidden_img, cover_img_path, hidden_img_path = self.load_img(index)
        # mask should be generated based on hidden_img since we apply mask to the watermarked image
        mask = self.load_mask(hidden_img)
        return {
            'cover_img': cover_img,
            'hidden_img': hidden_img,
            'cover_img_path': cover_img_path,
            'hidden_img_path': hidden_img_path,
            'mask': mask
        }


class RatioDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=False, ratio=0.5):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.dataset = dataset
        self.ratio = ratio

    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio = ratio
        # self.dataset.set_ratio(ratio, min_ratio, max_ratio)

class inpaint_dataset_remove(Dataset):
    def __init__(self, path, mask_type, micro=True, is_train=True, min_ratio=None, max_ratio=None):
        self.mask_type = mask_type
        self.target_size = (256,256)
        self.ratio = None
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        if os.path.isdir(path):
            if is_train:
                # txt_path = os.path.join(path, 'cover_train.txt')
                cover_txt_path = os.path.join(path, 'cover_train.txt')
                hidden_txt_path = os.path.join(path, 'hidden_train.txt')
            else:
                cover_txt_path = os.path.join(path, 'cover_test.txt')
                hidden_txt_path = os.path.join(path, 'hidden_test.txt')
        # else:
        #     txt_path = path

        with open(cover_txt_path, 'r') as f:
            self.cover_imgs = [_.rstrip('\n') for _ in f.readlines()]
        with open(hidden_txt_path, 'r') as f:
            self.hidden_imgs = [_.rstrip('\n') for _ in f.readlines()]


    def __len__(self):
        return len(self.cover_imgs)

    def shuffle_image_patches(self, img, shuffled_indices=None):
        """
        Split the image into 4x4 pieces, record the original location, and then randomly disrupt it.
        Args:
            img: torch.Tensor, shape (C, H, W) = (3, 256, 256)
            shuffled_indices: optional torch.Tensor of shape (16,); generated randomly when omitted
        Returns:
            shuffled_img: torch.Tensor, shape (C, H, W) = (3, 256, 256)
            position_matrix: torch.Tensor, shape (4, 4) Record raw block indexes for each location
        """
        C, H, W = img.shape
        patch_size = 64  # 256 / 4 = 64

        # Split images into 4x4
        patches = []
        for i in range(4):
            for j in range(4):
                patch = img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patches.append(patch)

        # Could not close temporary folder: %s
        if shuffled_indices is None:
            shuffled_indices = torch.randperm(16)

        # Disruption patches
        shuffled_patches = [patches[idx] for idx in shuffled_indices]

        # Record position matrix: which index of the current block from the original location is stored in each location (i,j)
        position_matrix = torch.zeros(4, 4, dtype=torch.long)
        for new_idx, old_idx in enumerate(shuffled_indices):
            row = new_idx // 4
            col = new_idx % 4
            position_matrix[row, col] = old_idx

        # Reassemble as Image
        shuffled_img = torch.zeros_like(img)
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                shuffled_img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = shuffled_patches[idx]

        return shuffled_img, position_matrix

    def restore_image_patches(self, shuffled_img, position_matrix):
        """
        Restores the image of a broken image to the original image according to position_matrix
        Args:
            shuffled_img: torch.Tensor, shape (C, H, W) = (3, 256, 256), Disruption of Images
            position_matrix: torch.Tensor, shape (4, 4), Position Matrix, Record raw block index for each new location
        Returns:
            restored_img: torch.Tensor, shape (C, H, W) = (3, 256, 256), Restored original diagram
        """
        C, H, W = shuffled_img.shape
        patch_size = 64  # 256 / 4 = 64

        # Disrupte images into four x four pieces.
        shuffled_patches = []
        for i in range(4):
            for j in range(4):
                patch = shuffled_img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                shuffled_patches.append(patch)

        # Creates a restored image
        restored_img = torch.zeros_like(shuffled_img)

        # Restore the original map according to position_matrix
        # position_matrix[i, j] indicates new position (i,j) from original positionold_idx
        # We need to put the new position (i.j.) back to the original location.
        for i in range(4):
            for j in range(4):
                old_idx = position_matrix[i, j].item()  # Native Location Index
                old_row = old_idx // 4  # Original Line Location
                old_col = old_idx % 4   # Location of the original column

                # Place new (i,j) blocks back to their original (old_row, old_col)
                restored_img[:, old_row*patch_size:(old_row+1)*patch_size,
                            old_col*patch_size:(old_col+1)*patch_size] = shuffled_patches[i * 4 + j]

        return restored_img

    def load_img(self, index):
        # img_path = self.imgs[index]

        # img = Image.open(img_path).convert('RGB')
        # img = img.resize((256, 256), Image.Resampling.LANCZOS)
        # img = transforms.ToTensor()(img)

        cover_img_path = self.cover_imgs[index]
        hidden_img_path = self.hidden_imgs[index]

        cover_img = Image.open(cover_img_path).convert('RGB')
        hidden_img = Image.open(hidden_img_path).convert('RGB')
        cover_img = cover_img.resize((256, 256), Image.Resampling.LANCZOS)
        # Direct tensor processing, avoid numbery:PIL image->tensor
        cover_img = torch.tensor(list(cover_img.getdata()), dtype=torch.uint8)
        cover_img = cover_img.view(256, 256, 3).permute(2, 0, 1).float() / 255.0

        hidden_img = hidden_img.resize((256, 256), Image.Resampling.LANCZOS)
        hidden_img = torch.tensor(list(hidden_img.getdata()), dtype=torch.uint8)
        hidden_img = hidden_img.view(256, 256, 3).permute(2, 0, 1).float() / 255.0

        # Generate a uniform mess index that allows cover_img and hidden_img to do the same.
        shuffled_indices = torch.randperm(16)

        # Blocking of cover_img and hidden_img, using the same indexing
        cover_img, position_matrix = self.shuffle_image_patches(cover_img, shuffled_indices)
        hidden_img, _ = self.shuffle_image_patches(hidden_img, shuffled_indices)

        return cover_img, hidden_img, cover_img_path, hidden_img_path, position_matrix

        # return img, img_path

    def load_mask(self, img):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mask_type) - 1)
        mask_type = self.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return mask_gen.center_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random regular mask
        if mask_type == 1:
            return mask_gen.random_regular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random irregular mask
        if mask_type == 2:
            return mask_gen.random_irregular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # from gated convolution (iccv 2019)
        if mask_type == 4:
            return mask_gen.random_freefrom_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)


    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio=ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __getitem__(self, index):
        cover_img, hidden_img, cover_img_path, hidden_img_path, position_matrix = self.load_img(index)
        # mask should be generated based on hidden_img since we apply mask to the watermarked image
        mask = self.load_mask(hidden_img)
        return {
            'cover_img': cover_img,
            'hidden_img': hidden_img,
            'cover_img_path': cover_img_path,
            'hidden_img_path': hidden_img_path,
            'mask': mask,
            'position_matrix': position_matrix
        }

class inpaint_dataset_wk(Dataset):
    def __init__(self, path, mask_type, micro=True, is_train=True, min_ratio=None, max_ratio=None):
        self.mask_type = mask_type
        self.target_size = (256,256)


        if os.path.isdir(path):
            if is_train:
                cover_txt_path = os.path.join(path, 'cover_train.txt')
                hidden_txt_path = os.path.join(path, 'hidden_train.txt')
            else:
                cover_txt_path = os.path.join(path, 'cover_test.txt')
                hidden_txt_path = os.path.join(path, 'hidden_test.txt')
        # else:
        #     txt_path = path

        with open(cover_txt_path, 'r') as f:
            self.cover_imgs = [_.rstrip('\n') for _ in f.readlines()]
        with open(hidden_txt_path, 'r') as f:
            self.hidden_imgs = [_.rstrip('\n') for _ in f.readlines()]


    def __len__(self):
        return len(self.cover_imgs)

    def load_img(self, index):
        cover_img_path = self.cover_imgs[index]
        hidden_img_path = self.hidden_imgs[index]

        cover_img = Image.open(cover_img_path).convert('RGB')
        hidden_img = Image.open(hidden_img_path).convert('RGB')
        cover_img = cover_img.resize((256, 256), Image.Resampling.LANCZOS)
        # Direct use of tensor to avoid ToTensor version compatibility
        cover_array = np.array(cover_img, dtype=np.uint8)
        cover_img = torch.tensor(cover_array, dtype=torch.float32).permute(2, 0, 1) / 255.0

        hidden_img = hidden_img.resize((256, 256), Image.Resampling.LANCZOS)
        hidden_array = np.array(hidden_img, dtype=np.uint8)
        hidden_img = torch.tensor(hidden_array, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return cover_img, hidden_img, hidden_img_path


    def __getitem__(self, index):
        cover_img, hidden_img, cover_img_path = self.load_img(index)

        return {'cover_img': cover_img, 'hidden_img': hidden_img,'cover_img_path':cover_img_path}

class inpaint_dataset_wk_moe(Dataset):
    # Watermark Type Tab Map
    WATERMARK_LABELS = {
        'vine': 0,
        'tree': 1,
        'rosteals': 2,
        'clean': 3,
    }

    def __init__(self, path, mask_type, micro=True, is_train=True, min_ratio=None, max_ratio=None):
        self.mask_type = mask_type
        self.target_size = (256,256)


        if os.path.isdir(path):
            if is_train:
                cover_txt_path = os.path.join(path, 'cover_train.txt')
                hidden_txt_path = os.path.join(path, 'hidden_train.txt')
            else:
                cover_txt_path = os.path.join(path, 'cover_test.txt')
                hidden_txt_path = os.path.join(path, 'hidden_test.txt')
        else:
            # If Path isn’t a directory, make a mistake.
            raise ValueError(
                f"Path must be a directory containing the required text files. "
                f"Expected directory: {path}\n"
                f"For training: 'cover_train.txt' and 'hidden_train.txt'\n"
                f"For testing: 'cover_test.txt' and 'hidden_test.txt'"
            )

        # Check if a file exists.
        if not os.path.exists(cover_txt_path):
            raise FileNotFoundError(f"Cover text file not found: {cover_txt_path}")
        if not os.path.exists(hidden_txt_path):
            raise FileNotFoundError(f"Hidden text file not found: {hidden_txt_path}")

        with open(cover_txt_path, 'r') as f:
            self.cover_imgs = [_.rstrip('\n') for _ in f.readlines()]
        with open(hidden_txt_path, 'r') as f:
            self.hidden_imgs = [_.rstrip('\n') for _ in f.readlines()]

        # Mask Related Parameters
        self.ratio = max_ratio if max_ratio else 0.5
        self.min_ratio = min_ratio if min_ratio else 0.1
        self.max_ratio = max_ratio if max_ratio else 0.5

    def load_mask(self, img):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mask_type) - 1)
        mask_type = self.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return mask_gen.center_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random regular mask
        if mask_type == 1:
            return mask_gen.random_regular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # random irregular mask
        if mask_type == 2:
            return mask_gen.random_irregular_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # from gated convolution (iccv 2019)
        if mask_type == 4:
            return mask_gen.random_freefrom_mask(img, ratio=self.ratio, max_ratio=self.max_ratio, min_ratio=self.min_ratio)

        # Default returns full 1 mask (no shield)
        return torch.ones(1, img.shape[1], img.shape[2])

    def set_ratio(self, ratio, min_ratio, max_ratio):
        self.ratio = ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def get_watermark_label(self, hidden_img_path):
        """
        Watermark type based on hidden_img path
        Supported path format:
        1. /data/rosteals/wk/wm_xxx.jpg → rosteals (Label 2)
        2. /data/tree_ring/w/xxx_w.jpg → tree (Label 1)
        3. /data/512_512/hidden/xxx_wm.png → vine (Label 0, Presumably based on folder name)
        4. /data/vine/xxx.jpg → vine (Label 0)
        5. /data/tree/xxx.jpg → tree (Label 1)

        Back: 0=vine, 1=tree, 2=rosteals, 3=clean
        """
        path_lower = hidden_img_path.lower()

        # Priority 1: Check standard formatting (vine, tree, rules, clean)
        for wm_type, label in self.WATERMARK_LABELS.items():
            if f'/{wm_type}/' in path_lower or f'\\{wm_type}\\' in path_lower:
                return label

        # Priority 2: Check special path formats
        # Tree_ring Folder _Tree Type
        if '/tree_ring/' in path_lower or '\\tree_ring\\' in path_lower:
            return self.WATERMARK_LABELS['tree']

        # 512_512 folder  vine type (based on models+vine+tree_ring111 folder name)
        if '/512_512/' in path_lower or '\\512_512\\' in path_lower:
            return self.WATERMARK_LABELS['vine']

        # returns clear (3) by default
        return 3


    def __len__(self):
        return len(self.cover_imgs)

    def load_img(self, index):
        cover_img_path = self.cover_imgs[index]
        hidden_img_path = self.hidden_imgs[index]

        cover_img = Image.open(cover_img_path).convert('RGB')
        hidden_img = Image.open(hidden_img_path).convert('RGB')
        cover_img = cover_img.resize((256, 256), Image.Resampling.LANCZOS)
        # Direct tensor processing, avoid numbery:PIL image->tensor
        cover_img = torch.tensor(list(cover_img.getdata()), dtype=torch.uint8)
        cover_img = cover_img.view(256, 256, 3).permute(2, 0, 1).float() / 255.0

        hidden_img = hidden_img.resize((256, 256), Image.Resampling.LANCZOS)
        hidden_img = torch.tensor(list(hidden_img.getdata()), dtype=torch.uint8)
        hidden_img = hidden_img.view(256, 256, 3).permute(2, 0, 1).float() / 255.0

        return cover_img, hidden_img, cover_img_path, hidden_img_path  # Returns two paths


    def __getitem__(self, index):
        cover_img, hidden_img, cover_img_path, hidden_img_path = self.load_img(index)  # Receive Two Paths

        # Get Watermark Type Tags
        watermark_label = self.get_watermark_label(hidden_img_path)

        # Generate mask
        mask = self.load_mask(hidden_img)

        return {
            'cover_img': cover_img,
            'hidden_img': hidden_img,
            'cover_img_path': cover_img_path,
            'hidden_img_path': hidden_img_path,  # Add hidden_img_path
            'watermark_label': watermark_label,  # Watermark type label: 0=vine, 1=tree, 2=rostials, 3=clean
            'mask': mask,  # Mask
        }
