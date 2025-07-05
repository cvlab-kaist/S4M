# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

__all__ = ["MaskFormerImageDatasetMapper"]

import random
from PIL import ImageFilter
import torchvision.transforms as transforms
from PIL import Image
import json 
from pycocotools import mask as mask_utils


import sys
import pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def build_strong_augmentation(is_train=True):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        # randcrop_transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.RandomErasing(
        #             p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
        #         ),
        #         transforms.RandomErasing(
        #             p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
        #         ),
        #         transforms.RandomErasing(
        #             p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
        #         ),
        #         transforms.ToPILImage(),
        #     ]
        # )
        # augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    img_w, img_h = img_size
    mask = torch.zeros(img_h, img_w)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_h * img_w
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_w)
        y = np.random.randint(0, img_h)

        if x + cutmix_w <= img_w and y + cutmix_h <= img_h:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

## unlabeled dataset
## 수정중 -- 에러나면 original로 돌리기
class MaskFormerImageDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for instance segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        size_divisibility,
        cutmix = False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility
        self.cutmix = cutmix
        self.strong_aug = build_strong_augmentation()
        # self.strong_aug2 = build_strong_augmentation()
        self.coloraugssdtransform = ColorAugSSDTransform(img_format='RGB')
        
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation

        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                )
            )
        # if cfg.INPUT.COLOR_AUG_SSD:
        #     augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        image_wo_weak = image
        image_wo_weak = torch.as_tensor(np.ascontiguousarray(image_wo_weak.transpose(2, 0, 1)))
        image = self.coloraugssdtransform.apply_image(image)

        # add strong augmentations later.
        image1 = Image.fromarray(image.astype("uint8"), "RGB")
        image_aug = self.strong_aug(image1)
        image_aug = torch.as_tensor(np.ascontiguousarray(image_aug).transpose(2, 0, 1))
        
        # image_aug_2 = self.strong_aug2(image1)
        # image_aug_2 = torch.as_tensor(np.ascontiguousarray(image_aug_2).transpose(2, 0, 1))
        # image_aug = torch.as_tensor(np.ascontiguousarray(image_aug.permute(2, 0, 1)))
        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            # pad image
            image = F.pad(image, padding_size, value=128).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image_wo_weak"] = image_wo_weak
        dataset_dict["image"] = image
        dataset_dict["image_aug"] = image_aug
        # dataset_dict["image_aug2"] = image_aug_2
        # dataset_dict["cutmix_box"] = obtain_cutmix_box(image_shape, p=0.5)
        

        return dataset_dict
