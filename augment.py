# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF

from torchvision import transforms
from torchvision import datasets, transforms
from PIL import ImageFilter, ImageOps, Image
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class MultiCrop(object):
    def __init__(
        self, global_crops_scale, local_crops_scale, local_crops_number, rand_aug=False
    ):
        rand_aug = transforms.RandAugment(num_ops=3)
        self.local_crops_number = local_crops_number
        color_jitter = transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=0.8,
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # self.local_transforms = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             96, scale=local_crops_scale, interpolation=Image.BICUBIC
        #         ),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.RandAugment(num_ops=3),
        #         GaussianBlur(1.0),
        #         normalize,
        #     ]
        # )

        # self.global_transforms1 = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             224, scale=local_crops_scale, interpolation=Image.BICUBIC
        #         ),
        #         transforms.Resize(224, interpolation=3),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.RandAugment(num_ops=2),
        #         GaussianBlur(1.0),
        #         normalize,
        #     ]
        # )

        # self.global_transforms2 = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             224, scale=local_crops_scale, interpolation=Image.BICUBIC
        #         ),
        #         transforms.Resize(224, interpolation=3),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.RandAugment(num_ops=3),
        #         GaussianBlur(1.0),
        #         normalize,
        #     ]
        # )

        if rand_aug:
            print("==== Using Rand-Augment")
            flip_and_color_jitter = transforms.RandAugment(num_ops=3)
        else:
            print("==== Using Normal flip and color jitter for multi-crop")
            flip_and_color_jitter = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]
            )

        # first global crop
        self.global_transforms1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                transforms.Resize(224, interpolation=3),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transforms2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                transforms.Resize(224, interpolation=3),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        # crops.append(self.global_transfo1(image))
        # crops.append(self.global_transfo2(image))
        crops.append(self.global_transforms1(image))
        crops.append(self.global_transforms2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transforms(image))
        return crops


class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2, activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def three_augment(args=None):
    """
    * 3-Augment from DeiT-III
    * (https://arxiv.org/pdf/2204.07118.pdf)
    """

    img_size = args.input_size
    remove_random_resized_crop = args.src
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    secondary_tfl = [
        transforms.Resize(img_size, interpolation=3),
        transforms.RandomCrop(img_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice(
            [gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)]
        ),
    ]

    if args.color_jitter is not None and not args.color_jitter == 0:
        secondary_tfl.append(
            transforms.ColorJitter(
                args.color_jitter, args.color_jitter, args.color_jitter
            )
        )
    final_tfl = [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    return transforms.Compose(secondary_tfl + final_tfl)


def new_data_aug_generator(args=None):
    img_size = args.input_size
    remove_random_resized_crop = args.src
    named_loss = args.named_loss
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    primary_tfl = []
    scale = (0.08, 1.0)
    interpolation = "bicubic"
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
        ]

    secondary_tfl = [
        transforms.RandomChoice(
            [gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)]
        )
    ]

    if args.color_jitter is not None and not args.color_jitter == 0:
        secondary_tfl.append(
            transforms.ColorJitter(
                args.color_jitter, args.color_jitter, args.color_jitter
            )
        )
    final_tfl = [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)