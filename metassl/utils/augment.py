import math

import torch
import torchvision.transforms
import torchvision.transforms as transforms
from torch import Tensor, nn
from torchvision.transforms import RandomApply, ColorJitter, GaussianBlur, RandomHorizontalFlip, RandomGrayscale, RandomResizedCrop

from metassl.utils.data import normalize_imagenet
from metassl.utils.simsiam import TwoCropsTransform
from metassl.utils.torch_utils import get_sample_logprob


class DataAugmentation(nn.Module):
    """Module to perform data augmentation."""
    
    def __init__(self) -> None:
        super().__init__()
        
        # augmentation strengths
        self.color_jitter_strengths_brightness = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        self.color_jitter_strengths_contrast = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        self.color_jitter_strengths_saturation = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        self.color_jitter_strengths_hue = [0.0, 0.1, 0.2, 0.3, 0.4]
        
        self.aug_w_b = nn.Parameter(torch.zeros(len(self.color_jitter_strengths_brightness)), requires_grad=True)
        self.aug_w_c = nn.Parameter(torch.zeros(len(self.color_jitter_strengths_contrast)), requires_grad=True)
        self.aug_w_s = nn.Parameter(torch.zeros(len(self.color_jitter_strengths_saturation)), requires_grad=True)
        self.aug_w_h = nn.Parameter(torch.zeros(len(self.color_jitter_strengths_hue)), requires_grad=True)
        
        bound = 1. / math.sqrt(self.aug_w_b.size(0))
        bound_h = 1. / math.sqrt(self.aug_w_h.size(0))
        
        nn.init.uniform(self.aug_w_b, -bound, bound)
        nn.init.uniform(self.aug_w_c, -bound, bound)
        nn.init.uniform(self.aug_w_s, -bound, bound)
        nn.init.uniform(self.aug_w_h, -bound_h, bound_h)
        
        self.aug_param_dict = {
            "aug_w_b": self.aug_w_b,
            "aug_w_c": self.aug_w_c,
            "aug_w_s": self.aug_w_s,
            "aug_w_h": self.aug_w_h
            }
        
        # histograms
        self.color_jitter_histogram_brightness = {k: 0 for k in self.color_jitter_strengths_brightness}
        self.color_jitter_histogram_contrast = {k: 0 for k in self.color_jitter_strengths_contrast}
        self.color_jitter_histogram_saturation = {k: 0 for k in self.color_jitter_strengths_saturation}
        self.color_jitter_histogram_hue = {k: 0 for k in self.color_jitter_strengths_hue}
        
        self.color_jitter_hists = {
            "b": self.color_jitter_histogram_brightness,
            "c": self.color_jitter_histogram_contrast,
            "s": self.color_jitter_histogram_saturation,
            "h": self.color_jitter_histogram_hue
            }
        
        self.color_jitter_strengths = {
            "b": self.color_jitter_strengths_brightness,
            "c": self.color_jitter_strengths_contrast,
            "s": self.color_jitter_strengths_saturation,
            "h": self.color_jitter_strengths_hue
            }
    
    def sample_logprobs(self):
        color_jitter_action_idx_b, color_jitter_logprob_b, _ = get_sample_logprob(logits=self.aug_w_b)
        color_jitter_action_idx_c, color_jitter_logprob_c, _ = get_sample_logprob(logits=self.aug_w_c)
        color_jitter_action_idx_s, color_jitter_logprob_s, _ = get_sample_logprob(logits=self.aug_w_s)
        color_jitter_action_idx_h, color_jitter_logprob_h, _ = get_sample_logprob(logits=self.aug_w_h)
        
        indices = {
            "idx_b": color_jitter_action_idx_b,
            "idx_c": color_jitter_action_idx_c,
            "idx_s": color_jitter_action_idx_s,
            "idx_h": color_jitter_action_idx_h,
            }
        
        logprobs = {
            "logprob_b": color_jitter_logprob_b,
            "logprob_c": color_jitter_logprob_c,
            "logprob_s": color_jitter_logprob_s,
            "logprob_h": color_jitter_logprob_h
            }
        
        return indices, logprobs
    
    def forward(self, x: Tensor, idx_b: int, idx_c: int, idx_s: int, idx_h: int) -> Tensor:
        strength_b = self.color_jitter_strengths_brightness[idx_b]
        strength_c = self.color_jitter_strengths_contrast[idx_c]
        strength_s = self.color_jitter_strengths_saturation[idx_s]
        strength_h = self.color_jitter_strengths_hue[idx_h]
        
        self.color_jitter_histogram_brightness[strength_b] += 1
        self.color_jitter_histogram_contrast[strength_c] += 1
        self.color_jitter_histogram_saturation[strength_s] += 1
        self.color_jitter_histogram_hue[strength_h] += 1
        
        trans = nn.Sequential(
            RandomResizedCrop(224, scale=(0.2, 1.)),
            RandomApply([ColorJitter(brightness=strength_b, contrast=strength_c, saturation=strength_s, hue=strength_h)], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            RandomHorizontalFlip(),
            normalize_imagenet
            )
        
        x_out = trans(x)  # BxCxHxW
        x_out = TwoCropsTransform(x_out)
        return x_out


def create_transforms(strengths_b, strengths_c, strengths_s, strengths_h, image_height=224, image_width=224):
    if isinstance(strengths_b, float):
        strengths_b, strengths_c, strengths_s, strengths_h = [strengths_b], [strengths_c], [strengths_s], [strengths_h]
    
    assert len(strengths_b) == len(strengths_c) == len(strengths_s) == len(strengths_h), "strengths must all have the same length"
    
    transforms_list = []
    
    kernel_h = image_height // 10
    kernel_w = image_width // 10
    
    if kernel_h % 2 == 0:
        kernel_h -= 1
    if kernel_w % 2 == 0:
        kernel_w -= 1
    
    for i in range(len(strengths_b)):
        
        trans = TwoCropsTransform(
            transforms.Compose(
                [
                    transforms.RandomApply([transforms.ColorJitter(brightness=strengths_b[i], contrast=strengths_c[i], saturation=strengths_s[i], hue=strengths_h[i])], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([torchvision.transforms.GaussianBlur([kernel_h, kernel_w], [.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    normalize_imagenet
                    ]
                )
            )
        
        transforms_list.append(trans)
    
    return transforms_list


def augment_per_image(transformed_list, images):
    assert len(transformed_list) == images.shape[0]
    
    print(images.size())
    images = [transform(img.unsqueeze(0)) for transform, img in zip(transformed_list, images)]
    print(images)
    print(torch.cat(images, dim=1).size())
    
    return torch.cat(images, dim=1)
