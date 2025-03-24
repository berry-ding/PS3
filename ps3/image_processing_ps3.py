from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from PIL import Image

import torch
import torchvision.transforms.functional as F
from torchvision.transforms.v2 import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop, ColorJitter, Grayscale
from torchvision.transforms.functional import normalize, to_tensor, resize

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.utils import TensorType

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def _convert_to_rgb(image):
    return image.convert('RGB')


class PS3ImageProcessor(BaseImageProcessor):
    def __init__(
        self, 
        image_size: Union[int, Tuple[int, int]] = None,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_mode: Optional[str] = None,
        interpolation: Optional[str] = None,
    ):
        self.image_size = image_size
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)

        self.mean = mean or OPENAI_DATASET_MEAN
        if not isinstance(self.mean, (list, tuple)):
            self.mean = (self.mean,) * 3

        self.std = std or OPENAI_DATASET_STD
        if not isinstance(self.std, (list, tuple)):
            self.std = (self.std,) * 3

        self.resize_mode = resize_mode or 'squash'
        assert self.resize_mode in ('squash')

        self.interpolation = interpolation or 'bicubic'
        assert self.interpolation in ['bicubic', 'bilinear', 'random']

        # Define some attributes to align with vila code
        self.size = {'shortest_edge': self.image_size[0]}
        self.crop_size = {'height': self.image_size[0], 'width': self.image_size[0]}
        self.image_mean = self.mean
        self.image_std = self.std
    
    def preprocess(
            self, 
            image: Image.Image,
            return_tensors: Optional[Union[str, TensorType]] = None,
        ):
        image = resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR if self.interpolation == 'bilinear' else InterpolationMode.BICUBIC)
        image = _convert_to_rgb(image)
        image = to_tensor(image)
        image = normalize(image, mean=self.mean, std=self.std)

        data = {"pixel_values": [image]}
        return data


