# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union, Any
from PIL import Image

from torchvision.transforms.v2 import Normalize, InterpolationMode, ToTensor, Resize
from torchvision.transforms.v2.functional import normalize, to_tensor, resize

from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import TensorType

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def _convert_to_rgb(image):
    if isinstance(image, tuple):
        return (image[0].convert('RGB'),) + image[1:]
    else:
        return image.convert('RGB')



class PS3ImageProcessor(BaseImageProcessor):
    def __init__(
        self, 
        image_size: Union[int, Tuple[int, int]] = None,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_mode: Optional[str] = None,
        interpolation: Optional[str] = None,
        **kwargs,
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
            image: Any,
            return_tensors: Optional[Union[str, TensorType]] = None,
        ):
        image = Resize(self.image_size, interpolation=InterpolationMode.BILINEAR if self.interpolation == 'bilinear' else InterpolationMode.BICUBIC)(image)
        image = _convert_to_rgb(image)
        image = ToTensor()(image)
        image = Normalize(mean=self.mean, std=self.std)(image)

        data = {"pixel_values": [image]}
        return data


