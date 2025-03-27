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

import math
from typing import Dict

import torch
from torch import nn

from einops import rearrange


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_inner: int = 0, device: torch.device = None, **kwargs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, device=device)
        self.norm = nn.LayerNorm(hidden_size, device=device)
        self.relu = nn.ReLU()

        inner = []
        for _ in range(num_inner):
            inner.extend([
                nn.Linear(hidden_size, hidden_size, device=device),
                nn.LayerNorm(hidden_size, device=device),
                nn.ReLU(),
            ])
        if inner:
            self.inner = nn.Sequential(*inner)
        else:
            self.inner = nn.Identity()

        self.fc2 = nn.Linear(hidden_size, output_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.inner(x)
        x = self.fc2(x)
        return x


class MLP2(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_inner: int = 0,
                 pre_norm: bool = False, device: torch.device = None,
                 upsample_factor: int = 1,
                 **kwargs):
        super().__init__()

        self.pre_norm = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.GELU(),
        ) if pre_norm else nn.Identity()

        self.upsample_factor = upsample_factor
        self._real_output_dim = output_size

        hidden_size *= upsample_factor
        output_size *= (upsample_factor ** 2)

        self.fc1 = nn.Linear(input_size, hidden_size, device=device)

        blocks = []
        for _ in range(num_inner):
            blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_size, device=device),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size, device=device),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.final = nn.Sequential(
            nn.LayerNorm(hidden_size, device=device),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.fc1(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.final(x)

        if self.upsample_factor > 1:
            h = w = int(math.sqrt(x.shape[1]))
            x = rearrange(x, 'b (h w) (u1 u2 c) -> b (u1 h u2 w) c',
                          h=h, w=w, u1=self.upsample_factor, u2=self.upsample_factor,
                          c=self._real_output_dim)

        return x


MLP_FACTORY = {
    'v1': MLP,
    'v2': MLP2,
}


def strip_prefix(state: Dict[str, torch.Tensor], prefix: str):
    state = {
        k[len(prefix):]: v
        for k, v in state.items()
        if k.startswith(prefix)
    }
    return state


def get_mlp_info_from_state(version: str, state: Dict[str, torch.Tensor], prefix: str = ''):
    state = strip_prefix(state, prefix)

    if version == 'v1':
        hidden_dim, input_dim = state['fc1.weight'].shape
        output_dim = state['fc2.weight'].shape[0]

        for num_inner in range(1000):
            k = f'inner.{num_inner}.0.weight'
            if k not in state:
                break
    elif version == 'v2':
        hidden_dim, input_dim = state['fc1.weight'].shape
        output_dim = state['final.2.weight'].shape[0]

        for num_inner in range(1000):
            k = f'blocks.{num_inner}.0.weight'
            if k not in state:
                break
    else:
        raise ValueError(f'Unsupported MLP version: {version}')

    return input_dim, hidden_dim, output_dim, num_inner


def create_mlp_from_config(version: str, input_dim: int, hidden_dim: int, output_dim: int, num_inner: int):
    ret: nn.Module = MLP_FACTORY[version](input_dim, hidden_dim, output_dim, num_inner)

    return ret


def create_mlp_from_state(version: str, state: Dict[str, torch.Tensor], prefix: str = ''):
    state = strip_prefix(state, prefix)

    input_dim, hidden_dim, output_dim, num_inner = get_mlp_info_from_state(version, state)

    ret: nn.Module = create_mlp_from_config(version, input_dim, hidden_dim, output_dim, num_inner)

    ret.load_state_dict(state)

    return ret