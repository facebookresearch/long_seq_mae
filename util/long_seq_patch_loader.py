# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import torch
import numpy as np


class SampleVisiblePatchIndices:
    def __init__(
        self, transforms, num_patches, mask_ratio, mask_downsampling,
    ):
        self.transforms = transforms
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.num_keep_patches = int(num_patches * (1 - mask_ratio))

        assert isinstance(mask_downsampling, int) and mask_downsampling >= 1
        self.mask_downsampling = mask_downsampling
        if self.mask_downsampling > 1:
            grid_size = int(np.sqrt(num_patches))
            assert num_patches == grid_size ** 2
            self.grid_size = grid_size
            self.rounding_needed = (grid_size % mask_downsampling) > 0
            mask_grid_size = math.ceil(grid_size / mask_downsampling)
            num_mask_patches = mask_grid_size ** 2
            self.grid_size = grid_size
            self.mask_grid_size = mask_grid_size
            self.num_mask_patches = num_mask_patches

    def _clip_by_grid(self, x):
        if not self.rounding_needed:
            return x
        return np.where(x < self.grid_size, x, -100000 - self.num_patches)

    def __call__(self, img):
        img = self.transforms(img)

        # generating shuffling and masking indices
        if self.mask_downsampling > 1:
            mask_ids_shuffle = np.random.permutation(self.num_mask_patches)
            mask_x = (mask_ids_shuffle % self.mask_grid_size) * self.mask_downsampling
            mask_y = (mask_ids_shuffle // self.mask_grid_size) * self.mask_downsampling
            stacks = [
                self._clip_by_grid(mask_y + ry) * self.grid_size + self._clip_by_grid(mask_x + rx)
                for rx in range(self.mask_downsampling)
                for ry in range(self.mask_downsampling)
            ]
            ids_shuffle = np.stack(stacks, axis=1).reshape(-1)
            if not self.rounding_needed:
                assert len(ids_shuffle) == self.num_patches, f"{len(ids_shuffle)} vs {self.num_patches}"
            else:
                ids_shuffle = ids_shuffle[ids_shuffle >= 0]
                assert len(ids_shuffle) == self.num_patches, f"{len(ids_shuffle)} vs {self.num_patches}"
        else:
            ids_shuffle = np.random.permutation(self.num_patches)
        ids_restore = np.empty(self.num_patches, dtype=np.int64)
        ids_restore[ids_shuffle] = np.arange(self.num_patches, dtype=np.int64)
        ids_keep = ids_shuffle[:self.num_keep_patches]

        ids_shuffle = torch.tensor(ids_shuffle, dtype=torch.long)
        ids_keep = torch.tensor(ids_keep, dtype=torch.long)
        ids_restore = torch.tensor(ids_restore, dtype=torch.long)

        out = {
            "img": img,
            "ids_shuffle": ids_shuffle,
            "ids_keep": ids_keep,
            "ids_restore": ids_restore,
        }

        return out


class MAEIndexCollator:
    def __call__(self, sample_and_label_list):
        sample_list = [sample for sample, _ in sample_and_label_list]

        imgs = torch.stack([sample["img"] for sample in sample_list])
        ids_keep = torch.stack([sample["ids_keep"] for sample in sample_list])
        ids_restore = torch.stack([sample["ids_restore"] for sample in sample_list])

        return imgs, ids_keep, ids_restore
