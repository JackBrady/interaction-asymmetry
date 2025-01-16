import math
from torch import nn
import numpy as np
import itertools
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_grid_indices(image_shape):
    xs = list(range(0, image_shape[1], 1))
    ys = list(range(0, image_shape[0], 1))
    coords = torch.Tensor(list(itertools.product(ys, xs)))/(image_shape[0]-1)
    return torch.unsqueeze(coords, 0)


def build_grid(resolution):
    """
    Builds grid from image given resolution
    Code taken from: https://github.com/evelinehong/slot-attention-pytorch/blob/master/model.py

    Args:
        resolution: tuple containing width and height of image (width, height)
    """
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Code from: https://github.com/evelinehong/slot-attention-pytorch/blob/master/model.py

        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class PositionalEncoding(nn.Module):
    # Code adapted from: https://github.com/stelzner/srt/blob/main/srt/layers.py
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords):
        batch_size, num_points, dim = coords.shape
        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result
