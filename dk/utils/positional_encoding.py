import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import math

# Option 1
# Positional encoding (section 5.1) NERF
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input' : True,
        'input_dims' : 3,
        'max_freq_log2' : multires-1,
        'num_freqs' : multires,
        'log_sampling' : True,
        'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

    # inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    # embedded = embed_fn(inputs_flat)


# Option 2
# TODO: Simple version with x,y
class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features
        self.sidelength = sidelength

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

        # self.register_buffer('coords', self.get_mgrid(sidelength))
        self.register_buffer('pos_enc', self.generate_pe(self.get_mgrid(sidelength, in_features)))

    def get_mgrid(self, sidelen, dim=2):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
        if isinstance(sidelen, int):
            sidelen = dim * (sidelen,)

        if dim == 2:
            pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
            pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
            pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
        elif dim == 3:
            pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
            pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
            pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
            pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        else:
            raise NotImplementedError('Not implemented for dim=%d' % dim)

        pixel_coords -= 0.5
        pixel_coords *= 2.
        pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
        return pixel_coords

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def generate_pe(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)
        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        coords_pos_enc = coords_pos_enc.reshape(-1, self.out_dim).transpose(1, 0) # .reshape(*self.sidelength, -1, self.out_dim)
        return coords_pos_enc

    def forward(self, x):
        pos_enc = self.pos_enc.reshape(1, self.out_dim, *self.sidelength).repeat_interleave(x.shape[0], dim=0)
        if x is None:
            x = pos_enc
        else:
            x = torch.cat([x, pos_enc], dim=-3)
        return x


class PosEncodingSimple(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features
        self.sidelength = sidelength
        self.out_dim = in_features #+ 2 * in_features * self.num_frequencies

        # self.register_buffer('coords', self.get_mgrid(sidelength))
        self.register_buffer('pos_enc', self.generate_pe(self.get_mgrid(sidelength, in_features)))

    def get_mgrid(self, sidelen, dim=2):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
        if isinstance(sidelen, int):
            sidelen = dim * (sidelen,)

        if dim == 2:
            pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
            pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
            pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
        elif dim == 3:
            pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
            pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
            pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
            pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        else:
            raise NotImplementedError('Not implemented for dim=%d' % dim)

        pixel_coords -= 0.5
        pixel_coords *= 2.
        pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
        return pixel_coordsd

    def generate_pe(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)
        coords_pos_enc = coords
        coords_pos_enc = coords_pos_enc.reshape(-1, self.out_dim).transpose(1, 0) # .reshape(*self.sidelength, -1, self.out_dim)
        return coords_pos_enc

    def forward(self, x):
        pos_enc = self.pos_enc.reshape(1, self.out_dim, *self.sidelength).repeat_interleave(x.shape[0], dim=0)
        if x is None:
            x = pos_enc
        else:
            x = torch.cat([x, pos_enc], dim=-3)
        return x
