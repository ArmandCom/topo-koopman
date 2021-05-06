import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import positional_encoding as pe

# TODO: Spectralnorm?
class ImageDecoder(nn.Module):
  def __init__(self, input_dim, chbd_res, map_scale, out_chan, SN=False, PE=False):
    super(ImageDecoder, self).__init__()

    # layers = \
    # [nn.Conv2d(state_dim, 128 * 2 * 2, 1), # * 2
    # nn.PixelShuffle(2),
    # nn.CELU(),
    # nn.BatchNorm2d(128),
    # nn.Conv2d(128, 128, 3, 1, 1),
    # nn.CELU(),
    # nn.BatchNorm2d(128),
    #
    # nn.Conv2d(128, 128 * 2 * 2, 1), # * 2
    # nn.PixelShuffle(2),
    # nn.CELU(),
    # nn.BatchNorm2d(128),
    # nn.Conv2d(128, 128, 3, 1, 1),
    # nn.CELU(),
    # nn.BatchNorm2d(128)]
    # # --
    # if map_scale >= 8:
    #   last_dim_8 = 64
    #   layers.extend([ nn.Conv2d(128, last_dim_8 * 2 * 2, 1), # * 2
    #                   nn.PixelShuffle(2),
    #                   nn.CELU(),
    #                   nn.BatchNorm2d(last_dim_8),
    #                   nn.Conv2d(last_dim_8, last_dim_8, 3, 1, 1),
    #                   nn.CELU(),
    #                   nn.BatchNorm2d(last_dim_8)])
    #   last_dim = last_dim_8
    #
    # # --
    # if map_scale >= 16:
    #   last_dim_16 = 32
    #   layers.extend([ nn.Conv2d(last_dim_8, last_dim_16 * 2 * 2, 1), # 16, 16
    #                   nn.PixelShuffle(2),
    #                   nn.CELU(),
    #                   nn.BatchNorm2d(last_dim_16),
    #                   nn.Conv2d(last_dim_16, last_dim_16, 3, 1, 1),
    #                   nn.CELU(),
    #                   nn.BatchNorm2d(last_dim_16)])
    #   last_dim = last_dim_16
    # # --
    # layers.extend([ nn.Conv2d(last_dim, 16, 3, 1, 1),
    #                 nn.CELU(),
    #                 nn.BatchNorm2d(16),
    #                 nn.Conv2d(16, out_chan, 1)]) # output channels. If your input is RGB, out_ch=3

    us = nn.Upsample(scale_factor=2, mode='bilinear') # TODO: try Trilinear
    layers = []
    if PE:
        self.pos_enc = pe.PosEncodingNeRF(in_features=2, sidelength=chbd_res) # TODO: Deterministic, auto-generate features.
        self.pos_enc_dim = self.pos_enc.out_dim
        layers.append(self.pos_enc)
    else: self.pos_enc_dim = 0
    layers.extend(
        [nn.Conv2d(input_dim + self.pos_enc_dim, 128, 3, 1, 1), # * 2
         nn.BatchNorm2d(128),
         nn.CELU(),
         us,
         nn.Conv2d(128, 128, 3, 1, 1),
         nn.BatchNorm2d(128),
         nn.CELU(),

         nn.Conv2d(128, 128, 3, 1, 1), # * 2
         nn.BatchNorm2d(128),
         nn.CELU(),
         us,
         nn.Conv2d(128, 128, 3, 1, 1),
         nn.BatchNorm2d(128),
         nn.CELU()])
    # --
    if map_scale >= 8:
        last_dim_8 = 64
        layers.extend([ nn.Conv2d(128, last_dim_8, 3, 1, 1), # * 2
                        nn.BatchNorm2d(last_dim_8),
                        nn.CELU(),
                        us,
                        nn.Conv2d(last_dim_8, last_dim_8, 3, 1, 1),
                        nn.BatchNorm2d(last_dim_8),
                        nn.CELU()])
        last_dim = last_dim_8

    # --
    if map_scale >= 16:
        last_dim_16 = 32
        layers.extend([ nn.Conv2d(last_dim_8, last_dim_16, 3, 1, 1), # * 2
                        nn.BatchNorm2d(last_dim_16),
                        nn.CELU(),
                        us,
                        nn.Conv2d(last_dim_16, last_dim_16, 3, 1, 1),
                        nn.BatchNorm2d(last_dim_16),
                        nn.CELU()])
        last_dim = last_dim_16
    if map_scale >= 32:
        last_dim_32 = 16
        layers.extend([ nn.Conv2d(last_dim_16, last_dim_32, 3, 1, 1), # * 2
                        nn.BatchNorm2d(last_dim_32),
                        nn.CELU(),
                        us,
                        nn.Conv2d(last_dim_32, last_dim_32, 3, 1, 1),
                        nn.BatchNorm2d(last_dim_32),
                        nn.CELU()])
        last_dim = last_dim_32
    # --

    if SN:
        layers.extend([ nn.utils.spectral_norm(nn.Conv2d(last_dim, 16, 3, 1, 1)),
                        nn.CELU(),
                        nn.BatchNorm2d(16),
                        nn.utils.spectral_norm(nn.Conv2d(16, out_chan, 1))]) # output channels. If your input is RGB, out_ch=3
    else:
        layers.extend([ nn.Conv2d(last_dim, 16, 3, 1, 1),
                        nn.CELU(),
                        nn.BatchNorm2d(16),
                        nn.Conv2d(16, out_chan, 1)]) # output channels. If your input is RGB, out_ch=3

    self.to_x = nn.Sequential(*layers)

  def forward(self, input):
    x = torch.sigmoid(self.to_x(input)) # output will be [0, 1]
    return x

