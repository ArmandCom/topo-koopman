import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from base import BaseModel
from utils import util as ut
from model.networks import ImageDecoder
from model.networks.checkerboard import CheckerBoard
# from torch.distributions import Normal, kl_divergence
from utils.util import linear_annealing
from utils import model_util as mut
import random
import matplotlib.pyplot as plt

def sum_objects(objs, n_obj, mask=False):
    if mask:
        objs = objs * objs.sum(-1, keepdims=True).sum(-2, keepdims=True)[..., -1:, :, :].tanh().abs()
    return objs.reshape(-1, n_obj, *objs.shape[1:]).sum(1)

class TopoKoopModel(BaseModel):
    def __init__(self, in_chan, s_sta_dim, s_dyn_dim, obj_enc_dim, g_dim, r_dim, u_dim, hidden_dim, image_size, attn_res, chbd_res, n_timesteps=1,
                 with_interactions=False, with_inputs=False):
        super().__init__()

        # Nomenclature rules: _dim: dimensionality, _size: [h, w], _chan: channels, (sta) static, (dyn) dynamic, _res: resolution [h', w']
        '''Dimensions'''
        self.s_sta_dim = s_sta_dim # Static part of the state
        self.s_dyn_dim = s_dyn_dim # Dynamic part of the state
        self.g_dim = g_dim # Observables
        self.u_dim = u_dim # Input (encoding)
        self.r_dim = r_dim # Relational (encoding) --> Might be computed by convolutional encoder
        self.hidden_dim = hidden_dim

        '''Resolutions'''
        out_chan = in_chan # Input and output channels
        self.in_chan = in_chan
        self.image_size = image_size
        self.attn_res = attn_res # Attention map resolution
        self.chbd_res = chbd_res # Checkerboard resolution

        '''State details'''
        self.n_timesteps = n_timesteps

        '''Flags'''
        self.deriv_in_state = False
        self.with_interactions = with_interactions
        self.with_inputs = with_inputs

        '''Networks'''
        # self.img_encoder = ImageEncoder(image_size, in_chan, s_sta_dim, s_dyn_dim, image_size, n_timesteps=n_timesteps)
        self.cnn_backbone = mut.make_encoder(model_type='imagenet18', remove_layers=['layer4']) #TODO: check 3d to 2d
        dummy_out = self.infer_dims_bb()
        self.mlp_head = self.make_head(depth=1, downsample=2)
        self.infer_dims_chbd(dummy_out)
        self.num_objects = self.chbd_res[0]*self.chbd_res[1]
        print("Number of objects: {}".format(self.num_objects) )
        self.chbd = CheckerBoard(s_sta_dim, s_dyn_dim, obj_enc_dim, g_dim, r_dim, u_dim, n_timesteps, chbd_res=self.chbd_res, deriv_in_state=self.deriv_in_state,
                                        hidden_dim=hidden_dim, with_interactions=with_interactions, with_inputs=with_inputs) # Intermediate dimensions. Note: Chbd_res == bb_res?
        self.state_dim = self.chbd.state_dim # Note: -1 is for this particular test
        self.cnn_decoder = ImageDecoder(input_dim = s_sta_dim, chbd_res=self.chbd_res, map_scale=self.chbd_map_scale, out_chan=out_chan, SN=False, PE=True) # Decode only with static features

        '''Counters'''
        self.print_epoch_count = 1

    def infer_dims_bb(self):
        # in_sz = 256
        dummy = torch.zeros(1, 1, 1, 3, *self.image_size).to(next(self.cnn_backbone.parameters()).device)
        dummy_out = self.cnn_backbone(dummy)
        self.enc_hid_dim = dummy_out.shape[-3]
        self.bb_res = dummy_out.shape[-2:]
        self.map_scale = self.image_size[0] // self.bb_res[0]
        return dummy_out

    def infer_dims_chbd(self, dummy_in):
        dummy_out = self.mlp_head(dummy_in)
        self.chbd_res = dummy_out.shape[-2:]
        self.chbd_map_scale = self.image_size[0] // self.chbd_res[0]

    def make_head(self, depth=1, downsample=4):
        head = []
        assert downsample in [1,2,4]
        if depth >= 0:
            dims = [self.enc_hid_dim] + [self.enc_hid_dim] * depth + [self.s_sta_dim + self.s_dyn_dim]
            # first = True
            for d1, d2 in zip(dims, dims[1:]):
                if downsample > 1:
                    h = nn.Conv2d(d1, d2, kernel_size=4, stride=2, padding=1)
                    downsample /= 2
                else:
                    h = nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)
                head += [h, nn.ReLU()]
            head = head[:-1]
        return nn.Sequential(*head)

    def print_each_epoch(self, epoch):
        if self.print_epoch_count < epoch:
            self.print_epoch_count = epoch
        elif self.print_epoch_count == epoch:
            # self.koopman.print_SV()
            self.print_epoch_count += 1

    def forward(self, input, epoch_iter, test=False):
        bs, T, chN, h, w = input.shape
        N, ch = chN//self.in_chan, self.in_chan
        T_rec = T - self.n_timesteps + 1
        T_pred_1 = T_rec - 1

        # TODO: function print_at_epoch
        output = {}
        input = input.reshape(bs, T, N, ch, h, w).transpose(1, 2)

        # if epoch_iter[0] is not -1:
        #     temp = linear_annealing(input.device, epoch_iter[1], start_step=4000, end_step=15000, start_value=1, end_value=0.1)
        # else: temp = 1.

        # self.print_each_epoch(epoch_iter[0])
        x = self.cnn_backbone(input)
        x = self.mlp_head(x)

        s = self.chbd.feat_to_s(x.reshape(bs, N, T, self.s_sta_dim + self.s_dyn_dim, *self.chbd_res))
        x_embed_dict = self.chbd.s_to_g(s) # x_dict: sta_obs, dyn_obs, comp_obs
        g_sta_pred, g_dyn_pred, residual_of, vis_g_fw_dict = self.chbd.forward_pass(x_embed_dict['sta_obs'], x_embed_dict['dyn_obs'])
        g_pred = torch.cat([g_sta_pred, g_dyn_pred], dim=-3)


        output['optical_flow'] = vis_g_fw_dict['optical_flow']
        output['obs_for_rec'], output['obs_for_pred'] = x_embed_dict['comp_obs'], g_pred

        cases = []
        cases.append({
            'name': 'rec',
            'T': T_rec,
            # 'obs': x_embed_dict['comp_obs']
            'obs': x_embed_dict['sta_obs'],
            'residual': None
        })
        cases.append({
            'name': 'pred_1',
            'T': T_pred_1,
            # 'obs': g_pred
            'obs': g_sta_pred,
            'residual': residual_of
        })
        # TODO: Decode each layer and sum all (weighted by a presence variable [0, 1])
        for case in cases:
            case_name, case_obs, case_T = case['name'], case['obs'], case['T']
            # obs_name = case_name + '_obs'
            # output[obs_name] = case_obs
            case_obs = sum_objects(case_obs, self.num_objects, mask=True)
            out = self.cnn_decoder(case_obs)#.sigmoid()
            # out = sum_objects(out, self.num_objects, mask=True)
            out = out.reshape(bs, -1, chN, h, w)[:, :case_T]
            # out = out.reshape(bs, N, self.num_objects, -1, ch, h, w).sum(2).transpose(1, 2).reshape(bs, -1, chN, h, w)[:, :case_T] #TODO: presence variable mask
            output[case_name] = out
        return output