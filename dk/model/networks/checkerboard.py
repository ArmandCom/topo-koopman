from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from model.networks_space.spectral_norm import SpectralNorm
from utils import positional_encoding as pe
from utils.softmax_forward_warp import SoftForwardWarp
from utils import tracker_util as tut
from utils import util as ut
import math

def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega

'''My definition'''
class ObservableDecoderNetwork(nn.Module):
    def __init__(self, s, c, hidden_size, output_size, SN=False):
        super(ObservableDecoderNetwork, self).__init__()

        # self.fc = SpectralNorm(nn.Linear(c, 1))
        # self.model = nn.Sequential(
        #     SpectralNorm(nn.Linear(s, hidden_size)),
        #     nn.ReLU(),
        #     SpectralNorm(nn.Linear(hidden_size, hidden_size)),
        #     nn.ReLU(),
        #     SpectralNorm(nn.Linear(dhidden_size, hidden_size)),
        #     nn.ReLU(),
        #     SpectralNorm(nn.Linear(hidden_size, output_size +1))
        # )
        self.model = nn.Sequential(
            nn.Linear(s, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size + 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        B, N, T, c_d, g_d = x.shape
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = x.reshape(B*N*T, c_d, g_d)
        # x = self.fc(x.transpose(-2, -1)).squeeze(-1)
        x = x.sum(1)

        o = self.model(x).reshape(B, N, T, -1)
        confi = o[..., -1:]#.sigmoid()
        # out = torch.clamp(o[..., :-1], min=-1.1, max=1.1)
        # out = o[..., :-1].tanh()
        out = o[..., :-1]
        return out, confi

class Dynamics(nn.Module):
    def __init__(self, g_dim, u_dim, num_heads, with_inputs=False, with_interactions=False):
        super(Dynamics, self).__init__()

        # TODO: Doubts about the size. Different matrix for each interaction? We start by aggregating relation_states and using same matrix?
        # TODO: Maybe get all states and sum them with a matrix

        self.num_heads = num_heads - 1
        self.with_interactions = with_interactions
        self.with_inputs = with_inputs

        '''Basic blocks'''
        self.main_u_dynamics = nn.Conv2d(u_dim, g_dim, 1, bias=False)
        self.main_dynamics = nn.Conv2d(g_dim, g_dim, 1, bias=False)


        if with_inputs: #Note: Add a u_dynamics layer corresponding to each head.
            self.u_dynamics = []
            for _ in range(self.num_heads):
                self.u_dynamics.append(nn.Conv2d(u_dim, g_dim, 1, bias=False))

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
            # eig_vec = torch.eig(self.dynamics.weight.data, eigenvectors=True)[1]
            # self.dynamics.weight.data = eig_vec
            for i in range(self.num_heads):
                U, S, V = torch.svd(self.u_dynamics[i].weight.data) # TODO: Review for Conv2d, or transform data to run with linear
                self.u_dynamics[i].weight.data = torch.mm(U, V.t()) * 0.8

        #Option 1
        self.dynamics = []
        for _ in range(self.num_heads):
            K = nn.Conv2d(g_dim, g_dim, 1, bias=False)
            K.weight.data = torch.zeros_like(K.weight.data) + 0.001 # Equivalent for Conv2d
            self.dynamics.append(K)

            # if with_inputs: ...


        #Option 2

        # self.dynamics.weight.data = gaussian_init_(g, std=0.1)
        # U, S, V = torch.svd(self.dynamics.weight.data)
        # self.dynamics.weight.data = torch.mm(U, V.t()) * 0.7

        #Option 3

        # k = 5
        # S = torch.ones_like(S)
        # S[..., k:] = 0
        # self.dynamics.weight.data = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.t()) * init_scale

        '''Attention'''
        # TODO: in attention,
        #  impose kernels and 0-out the center (before softmax)
        #  for other attention heads than the ones hardcoded.

    def forward(self, g, u=None):
        g = self.main_dynamics(g) # Should we consider the surroundings?

        # for i in self.num_heads:
        #     ...

        if self.with_interactions:
            '''Unfold, Multihead attention, fold'''
            ...
        if self.with_inputs and u is not None:
            g = g + self.main_u_dynamics(u)
            for i in self.num_heads:
                # TODO: Attend both to u_embedding and observables with attn maps
                g = g + self.u_dynamics[i](u[i, ...])
        return g

class AttentiveDynamics(nn.Module):
    def __init__(self, input_res, g_dim, u_dim, pe_dim, num_heads, with_inputs=False, with_interactions=False):
        super(AttentiveDynamics, self).__init__()

        # TODO: Doubts about the size. Different matrix for each interaction? We start by aggregating relation_states and using same matrix?
        # TODO: Maybe get all states and sum them with a matrix

        self.num_heads = num_heads - 1
        self.with_interactions = with_interactions
        self.with_inputs = with_inputs

        '''Basic blocks'''
        hidden_dim = 50
        self.scale = hidden_dim ** -0.5
        self.main_key = nn.Sequential(pe.PosEncodingNeRF(in_features=2, sidelength=input_res),
                                        nn.Conv2d(pe_dim, hidden_dim, 1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 1))
        self.main_query = nn.Conv2d(g_dim, hidden_dim, 1)
        self.main_dynamics = nn.Conv2d(g_dim, g_dim, 1, bias=False)

        #Option 1
        self.dynamics = []
        for _ in range(self.num_heads):
            K = nn.Conv2d(g_dim, g_dim, 1, bias=False)
            K.weight.data = torch.zeros_like(K.weight.data) + 0.001 # Equivalent for Conv2d
            self.dynamics.append(K)

            # if with_inputs: ...


        #Option 2

        # self.dynamics.weight.data = gaussian_init_(g, std=0.1)
        # U, S, V = torch.svd(self.dynamics.weight.data)
        # self.dynamics.weight.data = torch.mm(U, V.t()) * 0.7

        #Option 3

        # k = 5
        # S = torch.ones_like(S)
        # S[..., k:] = 0
        # self.dynamics.weight.data = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.t()) * init_scale

        '''Attention'''
        # TODO: in attention,
        #  impose kernels and 0-out the center (before softmax)
        #  for other attention heads than the ones hardcoded.

    def forward(self, g, u=None, temp=1):
        # x_att = F.normalize(g, p=2, dim=-3)

        bs, C, H, W = g.shape

        k = self.main_key(None)
        q = self.main_query(g)
        k = F.normalize(k, p=2, dim=-3).reshape(bs, -1, H*W)
        q = F.normalize(q, p=2, dim=-3).reshape(bs, -1, H*W)

        v = self.main_dynamics(g) # Should we consider the surroundings?

        # qs, ks = x_att[:, 1:].flatten(0, 1).flatten(-2, -1), x_att[:, :-1].flatten(0, 1).flatten(-2, -1)
        #
        # # Feature linear transformation + stacked positional encoding
        # vs_pe = self.to_v(x.flatten(0, 1)).reshape(bs*N, T, -1, h*w)
        #
        energy = torch.einsum('bcn,bcm->bnm', q, k) * self.scale
        # TODO: zeroout diagonal?
        attn = F.softmax(energy/temp, dim=-1)

        g = torch.einsum('bcm,bnm->bcn', v, attn) # TODO: Not well, it needs to locate the value to the attended locations, not the other way around.
        # Note: Layernorm?

        return g

# Note First version of the koopman embedding
class SpatialPropagation(nn.Module):
    def __init__(self, input_res, input_dyn_dim, input_sta_dim, hidden_dim, activation_type = 'relu'):
        super(SpatialPropagation, self).__init__()

        self.input_dim = input_dyn_dim # Dynamic state, will be concatenated with the static part.
        self.input_sta_dim = input_sta_dim
        self.input_res = input_res

        if activation_type == 'relu':
            act = nn.ReLU()
        elif activation_type == 'lrelu':
            act = nn.LeakyReLU()
        elif activation_type == 'celu':
            act = nn.CELU()
        elif activation_type == 'tanh':
            act = nn.Tanh()
        else: raise NotImplementedError

        # TODO: to decide dimensions of embeddings, check CompoKoop

        '''Encoder Networks'''
        # layers_embedding = [nn.Conv2d(input_dyn_dim, hidden_dim, 3, 1, 1), act, # TODO: Add batchnorm? # Check how much does it check its surroundings
        #                     # nn.Conv2d(hidden_dim, hidden_dim, 1), act,
        #                     nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0), act,
        #                     nn.Conv2d(hidden_dim, 2, 1)]
        # self.motion = nn.Sequential(*layers_embedding)

        layers_embedding = [nn.Linear(input_dyn_dim, hidden_dim), act, # TODO: Add batchnorm? # Check how much does it check its surroundings
                            nn.Linear(hidden_dim, 2)]
        self.motion_linear = nn.Sequential(*layers_embedding)

        self.soft_fw = SoftForwardWarp(input_res)

        #TODO: only merge with input_sta_dim
        self.soft_residual = nn.Sequential(nn.Conv2d(2, hidden_dim, 1, 1, 0), act,
                                           nn.Conv2d(hidden_dim, input_sta_dim, 1, 1, 0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                # TODO: initialize random too.
                if m.out_channels == 2: m.weight.data = torch.zeros_like(m.weight.data) + 0.0001
                else: nn.init.xavier_normal_(m.weight)

        # self.round = tut.Round()

    def constrain_of(self, of):
        # of = of.tanh()
        # of = torch.cat([of[..., 0:1, :, :]*((self.input_res[0]-1)/2),
        #                 of[..., 1:2, :, :]*((self.input_res[1]-1)/2)], dim=-3)
        round_of = torch.round(of)
        cons_of = round_of - of.detach() + of
        residual_of = cons_of - of

        return cons_of, residual_of

    def forward_fw_warp(self, sta_obs_s0, dyn_obs_t1_s0, dyn_obs_t0_s0):
        # dyn_obs_t1_s0.register_hook(lambda grad: print(torch.norm(grad)))
        of = self.motion(dyn_obs_t1_s0 - dyn_obs_t0_s0) #Range: [-1, 1]
        # of.register_hook(lambda grad: print(torch.norm(grad)))
        of, residual_of = self.constrain_of(of)
        vis_dict_out = {'optical_flow': of}

        #TODO: optical flow with softmax on the differenced.
        # of = torch.zeros_like(of) + 0.01
        # dyn_obs_s1, sta_obs_s1 = dyn_obs_s0, sta_obs_s0

        # mode = 'nearest'
        # Note: uv can be rounded? Check Pytorch docs for nearest.
        # dyn_obs_s1 = F.grid_sample(dyn_obs_s0, of.permute(0, 2, 3, 1), mode=mode) # NTO * 1 * H * W
        # sta_obs_s1 = F.grid_sample(sta_obs_s0, of.permute(0, 2, 3, 1), mode=mode) # NTO * 1 * H * W
        # dyn_obs_s1 = softsplat.FunctionSoftsplat(tenInput=dyn_obs_s0, tenFlow=of, tenMetric=None, strType='summation')
        # sta_obs_s1 = softsplat.FunctionSoftsplat(tenInput=sta_obs_s0.contiguous(), tenFlow=of, tenMetric=None, strType='summation')
        # TODO: Test with of 0
        obs_s0 = torch.cat([sta_obs_s0, dyn_obs_t1_s0], dim=-3) #+ self.soft_residual(residual_of)
        # obs_s0 = obs_s0 + self.soft_residual(residual_of)

        obs_s1 = self.soft_fw(obs_s0, of, temp=0.1, constrain_of=True)
        # obs_s1.register_hook(lambda grad: print(torch.norm(grad), 'after soft_fw'))
        tut.norm_grad(obs_s1, 5) #TODO: Sure?
        sta_obs_s1, dyn_obs_t1_s1 = obs_s1[..., :-self.input_dim, :, :], obs_s1[..., -self.input_dim:, :, :]

        # Option 2
        # TODO: Unfold + SpaTX Grid_sample for each location.
        return sta_obs_s1, dyn_obs_t1_s1, vis_dict_out

    def forward(self, sta_obs_s0, dyn_obs_t1_s0, dyn_obs_t0_s0):
        # dyn_obs_t1_s0.register_hook(lambda grad: print(torch.norm(grad)))
        residual_of = None
        motion_input = dyn_obs_t0_s0 #dyn_obs_t1_s0 -
        #TODO: sum all unrolled to have 1 of per plane. Then nn.Linear() for self.motion.
        motion_input_flat = motion_input.sum(-1).sum(-1)
        of = self.motion_linear(motion_input_flat)\
            .reshape(-1, *self.input_res, 2).permute(0, 3, 1, 2) # Range: [-1, 1]
        # of.register_hook(lambda grad: print(torch.norm(grad)))
        # of, residual_of = self.constrain_of(of)
        # residual_of = self.soft_residual(residual_of)
        vis_dict_out = {'optical_flow': of}

        obs_s0 = torch.cat([sta_obs_s0, dyn_obs_t1_s0], dim=-3) #+ self.soft_residual(residual_of)
        # obs_s0 = obs_s0 + self.soft_residual(residual_of)

        obs_s1 = self.soft_fw.backward_warp(obs_s0, of, constrain_of=True, mode='bilinear')
        # obs_s1 = self.soft_fw(obs_s0, of, temp=0.1, constrain_of=True)
        # obs_s1.register_hook(lambda grad: print(torch.norm(grad), 'after soft_fw'))
        # tut.norm_grad(obs_s1, 5)
        sta_obs_s1, dyn_obs_t1_s1 = obs_s1[..., :-self.input_dim, :, :], obs_s1[..., -self.input_dim:, :, :]

        # Option 2
        return sta_obs_s1, dyn_obs_t1_s1, residual_of, vis_dict_out

class KoopmanMapping(nn.Module):
    def __init__(self, input_dim, s_sta_dim, obj_enc_dim, rel_enc_dim, hidden_dim, g_dim, u_dim, n_timesteps, spa_tf_class, with_inputs=False, with_interactions=False, activation_type = 'relu'):
        super(KoopmanMapping, self).__init__()

        self.with_interactions = with_interactions
        self.with_inputs = with_inputs
        self.n_timesteps = n_timesteps
        self.s_sta_dim = s_sta_dim
        self.soft_fw = spa_tf_class

        self.input_dim = input_dim # Dynamic state, will be concatenated with the static part.

        if activation_type == 'relu':
            act = nn.ReLU()
        elif activation_type == 'lrelu':
            act = nn.LeakyReLU()
        elif activation_type == 'celu':
            act = nn.CELU()
        elif activation_type == 'tanh':
            act = nn.Tanh()
        else: raise NotImplementedError

        # TODO: to decide dimensions of embeddings, check CompoKoop

        '''Encoder Networks'''
        layers_obj_enc = [nn.Conv2d(input_dim, obj_enc_dim, 1), act]
        self.obj_encoding = nn.Sequential(*layers_obj_enc)

        layers_embedding = [nn.Conv2d(obj_enc_dim, hidden_dim, 1), act,
                            nn.Conv2d(hidden_dim, hidden_dim, 1), act,
                            nn.Conv2d(hidden_dim, g_dim, 1)]
        self.embedding = nn.Sequential(*layers_embedding)

        # if with_inputs:
        #     u_enc_dim = ...
        #     padding_size =
        #     self.coord_to_input = Coord2Input(u_enc_dim) # Call NERF coord embedding # Last layer maps to 0!!
        #     layers_u_embedding = [nn.Conv2d(obj_enc_dim + u_enc_dim, hidden_dim, 1), act,
        #                           nn.Conv2d(hidden_dim, hidden_dim, 1), act,
        #                           nn.Conv2d(hidden_dim, u_dim, 1)]
        #     self.u_embedding = nn.Sequential(*layers_u_embedding)

        # if with_interactions:
        #     # Consider adding hardcoded sobel kernels for spatial derivatives
        #     kernel_size = 5
        #     layers_rel_embedding = [nn.Conv2d(obj_enc_dim + rel_enc_dim, hidden_dim, kernel_size, 1, kernel_size//2), act,
        #                             nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, kernel_size//2), act,
        #                             nn.Conv2d(hidden_dim, g_dim, 1)] # More depth? Why Rel_enc_dim? Do we add features? Maybe sobel.
        #     self.rel_embedding = nn.Sequential(*layers_rel_embedding)


    def forward(self, states):
        dyn_states, sta_states = states[:, self.s_sta_dim:], states[:, :self.s_sta_dim]
        obj = self.obj_encoding(dyn_states)
        obs = self.embedding(obj)
        sta_states, obs = self.soft_fw.unroll_feat_map(sta_states).flatten(0,1), \
                          self.soft_fw.unroll_feat_map(obs).flatten(0,1)
        obs_stack = torch.cat([sta_states, obs], dim=1)
        out_dict = {'sta_obs': sta_states,
                    'dyn_obs': obs,
                    'comp_obs': obs_stack}
        return out_dict

class Attention(nn.Module):
    def __init__(self, s_sta_dim, s_dyn_dim, input_res, n_timesteps, feature_dropout=0, link_dropout=0):
        super(Attention, self).__init__()

        self.pos_enc = pe.PosEncodingNeRF(in_features=2, sidelength=input_res) # TODO: Deterministic, auto-generate features.
        # in_features = self.positional_encoding.out_dim
        self.n_timesteps = n_timesteps
        self.s_sta_dim = s_sta_dim
        self.s_dyn_dim = s_dyn_dim

        input_dim = s_sta_dim + s_dyn_dim
        self.to_q = nn.Conv2d(input_dim, input_dim // 2, 1)
        self.to_k = nn.Conv2d(input_dim, input_dim // 2, 1)
        self.to_v = nn.Sequential(nn.Conv2d(input_dim, input_dim, 1), self.pos_enc)
        # TODO: use Beta from tracking by animation

        self.feature_dropout_prob, self.link_dropout_prob = feature_dropout, link_dropout
        self.feat_dropout = nn.Dropout(p=feature_dropout, inplace=False)
        self.link_dropout = nn.Dropout(p=link_dropout, inplace=False)

        self.eps = 1e-8
        self.scale = input_dim ** -0.5

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0.0)

    def zeroout_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().to(A.device)
        return A * mask

    def stoch_mat(self, A, temp=1, zero_diagonal=False, do_dropout=True):
        ''' Affinity -> Stochastic Matrix '''

        if zero_diagonal:
            A = self.zeroout_diag(A)

        if do_dropout and self.link_dropout_prob > 0:
            A[torch.rand_like(A) < self.link_dropout_prob] = -1e20

        return F.softmax(A/temp, dim=-1)

    def forward(self, x, temp=1):

        bs, N, T, ch, h, w = x.shape
        x = x.flatten(0, 1)

        if self.feature_dropout_prob > 0: # Note: Apply to q,k?
            x_att = self.feat_dropout(x)
        else:
            x_att = x

        # TODO: Check if normalization should also be done for the Value features. Maybe it doesn't matter.
        x_att = F.normalize(x_att, p=2, dim=-3) # Note: Layernorm

        # Option 1
        # qs = self.to_q(x_att[:, 1:].flatten(0, 1)).flatten(-2, -1) # [bsNT, ch, hw]
        # ks = self.to_k(x_att[:, :-1].flatten(0, 1)).flatten(-2, -1)
        # Option 2
        qs, ks = x_att[:, 1:].flatten(0, 1).flatten(-2, -1), x_att[:, :-1].flatten(0, 1).flatten(-2, -1)

        # Feature linear transformation + stacked positional encoding
        vs_pe = self.to_v(x.flatten(0, 1)).reshape(bs*N, T, -1, h*w)

        # As = self.affinity(ks, qs) # We track backwards!
        energy = torch.einsum('bcn,bcm->bnm', qs, ks).reshape(bs*N, T-1, h*w, h*w) * self.scale
        As = [self.stoch_mat(energy[:, t], temp=temp, do_dropout=True) for t in range(T-1)]

        # vs_list = torch.split(vs_pe, 1, dim=1)

        acc_state = vs_pe[:, :-self.n_timesteps+1, self.s_sta_dim:] # Note: Make sure it keeps the temporal encoding
        attn_vec = torch.stack(As, dim=1)
        for t in range(self.n_timesteps-1):
            if t + 2 < self.n_timesteps:
                curr_state, curr_attn = vs_pe[:, t+1:-self.n_timesteps+2+t, self.s_sta_dim:], attn_vec[:, t:-self.n_timesteps+2+t]
            else: curr_state, curr_attn = vs_pe[:, t+1:], attn_vec[:, t:]
            #  attn_vec has one less timestep, so the range is slightly different.

            acc_state = torch.cat([curr_state, torch.einsum('btcm,btnm->btcn', acc_state, curr_attn)], dim=2)

            # Note: IGNORE. For testing purposes: reconstruct t without (t) features
            # if t + 2 < self.n_timesteps:
            #     acc_state = torch.cat([curr_state, torch.einsum('btcm,btnm->btcn', acc_state, curr_attn)], dim=2)
            # else:
            #     acc_state = torch.einsum('btcm,btnm->btcn', acc_state, curr_attn)

        # # Note: IGNORE. For testing purposes: Simple version of the function
        # acc_state = vs_list[0]
        # for t in range(T-1):
        #
        #     if t < T-3: # Note: Test attention. If we reverse the dimensionality it works poorly. Difficult to know why
        #         acc_state = torch.cat([vs_list[t+1], torch.einsum('bcm,bnm->bcn', acc_state.squeeze(1), As[t]).unsqueeze(1)], dim=2)
        #     else:
        #         acc_state = torch.einsum('bcm,bnm->bcn', acc_state.squeeze(1), As[t]).unsqueeze(1)

        # Option: Self-attention from SAGAN
        # m_batchsize,C,width ,height = x.size()
        # proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        # proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        # energy =  torch.bmm(proj_query,proj_key) # transpose check
        # attention = self.softmax(energy) # BX (N) X (N)
        # proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        #
        # out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # out = out.view(m_batchsize,C,width,height)
        # out = self.gamma*out + x

        return acc_state

class CheckerBoard(nn.Module, ABC):
    def __init__(self, s_sta_dim, s_dyn_dim, obj_enc_dim, g_dim, r_dim, u_dim, n_timesteps, chbd_res,
                 hidden_dim=128, deriv_in_state=False, with_interactions=False, with_inputs=False):
        super(CheckerBoard, self).__init__()

        self.n_timesteps = n_timesteps
        self.deriv_in_state = deriv_in_state

        self.s_sta_dim = s_sta_dim
        self.s_dyn_dim = s_dyn_dim
        self.g_dim = g_dim
        self.r_dim = r_dim
        self.u_dim = u_dim
        self.obj_enc_dim = obj_enc_dim
        self.with_interactions = with_interactions
        self.with_inputs = with_inputs

        if deriv_in_state and n_timesteps > 2:
            first_deriv_dim = n_timesteps - 1
            sec_deriv_dim = n_timesteps - 2
        else:
            first_deriv_dim = 0
            sec_deriv_dim = 0

        # n_chan = 1
        # input_dim = state_dim * (n_timesteps + first_deriv_dim + sec_deriv_dim) #+ g_dim # g_dim added for recursive sampling

        '''Attention Network'''
        self.attn = Attention(s_sta_dim=s_sta_dim, s_dyn_dim=s_dyn_dim, input_res=chbd_res, n_timesteps=n_timesteps, feature_dropout=0.2, link_dropout=0.1)
        #(pass dropout probs as parameter)
        self.pos_enc_dim = self.attn.pos_enc.out_dim
        self.state_dim = (self.pos_enc_dim + s_dyn_dim) * n_timesteps #+ s_sta_dim

        #
        '''Koopman propagation'''
        self.temp_propagation = Dynamics(g_dim=g_dim, u_dim=u_dim, num_heads=1, with_inputs=with_inputs, with_interactions=with_interactions) # att_field_size

        self.full_propagation = AttentiveDynamics(input_res=chbd_res, g_dim=g_dim, u_dim=u_dim, pe_dim=self.pos_enc_dim,
                                                  num_heads=1, with_inputs=with_inputs, with_interactions=with_interactions) # att_field_size

        '''Spatial propagation'''
        of_hidden_dim = hidden_dim // 2
        self.spa_propagation = SpatialPropagation(input_res=chbd_res, input_dyn_dim=g_dim, input_sta_dim=s_sta_dim, hidden_dim=of_hidden_dim)
        self.soft_fw = self.spa_propagation.soft_fw

        '''Encoder Network'''
        self.mapping = KoopmanMapping(input_dim=self.state_dim, s_sta_dim=s_sta_dim, obj_enc_dim=obj_enc_dim, rel_enc_dim=None, hidden_dim=hidden_dim,
                                      g_dim=g_dim, u_dim=u_dim, n_timesteps=n_timesteps, spa_tf_class=self.soft_fw, with_inputs=with_inputs, with_interactions=with_interactions,
                                      activation_type='relu')

        # It is an inverse mapping from g_dim to something like a state dimension (or the state). And from here to u,v.
        # Note: This could be self-supervised if we decode back to the states,
        #  or we could ignore the inverse mapping and simply decide u,v
        #  depending on the observables at x,y and its surroundings.
        #  Therefore in the relational setting, the spatial propagation should have a field of view > 1.
        # Note: another option is to self-supervise with the static component of the vector.
        # Note: Should we impose any local smoothness regularizer? Sampling + KLdiv? Spectral normalization? --> wouldn't hurt

        # TODO: Set Confidence
        # TODO: tahn().abs() for confi and STN?

        # '''Decoder Network'''
        # self.inv_mapping = ObservableDecoderNetwork(g_dim, n_chan, hidden_size=nf_particle, output_size=input_dim, SN=False)
        # Note: The decoding should be independent of any positional information (except for residuals to a nearest neighbor assignment).
        #  However, for a variable appearance it should take the observables, as they contain all dynamical info.

    def feat_to_s(self, feat):
        """ state decoder """
        states = self.attn(feat)
        states = states.reshape(-1, states.shape[-2], *feat.shape[-2:]) # TODO: Create a confidence variable?
        return states

    def s_to_g(self, states):
        """ state encoder """
        # TODO: 1 chan Convolution. 3x3 / 5x5 / ... in case of interactions
        g = self.mapping(states)
        return g


    def g_to_s(self, g):
        """ state decoder """
        states, confi = self.inv_mapping(g) # TODO: Create a confidence variable?
        return states, confi

    def forward_pass(self, g_sta_s0, g_t0_s0):
        g_t1_s0 = self.temp_propagation(g_t0_s0)
        g_sta_s1, g_t1_s1, residual_of, vis_dict_of = self.spa_propagation(g_sta_s0, g_t1_s0, g_t0_s0)
        vis_dict_out = {'g(t,s)': g_t0_s0,
                        'g(t+1,s)': g_t1_s0,
                        'g(t+1,s+1)': g_t1_s1,
                        'g(sta,s+1)': g_sta_s1}
        vis_dict_out = {**vis_dict_out, **vis_dict_of}
        return g_sta_s1, g_t1_s1, residual_of, vis_dict_out

    def forward_pass_attn(self, g_sta_s0, g_t0_s0):
        g_sta_s1, g_t1_s1 = self.full_propagation(g_sta_s0, g_t0_s0)
        vis_dict_out = {'g(t,s)': g_t0_s0,
                        'g(sta,s+1)': g_sta_s1}
        return g_sta_s1, g_t1_s1, vis_dict_out

    def get_A(self):
        A = self.dynamics.dynamics.weight.data
        return A

    def get_AB(self):
        A = self.dynamics.dynamics.weight.data
        B = self.dynamics.u_dynamics.weight.data
        return A, B

    def add_noise_to_A(self, std=0.1):
        A = self.dynamics.dynamics.weight.data
        self.dynamics.dynamics.weight.data = A + torch.randn_like(A)*std / (A.shape[0])

    def limit_rank_k(self, k=None):
        if k is not None:
            U, S, V = torch.svd(self.dynamics.dynamics.weight.data)
            print('Singular Values FW: ',S)
            S[..., k:] = 0 # Review if this is right
            self.dynamics.dynamics.weight.data = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))
            #
            U, S, V = torch.svd(self.backdynamics.dynamics_back.weight.data)
            S[..., k:] = 0
            print('Singular Values BW: ',S)
            self.backdynamics.dynamics_back.weight.data = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))

    def print_SV(self):
            A = self.dynamics.dynamics.weight.data
            # A_I = torch.eye(A.shape[0]).to(A.device)
            A_I = torch.zeros_like(A)
            U, S, V = torch.svd(A + A_I)
            E = torch.eig(A + A_I)[0]
            mod_E = torch.norm(E, dim=-1)
            print('Singular Values FW:\n',str(S.data.detach().cpu().numpy()),'\nEigenvalues FW:\n',str(mod_E.data.detach().cpu().numpy()))

    def linear_forward(self, g, u=None):
        """
        :param g: B x N x D
        :return:
        """
        ''' B x N x R D '''
        B, N, T, c, dim = g.shape
        g_in = g.reshape(B*N*T, *g.shape[-2:])
        if u is not None:
            u = u.reshape(B*N*T, *u.shape[-2:])
        new_g = (self.dynamics(g_in, u)).reshape(B, N, T, *g.shape[-2:])
        # new_g.register_hook(lambda x: torch.clamp(x, min=-0, max=0))
        # new_g.register_hook(lambda x: print(x.norm()))
        return new_g

    def rollout_in_out(self, g, T, init_s, inputs=None, temp=1, logit_out=False):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        # TODO: Test function.
        B, N, _, gs = g.shape
        g_list = [g]
        u_list = [] # Note: Realize that first element of u_list corresponds to g(t) and first element of g_list to g(t+1)
        u_logit_list = []
        sel_list = []
        sel_logit_list = []
        selector_list = []
        if inputs is None:
            in_t = None

        out_states = [self.inv_mapping(g)]
        in_states = [torch.cat([init_s, out_states[-1]], dim=-2)]
        # ut.print_var_shape(out_states[0], 's in rollout')

        for t in range(T):
            in_full_state, _ = self.get_full_state_hankel(in_states[-1].reshape(B*N, self.n_timesteps, -1),
                                                          T=self.n_timesteps)
            in_full_state = in_full_state.reshape(B, N, 1, -1)
            # TODO: check correspondence between g and in_states
            if inputs is not None:
                in_t = inputs[..., t, :]
            g_stack, u_tp, sel_tp, selector = self.mapping(in_full_state, temp=temp, collision_margin=self.collision_margin, input=in_t)
            u, u_logits = u_tp
            sel, sel_logits = sel_tp
            selector_list.append(selector)
            if u is None:
                logit_out = True

            if len(u_list) == 0:
                u_list.append(u)
                u_logit_list.append(u_logits)
                # input_list.append(inputs[:, None, :])
            if self.with_interactions and len(sel_list)==0:
                sel_list.append(sel)
                sel_logit_list.append(sel_logits)

            # Propagate with A
            # g_out = self.linear_forward(g_stack.transpose(-2, -1).flatten(start_dim=-2))
            g_out = self.rollout_1step(g_stack)

            out_states.append(self.inv_mapping(g_out))
            g_list.append(g_out)
            in_states.append(torch.cat([in_states[-1][..., 1:, :], out_states[-1]], dim=-2))
            u_list.append(u)
            u_logit_list.append(u_logits)
            # input_list.append(inputs[:, None, :])

            if self.with_interactions:
                    sel_list.append(sel)
                    sel_logit_list.append(sel_logits)

        if self.with_interactions:
            sel_list = torch.cat(sel_list, 2)
            sel_logit_list = torch.cat(sel_logit_list, 2) if logit_out else None
        else:
            sel_list, sel_logit_list = None, None
        u_list = torch.cat(u_list, 2) if u_list[0] is not None else None
        u_logit = torch.cat(u_logit_list, 2) if logit_out and u_logit_list[0] is not None else None

        selectors = torch.cat(selector_list, 2)

        return torch.cat(out_states, 2), torch.cat(g_list, 2), (u_list,  u_logit), (sel_list, sel_logit_list), selectors

    def rollout(self, g, T, inputs=None, temp=1, logit_out=False, with_u = True):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        # TODO: Test function.
        B, N, _, gs, ns = g.shape
        g_list = []

        u_list = [] # Note: Realize that first element of u_list corresponds to g(t) and first element of g_list to g(t+1)
        u_logit_list = []
        sel_list = []
        sel_logit_list = []
        selector_list = []
        if inputs is None:
            in_t = None

        out_s, out_c = self.inv_mapping(g)
        out_states = [out_s]
        out_confis = [out_c]

        for t in range(T):
            if inputs is not None:
                in_t = inputs[..., t, :]
            g_in, u_in, u_tp, sel_tp, selector = self.mapping(out_states[t], temp=temp, collision_margin=self.collision_margin, inputs=in_t, with_u = with_u)
            u, u_logits = u_tp
            sel, sel_logits = sel_tp
            selector_list.append(selector)

            if len(u_list) == 0:
                u_list.append(u)
                u_logit_list.append(u_logits)
                # input_list.append(inputs[:, None, :])
            if self.with_interactions and len(sel_list)==0:
                sel_list.append(sel)
                sel_logit_list.append(sel_logits)

            if t == 0:
                g_list.append(g_in)
            # Propagate with A
            g_out = self.rollout_1step(g_in, u_in)
            g_list.append(g_out)
            out_s, out_c = self.inv_mapping(g_out)
            out_states.append(out_s)
            out_confis.append(out_c)

        gs = torch.cat(g_list, 2)
        out_state = torch.cat(out_states, 2)
        out_confi = torch.cat(out_confis, 2)
        return out_state, out_confi, gs, (None,  None), (None, None), None

    def rollout_B(self, g, T, inputs=None, temp=1, logit_out=False, with_u = True):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        # TODO: Test function.
        B, N, _, gs, ns = g.shape
        g_list = []

        u_list = [] # Note: Realize that first element of u_list corresponds to g(t) and first element of g_list to g(t+1)
        u_logit_list = []
        sel_list = []
        sel_logit_list = []
        selector_list = []
        if inputs is None:
            in_t = None

        out_s, out_c = self.inv_mapping(g)
        out_states = [out_s]
        out_confis = [out_c]

        g_in = g
        for t in range(T):
            if inputs is not None:
                in_t = inputs[..., t, :]
            _, u_in, u_tp, sel_tp, selector = self.mapping(out_states[t], temp=temp, collision_margin=self.collision_margin, inputs=in_t, with_u = with_u)
            u, u_logits = u_tp
            sel, sel_logits = sel_tp
            selector_list.append(selector)

            if len(u_list) == 0:
                u_list.append(u)
                u_logit_list.append(u_logits)
                # input_list.append(inputs[:, None, :])
            if self.with_interactions and len(sel_list)==0:
                sel_list.append(sel)
                sel_logit_list.append(sel_logits)

            if t == 0:
                g_list.append(g_in)
            # Propagate with A
            g_in = self.rollout_1step(g_in, u_in)
            g_list.append(g_in)
            out_s, out_c = self.inv_mapping(g_in)
            out_states.append(out_s)
            out_confis.append(out_c)

        gs = torch.cat(g_list, 2)
        out_state = torch.cat(out_states, 2)
        out_confi = torch.cat(out_confis, 2)
        return out_state, out_confi, gs, (None,  None), (None, None), None

    def rollout_1step(self, g_stack, u = None):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        if len(g_stack.shape) == 5:
            B, N, T, c, gs = g_stack.shape
            g_in = g_stack
        else:
            raise NotImplementedError

        g_out = self.linear_forward(g_in, u).reshape(B, N, T, c, gs) #+ g_in

        return g_out

    # def get_full_state_hankel(self, x, T):
    #     '''
    #     :param x: features or observations
    #     :param T: number of time-steps before concatenation
    #     :return: Columns of a hankel matrix with self.n_timesteps rows.
    #     '''
    #     if self.n_timesteps < 2:
    #         return x, T
    #     new_T = T - self.n_timesteps + 1
    #
    #     x = x.reshape(-1, T, *x.shape[2:])
    #     new_x = []
    #     for t in range(new_T):
    #         new_x.append(torch.stack([x[:, t + (self.n_timesteps-idx-1)]
    #                                   for idx in range(self.n_timesteps)], dim=-1))
    #     # torch.cat([ torch.zeros_like( , x[:,0,0:1]) + self.t_grid[idx]], dim=-1)
    #     new_x = torch.stack(new_x, dim=1)
    #
    #     if self.deriv_in_state and self.n_timesteps > 2:
    #         d_x = new_x[..., 1:] - new_x[..., :-1]
    #         dd_x = d_x[..., 1:] - d_x[..., :-1]
    #         new_x = torch.cat([new_x, d_x, dd_x], dim=-1)
    #
    #     # print(new_x[0, 0, :4, :3], new_x.flatten(start_dim=-2)[0, 0, :12])
    #     # exit() out: [x1, x2, x3, y1, y2, y3]
    #     return new_x.flatten(start_dim=-2), new_T

    @staticmethod
    def batch_pinv(x, I_factor):
        """
        :param x: B x N x D (N > D)
        :param I_factor:
        :return:
        """

        B, N, D = x.size()

        if N < D:
            x = torch.transpose(x, 1, 2)
            N, D = D, N
            trans = True
        else:
            trans = False

        x_t = torch.transpose(x, 1, 2)

        use_gpu = torch.cuda.is_available()
        I = torch.eye(D)[None, :, :].repeat(B, 1, 1)
        if use_gpu:
            I = I.to(x.device)

        x_pinv = torch.bmm(
            torch.inverse(torch.bmm(x_t, x) + I_factor * I),
            x_t
        )

        if trans:
            x_pinv = torch.transpose(x_pinv, 1, 2)

        return x_pinv

    def accumulate_obs(self, obs_ini, increments):
        T = increments.shape[1]
        Tini = obs_ini.shape[1]
        assert Tini == 1

        obs = [obs_ini[:, 0]]
        for t in range(T):
            obs.append(obs[t] + increments[:, t])
        obs = torch.stack(obs, dim=1)
        return obs

# class encoderNet(nn.Module):
#     def __init__(self, N, b, ALPHA = 1):
#         super(encoderNet, self).__init__()
#         self.N = N
#         self.tanh = nn.Tanh()
#         self.b = b
#
#         self.fc1 = nn.Linear(self.N, 16*ALPHA)
#         self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
#         self.fc3 = nn.Linear(16*ALPHA, b)
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x):
#         x = x.view(-1, 1, self.N)
#         x = self.tanh(self.fc1(x))
#         x = self.tanh(self.fc2(x))
#         x = self.fc3(x).view(-1, self.b)
#
#         return x
#
# class decoderNet(nn.Module):
#     def __init__(self, N, b, ALPHA = 1, SN=False):
#         super(decoderNet, self).__init__()
#
#         self.N = N
#         self.b = b
#
#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()
#
#         if SN:
#             self.fc1 = SpectralNorm(nn.Linear(b, 16*ALPHA))
#             self.fc2 = SpectralNorm(nn.Linear(16*ALPHA, 16*ALPHA))
#             self.fc3 = SpectralNorm(nn.Linear(16*ALPHA, N))
#         else:
#             self.fc1 = nn.Linear(b, 16*ALPHA)
#             self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
#             self.fc3 = nn.Linear(16*ALPHA, N)
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x):
#         x = x.view(-1, 1, self.b)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = x.view(-1, self.N)
#         return x
