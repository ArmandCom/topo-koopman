import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from base import BaseModel
# from model.networks import ImageEncoder, SBImageDecoder, SimpleSBImageDecoder, \
#     KoopmanOperators, AttImageEncoder, ImageDecoder, \
#     deconvSpatialDecoder, linearSpatialDecoder, LinearEncoder, ObjectAttention
from model.networks import KoopmanOperators
# from model.networks_slot_attention import ImageEncoder, ImageDecoder
# from model.networks_self_attention import ImageEncoder, ImageDecoder
from model.networks_dyn_attention import ImageEncoder, ImageDecoder

# from model.networks_cswm.modules import TransitionGNN, EncoderCNNLarge, EncoderCNNMedium, EncoderCNNSmall, EncoderMLP, DecoderCNNLarge, DecoderCNNMedium, DecoderCNNSmall, DecoderMLP
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import random
import matplotlib.pyplot as plt

''' A main objective of the model is to separate the dimensions to be modeled/propagated by koopman operator
    from the constant dimensions - elements that remain constant - which should be the majority of info. 
    This way we only need to control a reduced amount of dimensions - and we can make them variable.
    The constant factors can also work as a padding for the dimensionality to remain constant. '''

def _get_flat(x, keep_dim=False):
    if keep_dim:
        return x.reshape(torch.Size([1, x.size(0) * x.size(1)]) + x.size()[2:])
    return x.reshape(torch.Size([x.size(0) * x.size(1)]) + x.size()[2:])

# If we want it to be variational:
def _sample_latent_simple(mu, logvar, n_samples=1):
    std = torch.exp(0.5 * logvar)
    # std = torch.stack([std]*n_samples, dim=-1)
    eps = torch.randn_like(std)
    sample = mu + eps * std
    return sample

# def _clamp_diagonal(A, min, max):
#     # Note: this doesn't help. We need spectral normalization.
#     eye = torch.zeros_like(A)
#     ids = torch.arange(0, A.shape[-1])
#     eye[..., ids, ids] = 1
#     return A*(1-eye) + torch.clamp(A * eye, min, max)

# TODO:
#  0: Calculate As and Bs in both directions (given the observations). Forward * Backward should be the identity.
#  0: Curriculum Learning by increasing T progressively.
#  1: Invert or Sample randomly u and contrastive loss.
#  2: Treat n_timesteps with conv_nn? Or does it make sense to mix f1(t) and f2(t-1)?
#  3: Eigenvalues and Eigenvector. Fix one get the other.
#  4: Observation selector using Gumbel (should apply mask emissor and receptor
#       and it should be the same across time). Compositional koopman
#  5: Block diagonal l1(max(off_diag)) loss. Try without specifying objects.

class RecKoopmanModel(BaseModel):
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim, u_dim,
                 n_objects, I_factor=10, n_blocks=1, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64]):
        super().__init__()
        out_channels = 1
        n_layers = int(np.log2(image_size[0])) - 1

        self.u_dim = u_dim

        # Set state dim with config, depending on how many time-steps we want to take into account
        self.image_size = image_size
        self.n_timesteps = n_timesteps
        self.state_dim = feat_dim
        self.I_factor = I_factor
        self.psteps = psteps
        self.g_dim = g_dim
        self.n_objects = n_objects

        # self.softmax = nn.Softmax(dim=-1)
        # self.sigmoid = nn.Sigmoid()

        # self.initial_conditions = nn.Sequential(nn.Linear(feat_dim * n_timesteps * 2, feat_dim * n_timesteps),
        #                                         nn.ReLU(),
        #                                         nn.Linear(feat_dim * n_timesteps, g_dim * 2))

        self.content = None

        feat_dyn_dim = feat_dim // 6
        self.feat_dyn_dim = feat_dyn_dim
        self.feat_cte_dim = feat_dim - feat_dyn_dim

        self.n_iters = 1
        self.ini_alpha = 1
        # Note:
        #  - I leave it to 0 now. If it increases too fast, the gradients might be affected
        self.incr_alpha = 0.1

        # Note:
        #  - May be substituted by 0s.
        #  - This noise has lower variance than prior. Otherwise it would sample regular features.
        self.f_cte_ini_std = 0.0

        # self.rnn_f_cte = nn.LSTM(feat_dim - feat_dyn_dim, feat_dim - feat_dyn_dim, 1, bias=False, batch_first=True)
        # self.rnn_f_cte = nn.GRU(feat_dim - feat_dyn_dim, feat_dim - feat_dyn_dim, 1, bias=False, batch_first=True)

        self.linear_f_cte_mu = nn.Linear(self.feat_cte_dim, self.feat_cte_dim)
        self.linear_f_cte_logvar = nn.Linear(self.feat_cte_dim, self.feat_cte_dim)
        self.linear_f_dyn_mu = nn.Linear(feat_dyn_dim, feat_dyn_dim)
        self.linear_f_dyn_logvar = nn.Linear(feat_dyn_dim, feat_dyn_dim)

        self.image_encoder = ImageEncoder(in_channels, self.feat_cte_dim, self.feat_dyn_dim, n_objects, ngf, n_layers)  # feat_dim * 2 if sample here
        self.image_decoder = ImageDecoder(feat_dim, out_channels, ngf, n_layers)
        self.koopman = KoopmanOperators(feat_dyn_dim, nf_particle, nf_effect, g_dim, u_dim, n_timesteps, n_blocks)

    def _get_full_state(self, x, T):

        if self.n_timesteps < 2:
            return x, T
        new_T = T - self.n_timesteps + 1
        x = x.reshape(-1, T, *x.shape[1:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.cat([x[:, t + idx]
                         for idx in range(self.n_timesteps)], dim=-1))
        # torch.cat([ torch.zeros_like( , x[:,0,0:1]) + self.t_grid[idx]], dim=-1)
        new_x = torch.stack(new_x, dim=1)
        return new_x.reshape(-1, new_x.shape[-1]), new_T

    def _get_full_state_hankel(self, x, T):
        '''
        :param x: features or observations
        :param T: number of time-steps before concatenation
        :return: Columns of a hankel matrix with self.n_timesteps rows.
        '''
        if self.n_timesteps < 2:
            return x, T
        new_T = T - self.n_timesteps + 1

        x = x.reshape(-1, T, *x.shape[2:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.stack([x[:, t + idx]
                                    for idx in range(self.n_timesteps)], dim=-1))
        # torch.cat([ torch.zeros_like( , x[:,0,0:1]) + self.t_grid[idx]], dim=-1)
        new_x = torch.stack(new_x, dim=1)

        return new_x.reshape(-1, new_T, new_x.shape[-2] * new_x.shape[-1]), new_T


    def forward(self, input, epoch = 1):
        bs, T, ch, h, w = input.shape
        # Percentage of output
        free_pred = T//4

        # Backbone deterministic features
        f_bb = self.image_encoder(input, block='backbone')

        # Dynamic features
        T_inp = T
        f_dyn = self.image_encoder(f_bb[:, :T_inp], block='dyn')
        f_dyn = f_dyn.reshape(-1, f_dyn.shape[-1])

        # Sample dynamic features
        f_mu_dyn, f_logvar_dyn = self.linear_f_dyn_mu(f_dyn), \
                                 self.linear_f_dyn_logvar(f_dyn)
        f_dyn = _sample_latent_simple(f_mu_dyn, f_logvar_dyn)
        f_dyn = f_dyn.reshape(bs * self.n_objects, T_inp, *f_dyn.shape[1:])

        # Get delayed dynamic features
        f_dyn_s, T_inp = self._get_full_state_hankel(f_dyn, T_inp)

        # Get inputs from delayed dynamic features
        # Note:
        #  - U might depend also in features from the scene. Residual features (slot n+1 / Background)
        #  - Temperature increase might not be necessary
        #  - Gumbel softmax might help
        u, u_dist = self.koopman.to_u(f_dyn_s, temp=self.ini_alpha + epoch * self.incr_alpha)

        # Get observations from delayed dynamic features
        g = self.koopman.to_g(f_dyn_s.reshape(bs * self.n_objects * T_inp, -1), self.psteps)
        g = g.reshape(bs * self.n_objects, T_inp, *g.shape[1:])

        # Get shifted observations for sys ID
        randperm = torch.arange(g.shape[0]) # No permutation
        # randperm = torch.randperm(g.shape[0]) # Random permutation
        if free_pred > 0:
            G_tilde = g[randperm, :-1-free_pred, None]
            H_tilde = g[randperm, 1:-free_pred, None]
        else:
            G_tilde = g[randperm, :-1, None]
            H_tilde = g[randperm, 1:, None]

        # Sys ID
        A, B, A_inv, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, U=u[randperm, :T_inp-free_pred-1], I_factor=self.I_factor) # Try not permuting U when inp is permutted

        # Rollout from start_step onwards.
        start_step = 2 # g and u must be aligned!!
        #TODO: Try one more step for any of the inputs and check if it blows up
        G_for_pred = self.koopman.simulate(T=T_inp-start_step-1, g=g[:,start_step], u=u[:,start_step:], A=A, B=B)
        g_for_koop= G_for_pred

        rec = {"obs": g,
               "backbone_features": f_bb[:, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
               "T": T_inp,
               "name": "rec"}
        pred = {"obs": G_for_pred,
                "backbone_features": f_bb[:, -G_for_pred.shape[1]:],
                "T": G_for_pred.shape[1],
                "name": "pred"}
        outs = {}
        cte_distr = {}
        # TODO: I'm here. Check if the indices for f_bb and supervision are correct.
        # Recover partial shape with decoded dynamical features. Iterate with new estimates of the appearance.
        # Note: This process could be iterative.
        for idx, case in enumerate([rec, pred]):

            case_name = case["name"]

            # get back dynamic features
            # if case_name == "rec":
            #     f_dyn_tmp = f_dyn[:, -T_inp:].reshape(-1, *f_dyn.shape[2:])
            # else:
            f_dyn_tmp = self.koopman.to_s(gcodes=_get_flat(case["obs"]),
                                          psteps=self.psteps)

            # Initial noisy (low variance) constant vector.
            # Note:
            #  - Different realization for each time-step.
            #  - Is it a Variable?
            f_cte_ini = torch.randn(bs * self.n_objects * case["T"], self.feat_cte_dim).to(f_dyn_tmp.device) * self.f_cte_ini_std

            # Get full feature vector
            f = torch.cat([ f_dyn_tmp,
                            f_cte_ini], dim=-1)

            # Get coarse features from which obtain queries and/or decode
            f_coarse = self.image_decoder(f, block = 'coarse')
            f_coarse = f_coarse.reshape(bs, self.n_objects, case["T"], *f_coarse.shape[1:])

            for _ in range(self.n_iters):

                # Get constant feature vector through attention
                f_cte = self.image_encoder(case["backbone_features"], f_coarse, block='cte')

                # Sample cte features
                f_cte = f_cte.mean(2)[:, :, None].repeat(1, 1, case["T"], 1)\
                    .reshape(bs * self.n_objects * case["T"], self.feat_cte_dim) # Temporal average #Note: check dimensions
                # f_cte = f_cte.repeat(1, 1, case["T"], 1) \
                #     .reshape(bs * self.n_objects * case["T"], self.feat_cte_dim)
                f_mu_cte, f_logvar_cte = self.linear_f_cte_mu(f_cte), \
                                         self.linear_f_cte_logvar(f_cte)
                f_cte = _sample_latent_simple(f_mu_cte, f_logvar_cte)

                # Register statistics
                if case_name not in cte_distr:
                    cte_distr[case_name] = (f_mu_cte, f_logvar_cte)
                else:
                    cte_distr[case_name] = (torch.cat([cte_distr[case_name][0], f_mu_cte],     dim=-1),
                                            torch.cat([cte_distr[case_name][1], f_logvar_cte], dim=-1))

                # Get full feature vector
                f = torch.cat([ f_dyn_tmp,
                                f_cte], dim=-1)

                # Get coarse features from which obtain queries and/or decode
                f_coarse = self.image_decoder(f, block = 'coarse')
                f_coarse = f_coarse.reshape(bs, self.n_objects, case["T"], *f_coarse.shape[1:])

            # Get output
            outs[case_name] = self.image_decoder(f_coarse, block = 'to_x')


        # Convolutional decoder. Normally Spatial Broadcasting decoder
        out_rec = outs["rec"]
        out_pred = outs["pred"]
        # TODO: concatenate cte_mu and cte_logvar to dyn statistics.
        #  - Check that this is done in last dimension!!
        # returned_mus = ...

        # Test disentanglement - TO REVIEW
        # if random.random() < 0.1 or self.content is None:
        #     self.content = f_cte
        # f_cte = self.content

        ''' -------------------- '''
        returned_g = torch.cat([g, G_for_pred], dim=1)
        returned_mus = [cte_distr["rec"][0], f_mu_dyn, cte_distr["pred"][0]]
        returned_logvars = [cte_distr["rec"][1], f_logvar_dyn, cte_distr["pred"][1]]
        # returned_mus = torch.cat([f_mu, f_mu_pred], dim=1)
        # returned_logvars = torch.cat([f_logvar, f_logvar_pred], dim=1)

        o_touple = (out_rec, out_pred, returned_g.reshape(-1, returned_g.size(-1)),
                    returned_mus,
                    returned_logvars)
                    # f_mu.reshape(-1, f_mu.size(-1)),
                    # f_logvar.reshape(-1, f_logvar.size(-1)))
        o = [item if isinstance(item, list) else item.reshape(torch.Size([bs * self.n_objects, -1]) + item.size()[1:]) for item in o_touple]
        # Option 1: one object mapped to 0
        # shape_o0 = o[0].shape
        # o[0] = o[0].reshape(bs, self.n_objects, *o[0].shape[1:])
        # o[0][:,0] = o[0][:,0]*0
        # o[0] = o[0].reshape(*shape_o0)

        # o[:2] = [torch.clamp(torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:]), dim=1), min=0, max=1) for item in o[:2]]

        # Test object decomposition
        # o[:2] = [torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:])[:,0:1], dim=1) for item in o[:2]]
        o[:2] = [torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:]), dim=1) for item in o[:2]]

        # Note: In case we want to sum KL between objects
        # o[3:5] = [item.reshape(bs, self.n_objects, *item.shape[1:]) for item in o[3:5]]

        o.append(A)
        o.append(B)

        o.append(u.reshape(bs * self.n_objects, -1, u.shape[-1]))
        o.append(u_dist.reshape(bs * self.n_objects, -1, *u_dist.shape[-1:])) #Append udist for categorical
        o.append(g_for_koop.reshape(bs * self.n_objects, -1, g_for_koop.shape[-1]))
        o.append(fit_err)

        return o