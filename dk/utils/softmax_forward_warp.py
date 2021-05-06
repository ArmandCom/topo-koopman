import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(0)
def normalize_of(of, res):
    assert of.shape[-3] == len(res)
    norm_of = []
    for dim, r in enumerate(res):
        norm_of.append(of[..., dim, :, :]/(r-1))
    return torch.stack(norm_of, dim=-3)

def merge_warped(out_tensor, mode='sum'):
    if mode=='sum':
        out = out_tensor.sum(-1)
    else:
        raise NotImplementedError
    return out

# TODO: Add 1x1 padding to allow content to leave the scene.
def soft_forward_warp(input, of, temp=0.01):
    bs = input.shape[0]
    res = input.shape[-2:]
    ranges = [torch.linspace(0.0, r-1, steps=r) for r in res]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=0)[None].to(input.device)
    of_grid = (grid + of).flatten(-2)[..., None, :]
    # Note: skip normalize_of(of, res).
    #  If we normalize the grid and OF,
    #  the softmax temperature must be changed
    #  according to the resolution
    of_grid = of_grid.repeat_interleave(of_grid.shape[-1], dim=-2)
    logits = (of_grid -
              grid
              .flatten(-2)
              [..., None, :].repeat_interleave(of_grid.shape[-1], dim=-2)
              .transpose(-2,-1))\
        .abs() # TODO: Clamp to [0,res]?

    logits = logits.sum(-3, keepdims=True)
    locs = F.softmax(-logits/temp, dim=-2)
    input = input.flatten(-2)[..., None, :].repeat_interleave(of_grid.shape[-1], dim=-2)
    out = input * locs
    out = merge_warped(out, mode='sum')
    return out.reshape(bs, -1, *res)

# def soft_forward_warp_1D(input, of):
#     res = input.shape[-1]
#     range = torch.linspace(0.0, 1.0, steps=res)
#     grid = range[None, None].repeat(1, res, 1)
#     of_grid = grid + of/res
#     logits = (of_grid - grid.transpose(-2,-1)).abs()
#     locs = F.softmax(-logits/0.01, dim=-2)
#     input = input.repeat_interleave(res, dim=-2)
#     out = input * locs
#     return out.sum(-1)


class SoftForwardWarp(nn.Module):
    def __init__(self, res):
        super().__init__()
        self.res = res
        self.register_buffer('grid', self.get_mgrid(res))

    def get_mgrid(self, res):
        ranges = [torch.linspace(0.0, r-1, steps=r) for r in res]
        grid = torch.meshgrid(*ranges)
        grid = torch.stack(grid, dim=0)[None]
        return grid

    def forward(self, x, of, temp=0.1, constrain_of=False):
        bs = x.shape[0]
        of_grid = (self.grid + of).flatten(-2)[..., None, :]
        if constrain_of:
            of_grid = torch.cat([torch.clamp(of_grid[..., 0:1, :, :], min=0, max=self.res[0]-1),
                                 torch.clamp(of_grid[..., 1:2, :, :], min=0, max=self.res[1]-1)], dim=-3)
        # Note: skip normalize_of(of, res).
        #  If we normalize the grid and OF,
        #  the softmax temperature must be changed
        #  according to the resolution
        hw = of_grid.shape[-1]
        of_grid = of_grid.repeat_interleave(hw, dim=-2)
        logits = (of_grid -
                  self.grid
                  .flatten(-2)
                  [..., None, :].repeat_interleave(hw, dim=-2)
                  .transpose(-2,-1)) \
            .abs() # TODO: Clamp to [0,res]?

        logits = logits.sum(-3, keepdims=True)
        locs = F.softmax(-logits/temp, dim=-2)
        x = x.flatten(-2)[..., None, :].repeat_interleave(hw, dim=-2)
        out = x * locs
        out = merge_warped(out, mode='sum')
        return out.reshape(bs, -1, *self.res)

    def unroll_feat_map(self, feat):
        # Option 1: Vectorized
        h, w = feat.shape[-2:]
        feat_repeat = feat[:, None].repeat_interleave(h*w, dim=1)

        # Mask
        zeros = torch.zeros_like(feat_repeat).detach()
        linspace = torch.linspace(0, h*w -1, h*w).long()
        flat_grid = self.grid.flatten(-2).squeeze().long()
        zeros[:, linspace, ..., flat_grid[0, linspace], flat_grid[1, linspace]] = 1

        # Unroll
        unrolled_feat = zeros[:, linspace] * \
                        feat_repeat[:, linspace]
        return unrolled_feat

    def fold_feat_map(self, feat):
        bs, C, h, w = feat.shape
        N = h*w
        return feat.reshape(bs//N, N, C, h, w).sum(1)

    def expand_pose(self, pose):
        '''
        param pose: N x 3
        Takes 3-dimensional vectors, and massages them into 2x3 affine transformation matrices:
        [s,x,y] -> [[s,0,x],
                    [0,s,y]]
        '''
        n = pose.size(0)
        expansion_indices = Variable(torch.LongTensor([1, 0, 2, 0, 1, 3]).to(pose.device), requires_grad=False)
        zeros = Variable(torch.zeros(n, 1).to(pose.device), requires_grad=False)
        out = torch.cat([zeros, pose], dim=1)
        return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)

    def backward_warp(self,input, xy, constrain_of=True, mode='bilinear'):
        xy_shape = xy.shape
        N, n_channels, h, w = input.shape
        h, w = xy_shape[-2:]
        if constrain_of:
            xy = torch.cat([xy[..., 0:1, :, :]/((self.res[0])/2), xy[..., 1:2, :, :]/((self.res[1])/2)], dim=-3)
            xy = torch.clamp(xy, min=-1, max=1)
        xy_flat = xy.flatten(-2).transpose(-2, -1).flatten(0, -2)
        transformer = self.expand_pose(torch.cat([torch.ones_like(xy_flat[..., 0:1]), xy_flat], dim=-1))
        grid = F.affine_grid(transformer,
                             torch.Size((N, n_channels, h, w)))
        # print('grid: \n', grid)
        # print('input_unrolled: \n', input_unrolled)
        warped = F.grid_sample(input, grid, mode=mode)
        # print('warped: \n', warped)

        return warped.reshape(N, -1, h, w)

if __name__ == '__main__':

    # Note: Test_1d
    # res = [6]
    # input_ten = torch.randint(low=1, high=9, size=(1, 1, *res))
    # of = torch.zeros(1, 1, *res)
    # of[0,0,1] = 2
    # of[0,0,2] = -1
    # out = soft_forward_warp_1D(input_ten, of)

    # Note: Test_2d
    # res = [8, 8]
    # of = torch.zeros(1, 2, *res)
    # of[0,0,0,0] = 4
    # of[0,1,0,0] = 4
    # # of[0,0,2,5] = 1
    # # of[0,1,2,5] = 3
    # input_ten = torch.randint(low=1, high=5, size=(1, 1, *res))
    # out = soft_forward_warp(input_ten, of, temp=0.1)
    # print('OF: \n', of.int())
    # print('Input: \n', input_ten)
    # print('Output round: \n', torch.round(out).int())
    # print('Output: \n', torch.round(out * 100.)/100.)

    # Note: Test_2d with class
    # res = [4, 4]
    # of = torch.zeros(1, 2, *res)
    # of[0,0,0,0] = -2
    # of[0,1,0,0] = -1
    # # of[0,0,2,5] = 1
    # # of[0,1,2,5] = 3
    # input_ten = torch.randint(low=1, high=5, size=(1, 1, *res)).float()
    # soft_fw = SoftForwardWarp(input_ten.shape[-2:])
    # # # out = soft_fw(input_ten, of, temp=0.05)
    # # # print('OF: \n', of.int())
    # # # print('Input: \n', input_ten)
    # # # print('Output round: \n', torch.round(out).int())
    # # # print('Output: \n', torch.round(out * 100.)/100.)
    # out_1 = soft_fw.backward_warp(input_ten, of)
    # out = soft_fw.unroll_feat_map(out_1).sum(-1).sum(-1)
    # print('Input: \n', input_ten, '\n', input_ten.shape)
    # print('Output: \n', out_1, '\n', out_1.shape)
    # print('Output: \n', out, '\n', out.shape)

    # Note: Test backwards warp with unroll
    res = [4, 4]
    of = torch.zeros(1, 2, *res)
    of[0,0,2,2] = -1#/((res[0])/2)
    of[0,1,2,2] = 1#/((res[1])/2)
    # of[0,0,2,5] = 1
    # of[0,1,2,5] = 3
    input_ten = torch.randint(low=1, high=5, size=(1, 1, *res)).float()
    soft_fw = SoftForwardWarp(input_ten.shape[-2:])
    input_ten = soft_fw.unroll_feat_map(input_ten).reshape(-1, 1, *res)
    out = soft_fw.backward_warp(input_ten, of)
    # out_flat = out.sum(-1).sum(-1)
    # out_flat_2 = out.sum(0)
    # out = soft_fw(input_ten, of, temp=0.05)
    print('Input: \n', input_ten)
    print('Output: \n', out)
    # print('Output 2: \n', out_flat_2)