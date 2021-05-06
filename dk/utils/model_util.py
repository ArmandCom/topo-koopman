import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from model.networks import resnet

def partial_load(pretrained_dict, model, skip_keys=[]):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not any([sk in k for sk in skip_keys])}
    skipped_keys = [k for k in pretrained_dict if k not in filtered_dict]

    # 2. overwrite entries in the existing state dict
    model_dict.update(filtered_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print('\nSkipped keys: ', skipped_keys)
    print('\nLoading keys: ', filtered_dict.keys())

def load_vince_model(path):
    checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
    checkpoint = {k.replace('feature_extractor.module.model.', ''): checkpoint[k] for k in checkpoint if 'feature_extractor' in k}
    return checkpoint

def load_tc_model():
    path = 'tc_checkpoint.pth.tar'
    model_state = torch.load(path, map_location='cpu')['state_dict']

    net = resnet.resnet50()
    net_state = net.state_dict()

    for k in [k for k in model_state.keys() if 'encoderVideo' in k]:
        kk = k.replace('module.encoderVideo.', '')
        tmp = model_state[k]
        if net_state[kk].shape != model_state[k].shape and net_state[kk].dim() == 4 and model_state[k].dim() == 5:
            tmp = model_state[k].squeeze(2)
        net_state[kk][:] = tmp[:]

    net.load_state_dict(net_state)

    return net

def load_uvc_model():
    net = resnet.resnet18()
    net.avgpool, net.fc = None, None

    ckpt = torch.load('uvc_checkpoint.pth.tar', map_location='cpu')
    state_dict = {k.replace('module.gray_encoder.', ''):v for k,v in ckpt['state_dict'].items() if 'gray_encoder' in k}
    net.load_state_dict(state_dict)

    return net

class From3D(nn.Module):
    ''' Use a 2D convnet as a 3D convnet '''
    def __init__(self, resnet):
        super(From3D, self).__init__()
        self.model = resnet

    def forward(self, x):
        B, N, T, C, h, w = x.shape
        xx = x.reshape(-1, C, h, w)
        m = self.model(xx)
        return m #.reshape(B, N, T, *m.shape[-3:])#.permute(0, 2, 1, 3, 4)


def make_encoder(model_type, remove_layers=None):
    if model_type == 'scratch':
        net = resnet.resnet18()
        net.modify(padding='reflect')

    elif model_type == 'scratch_zeropad':
        net = resnet.resnet18()

    elif model_type == 'imagenet18':
        net = resnet.resnet18(pretrained=True)

    elif model_type == 'imagenet50':
        net = resnet.resnet50(pretrained=True)

    elif model_type == 'moco50':
        net = resnet.resnet50(pretrained=False)
        net_ckpt = torch.load('moco_v2_800ep_pretrain.pth.tar')
        net_state = {k.replace('module.encoder_q.', ''):v for k,v in net_ckpt['state_dict'].items() \
                     if 'module.encoder_q' in k}
        partial_load(net_state, net)

    elif model_type == 'timecycle':
        net = load_tc_model()

    elif model_type == 'uvc':
        net = load_uvc_model()

    else:
        assert False, 'invalid model_type'

    if hasattr(net, 'modify') and remove_layers is not None:
        net.modify(remove_layers=remove_layers)
        # parser.add_argument('--remove-layers', default=['layer4'], help='layer[1-4]')

    if 'Conv2d' in str(net):
        net = From3D(net)

    return net


#             # Make spatial radius mask TODO use torch.sparse
#             restrict = utils.MaskedAttention(args.radius, flat=False)
#             D = restrict.mask(*feats.shape[-2:])[None]
#             D = D.flatten(-4, -3).flatten(-2)
#             D[D==0] = -1e10; D[D==1] = 0
#
# class MaskedAttention(nn.Module):
#     '''
#     A module that implements masked attention based on spatial locality
#     TODO implement in a more efficient way (torch sparse or correlation filter)
#     '''
#     def __init__(self, radius, flat=True):
#         super(MaskedAttention, self).__init__()
#         self.radius = radius
#         self.flat = flat
#         self.masks = {}
#         self.index = {}
#
#     def mask(self, H, W):
#         if not ('%s-%s' %(H,W) in self.masks):
#             self.make(H, W)
#         return self.masks['%s-%s' %(H,W)]
#
#     def index(self, H, W):
#         if not ('%s-%s' %(H,W) in self.index):
#             self.make_index(H, W)
#         return self.index['%s-%s' %(H,W)]
#
#     def make(self, H, W):
#         if self.flat:
#             H = int(H**0.5)
#             W = int(W**0.5)
#
#         gx, gy = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
#         D = ( (gx[None, None, :, :] - gx[:, :, None, None])**2 + (gy[None, None, :, :] - gy[:, :, None, None])**2 ).float() ** 0.5
#         D = (D < self.radius)[None].float()
#
#         if self.flat:
#             D = self.flatten(D)
#         self.masks['%s-%s' %(H,W)] = D
#
#         return D
#
#     def flatten(self, D):
#         return torch.flatten(torch.flatten(D, 1, 2), -2, -1)
#
#     def make_index(self, H, W, pad=False):
#         mask = self.mask(H, W).view(1, -1).byte()
#         idx = torch.arange(0, mask.numel())[mask[0]][None]
#
#         self.index['%s-%s' %(H,W)] = idx
#
#         return idx
#
#     def forward(self, x):
#         H, W = x.shape[-2:]
#         sid = '%s-%s' % (H,W)
#         if sid not in self.masks:
#             self.masks[sid] = self.make(H, W).to(x.device)
#         mask = self.masks[sid]
#
#         return x * mask[0]