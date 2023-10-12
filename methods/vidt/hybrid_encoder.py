import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

class HybridEncoder(nn.Module):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = []
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(
                        hidden_dim)))
        self.input_proj = nn.ModuleList(self.input_proj)

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        # top-down fpn
        self.lateral_convs = []
        self.fpn_blocks = []
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))
        self.lateral_convs = nn.ModuleList(self.lateral_convs)
        self.fpn_blocks = nn.ModuleList(self.fpn_blocks)

        # bottom-up pan
        self.downsample_convs = []
        self.pan_blocks = []
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))
        self.downsample_convs = nn.ModuleList(self.downsample_convs)
        self.pan_blocks = nn.ModuleList(self.pan_blocks)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return torch.concat(
            [
                torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),
                torch.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self, feats, for_mot=False, is_teacher=False):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, size=feat_low.shape[2:], mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat(
                    [upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat(
                [downsample_feat, feat_height], axis=1))
            outs.append(out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]

class ShapeSpec(
        namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super(ShapeSpec, cls).__new__(cls, channels, height, width,
                                             stride)

class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(* [
            RepVggBlock(
                hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

class BaseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(
            out_channels)

        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.bn(self.conv(x))
        y = x * F.sigmoid(x)
        return y

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', alpha=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act
        if alpha:
            self.alpha = self.create_parameter(
                shape=[1],
                attr=ParamAttr(initializer=Constant(value=1.)),
                dtype="float32")
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight = kernel
        self.conv.bias = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(
            ch_out)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

def conv_init_(module):
    bound = 1 / np.sqrt(np.prod(module.weight.shape[1:]))
    uniform_(module.weight, -bound, bound)
    if module.bias is not None:
        uniform_(module.bias, -bound, bound)
        
def uniform_(tensor, a, b):
    """
    Modified tensor inspace using uniform_
    Args:
        tensor (torch.Tensor): torch Tensor
        a (float|int): min value.
        b (float|int): max value.
    Return:
        tensor
    """
    return _no_grad_uniform_(tensor, a, b)

def _no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        tensor = torch.nn.init.uniform(
                tensor, a=a, b=b)
    return tensor

def identity(x):
    return x

def mish(x):
    return F.mish(x) if hasattr(F, mish) else x * F.tanh(F.softplus(x))

def silu(x):
    return F.silu(x)

def swish(x):
    return x * F.sigmoid(x)

TRT_ACT_SPEC = {'swish': swish, 'silu': swish}

ACT_SPEC = {'mish': mish, 'silu': silu}

def get_act_fn(act=None, trt=False):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'
    if not act:
        return identity

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    if trt and name in TRT_ACT_SPEC:
        fn = TRT_ACT_SPEC[name]
    elif name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)