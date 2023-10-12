from unicodedata import east_asian_width
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb
from util.misc import NestedTensor
from typing import Type, Callable, Tuple, Optional, Set, List, Union
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models._efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv

# Visualization
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import cv2

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class XCA_pure(nn.Module):
    """ Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim,bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B,N,C = x.shape
        C = int(C // 3)

        # Result of next line is (qkv, B, num (H)eads,  (C')hannels per head, N)
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # Paper section 3.2 l2-Normalization and temperature scaling
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, C', N), permute -> (B, N, H, C')
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows to augment the
    implicit communication performed by the block diagonal scatter attention. Implemented using 2 layers of separable
    3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.bn = LayerNorm(in_features, eps=1e-6, data_format='channels_first')
        self.conv2 = nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        B,C,H,W = x.shape
        x = self.bn(x)
        x = self.conv2(x)
        return x

# class MixBlock(nn.Module):
#     # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1.):
#         super().__init__()
#         self.dim = dim
#         self.pos_embed = PositionalEncodingFourier(dim=dim)
#         self.conv1 = nn.Conv2d(dim, 3 * dim, 1)
#         self.conv2 = nn.Conv2d(dim, dim, 1)
#         self.conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.local_mp = LPI(in_features=dim, act_layer=act_layer)
#         self.attn = XCA_pure(
#             dim,
#             num_heads=num_heads, qkv_bias=qkv_bias,
#             attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         self.sa_weight = nn.Parameter(torch.Tensor([0.0]))

#         self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
#         self.norm2 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
#         self.norm3 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        
#         self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
#         self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
#         self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        
#     def forward(self, x):
#         B, _, H, W = x.shape
#         x = x + self.pos_embed(B, H, W)

#         residual = x
#         x = self.norm1(x)
#         qkv = self.conv1(x)
        
#         conv = qkv[:, 2 * self.dim:, :, :]
#         conv = self.conv(conv)
        
#         sa = qkv.flatten(2).transpose(1, 2)
#         sa = self.attn(sa)
#         sa = sa.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
#         x = self.conv2((torch.sigmoid(self.sa_weight) * sa + (1 - torch.sigmoid(self.sa_weight)) * conv)).flatten(2).transpose(1, 2)
#         x = self.gamma1 * x
#         x = x.reshape(B,H,W,-1).permute(0, 3, 1, 2).contiguous()
#         x = residual + self.drop_path(x)

#         residual = x
#         x = self.norm2(x)
#         x = self.local_mp(x).flatten(2).transpose(1, 2)
#         x = self.gamma2 * x
#         x = x.reshape(B,H,W,-1).permute(0, 3, 1, 2).contiguous()
#         x = residual + self.drop_path(x)

#         residual = x
#         x = self.norm3(x)
#         x = self.mlp(x).flatten(2).transpose(1, 2)
#         x = self.gamma3 * x
#         x = x.reshape(B,H,W,-1).permute(0, 3, 1, 2).contiguous()
#         x = residual + self.drop_path(x)
#         return x

def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.
    Args:
        *args: Ignored.
        **kwargs: Ignored.
    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation

class MBConv(nn.Module):
    """ MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf.
        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))
        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).
        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.
        Note: This implementation differs slightly from the original MobileNet implementation!
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MBConv, self).__init__()
        # Save parameter
        self.drop_path_rate: float = drop_path
        # Check parameters for downscaling
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=2 if downscale else 1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            SqueezeExcite(in_chs=out_channels, rd_ratio=0.25),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        )
        # Make skip path
        self.skip_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        ) if downscale else nn.Identity()

    def forward(self,input:torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output

class MixBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1.):
        super().__init__()
        # Conv
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.mbconv = MBConv(dim, dim, downscale=False, act_layer=act_layer)
        self.norm1 = LayerNorm(dim, eps=1e-6)

        # SA
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.pwconv = nn.Linear(dim, dim, 1)
        self.attn = XCA_pure(
                    dim,
                    num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop)
        self.sa_weight = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)

        # MLP
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = act_layer()
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)
        self.gamma1 = nn.Parameter(eta * torch.ones((dim)), 
                                    requires_grad=True) if eta > 0 else None
        self.gamma2 = nn.Parameter(eta * torch.ones((dim)), 
                                    requires_grad=True) if eta > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        residual = x

        # Conv
        x = self.mbconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # SA
        x = self.norm1(x)
        qkv = self.qkv(x.permute(0, 3, 1, 2))
        qkv = qkv.flatten(2).transpose(1, 2)
        sa = self.attn(qkv).reshape(B,H,W,-1)

        x = torch.sigmoid(self.sa_weight) * sa + (1 - torch.sigmoid(self.sa_weight)) * x
        x = self.pwconv(x)
        if self.gamma1 is not None:
            x = self.gamma1 * x
        x = x.permute(0,3,1,2)
        x = residual + self.drop_path(x)

        # MLP
        residual = x
        x = x.permute(0,2,3,1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = residual + self.drop_path(x)
        return x
        
class Container(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, patch_size=[16, 8, 4, 2], in_chans=3, base_embed=256, embed_dim=[64, 128, 320, 512], depth=[3, 4, 8, 3],
                 num_heads=12, mlp_ratio=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, 
                 eta=1.0, tokens_norm=True, det_attn_layers=2):
        """
        Args:
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        self.embed_dim = embed_dim
        self.num_channels = embed_dim
        self.depth = depth
        self.num_heads=num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.act_layer = nn.GELU
        self.norm_layer = norm_layer
        self.eta = eta 
        self.tokens_norm = tokens_norm
        self.det_attn_layers = det_attn_layers      

        # self.stem = nn.Sequential(
        #             nn.Conv2d(in_channels=in_chans, out_channels=base_embed, kernel_size=(3, 3), stride=(2, 2),
        #                       padding=(1, 1)),
        #             self.act_layer(),
        #             nn.Conv2d(in_channels=base_embed, out_channels=embed_dim[0], kernel_size=(3, 3), stride=(1, 1),
        #                       padding=(1, 1)),
        #             self.act_layer(),
        #         )

        self.patch_embed1 = nn.Sequential(
                                nn.Conv2d(in_chans, embed_dim[0], kernel_size=patch_size[0], stride=patch_size[0]),
                                LayerNorm(embed_dim[0], eps=1e-6, data_format="channels_first")
                            )
        self.patch_embed2 = nn.Sequential(
                                LayerNorm(embed_dim[0], eps=1e-6, data_format="channels_first"),
                                nn.Conv2d(embed_dim[0], embed_dim[1], kernel_size=patch_size[1], stride=patch_size[1]),
                            )
        self.patch_embed3 = nn.Sequential(
                                LayerNorm(embed_dim[1], eps=1e-6, data_format="channels_first"),
                                nn.Conv2d(embed_dim[1], embed_dim[2], kernel_size=patch_size[2], stride=patch_size[2]),
                            )
        self.patch_embed4 = nn.Sequential(
                                LayerNorm(embed_dim[2], eps=1e-6, data_format="channels_first"),
                                nn.Conv2d(embed_dim[2], embed_dim[3], kernel_size=patch_size[3], stride=patch_size[3]),
                            )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mixture =True
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        
        self.blocks1 = nn.Sequential(*[
            MixBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=self.act_layer, norm_layer=norm_layer, eta=eta)
            for i in range(depth[0])])
        self.blocks2 = nn.Sequential(*[
            MixBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], act_layer=self.act_layer,  norm_layer=norm_layer, eta=eta)
            for i in range(depth[1])])
        self.blocks3 = nn.Sequential(*[
            MixBlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], act_layer=self.act_layer,  norm_layer=norm_layer, eta=eta)
            for i in range(depth[2])])
        self.blocks4 = nn.Sequential(*[
            MixBlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], act_layer=self.act_layer,  norm_layer=norm_layer, eta=eta)
            for i in range(depth[3])])

#         self.fpn1 = nn.ConvTranspose2d(base_embed, base_embed, kernel_size=2, stride=2)

#         self.fpn2 = nn.Conv2d(base_embed, base_embed, kernel_size=1)

#         self.fpn3 = nn.Sequential(
#                     nn.Conv2d(base_embed, base_embed, kernel_size=1),
#                     nn.MaxPool2d(kernel_size=2, stride=2))

#         self.fpn4 = nn.Sequential(
#                     nn.Conv2d(base_embed, base_embed, kernel_size=1),
#                     nn.MaxPool2d(kernel_size=4, stride=4))

        self.norm1 = LayerNorm(embed_dim[0], eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(embed_dim[1], eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(embed_dim[2], eps=1e-6, data_format="channels_first")
        self.norm4 = LayerNorm(embed_dim[3], eps=1e-6, data_format="channels_first")

        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_features(self, x):
        # sample = x
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # x = self.stem(x)
        features = []

        x = self.patch_embed1(x)
        x = self.blocks1(x)
        features.append(self.norm1(x))
        
        x = self.patch_embed2(x)
        x = self.blocks2(x)
        features.append(self.norm2(x))

        x = self.patch_embed3(x)
        x = self.blocks3(x)
        features.append(self.norm3(x))

        x = self.patch_embed4(x)
        x = self.blocks4(x)
        features.append(self.norm4(x))

        # # Visualization
        # import matplotlib.pyplot as plt 
        # import matplotlib.patches as patches
        # import cv2
        # import numpy as np
        
        # with torch.no_grad():
        #     scale=3
        #     b=0
        #     ori_mean = torch.as_tensor([0.485, 0.456, 0.406])
        #     ori_std = torch.as_tensor([0.229, 0.224, 0.225])
        #     sample[b] = sample[b] * ori_mean.view(3,1,1).cuda()
        #     sample[b] = sample[b] + ori_std.view(3,1,1).cuda()
            
        #     img_h = sample[b].size(1)
        #     img_w = sample[b].size(2)
        #     plt.imshow(sample[b].permute(1,2,0).cpu().numpy())
            
        #     plt.imshow(cv2.resize(np.mean(features[scale][b].detach().cpu().numpy(),axis=0), (img_w,img_h)),cmap='rainbow', alpha=0.5)
        #     # plt.savefig("PredictiveUncertainty2_hm.png",bbox_inches='tight',dpi=600)
        #     plt.show()
        #     plt.clf()

        return features

    def forward(self, x):
        features = self.forward_features(x)    
        return features

def container_s(pretrained=False, **kwargs):
    embed_dim= [64, 128, 256, 512]
    model = Container(patch_size=[4, 2, 2, 2], in_chans=3, base_embed=64, embed_dim=embed_dim, 
                  depth=[4, 4, 12, 4],
                 num_heads=[2, 4, 8, 16], mlp_ratio=[4, 4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 eta=1e-6, tokens_norm=True)
    return model

def container_m(pretrained=False, **kwargs):
    embed_dim= [96, 192, 384, 768]
    model = Container(patch_size=[4, 2, 2, 2], in_chans=3, base_embed=256, embed_dim=embed_dim, 
                  depth=[3, 4, 8, 3],
                 num_heads=[3, 6, 12, 24], mlp_ratio=[4, 4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 eta=1e-6, tokens_norm=True)
    return model

def container_l(pretrained=False, **kwargs):
    embed_dim= [128, 256, 512, 1024]
    model = Container(patch_size=[4, 2, 2, 2], in_chans=3, base_embed=256, embed_dim=embed_dim, 
                  depth=[3, 4, 8, 3],
                 num_heads=[4, 8, 16, 32], mlp_ratio=[4, 4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 eta=1e-6, tokens_norm=True)
    return model

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.num_channels = backbone.num_channels
        self.return_layers = [0,1,2,3]

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}

        if len(self.num_channels) > 1:
            for layer, x in enumerate(xs):
                if layer in self.return_layers:
                  m = tensor_list.mask
                  assert m is not None
                  mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                  out['layer{}'.format(layer)] = NestedTensor(x, mask)
                else:
                  continue
        else:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=xs[-1].shape[-2:]).to(torch.bool)[0]
            out['final_layer'] = NestedTensor(xs[-1], mask)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool):
        if name == 'container_s':
              backbone = container_s(pretrained=False)
        elif name == 'container_m':
              backbone = container_m(pretrained=False)
        elif name == 'container_l':
              backbone = container_l(pretrained=False)
        print(sum(param.numel() for param in backbone.parameters()))
        super().__init__(backbone, train_backbone)

class Joiner(nn.Sequential):
    def __init__(self, backbone):
        self.num_channels = backbone.num_channels
        super().__init__(backbone)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        for name, x in xs.items():
            out.append(x)
        return out
def build_container(args):
    backbone = Backbone(args.backbone_name, args.pre_trained)
    model = Joiner(backbone)
    return model