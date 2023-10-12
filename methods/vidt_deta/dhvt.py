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
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv

# Visualization
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import cv2

class DAFF (nn.Module ):
    def __init__ (self, in_dim, hid_dim, out_dim, kernel_size =3 ):
        self.conv1 = nn.Conv2d(in_dim, hid_dim, kernel_size =1,
                    stride=1, padding =0)
        self.conv2 = nn.Conv2d(
                    hid_dim, hid_dim, kernel_size=3, stride =1 ,
                    padding=(kernel_size-1)//2, groups=hid_dim)
        self.conv3 = nn.Conv2d(hid_dim, out_dim, kernel_size =1,
                    stride=1, padding=0)
        self.act = nn.GELU()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Linear(in_dim, in_dim//4)
        self.excitation = nn.Linear(in_dim//4, in_dim)
        self.bn1 = nn.BatchNorm2d(hid_dim)
        self.bn2 = nn.BatchNorm2d(hid_dim)
        self.bn3 = nn.BatchNorm2d(out_dim)

    def forward (self, x):
        B, N, C = x.size()
        cls_token, tokens = torch.split(x, [1, N-1], dim =1)
        x = tokens.reshape(B, int(math.sqrt(N-1)),
                      int(math.sqrt(N-1)), C).permute(0, 3, 1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        x = x + self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        weight = self.squeeze(x).flatten(1).reshape(B, 1, C)
        weight = self.excitation(self.act(self.compress(weight)))
        cls_token = cls_token * weight
        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim =1)
    return out

def conv3x3(in_dim, out_dim):
    return torch.nn.Sequential(
          nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),    
          nn.BatchNorm2d(out_dim)
    )

class Affine(nn.Module):
    def __init__(self, dim):
        self.alpha = nn.Parameter(torch.ones([1, dim, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, dim, 1, 1]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

class SOPE(nn.Module):
    def __init__(self, patch_size, embed_dim):
        self.pre_affine = Affine(3)
        self.post_affine = Afffine(embed_dim)
        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim//8, 2),
                nn.GELU(),
                conv3x3(embed_dim//8, embed_dim//4, 2),
                nn.GELU(),
                conv3x3(embed_dim//4, embed_dim//2, 2),
                nn.GELU(),
                conv3x3(embed_dim//2, embed_dim, 2),
                )
        elif patch_size[0] == 4:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim//2, 2),
                nn.GELU(),
                conv3x3(embed_dim//2, embed_dim, 2),
                )
        elif patch_size[0] == 2:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim, 2),
                nn.GELU(),
                )

    def forward(self, x):
        B, C, H, W = x.shape
        x = selff.pre_affine(x)
        x = self.proj(x)
        x = self.post_affine(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1,2)
        return x
        
class Attention(nn.Module):
    def __init(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads  
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linearr(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self..act = nn.GELU()
        self.ht_proj = nn.Linear(head_dim, dim, bias=True)
        self.ht_norm = nn.LayerNorm(head_dim)
        self.pos_embed = nn.Parameter(
                        torch.zeros(1, self.num_heads, dim))

    def forward(self, x):
        B, N, C = x.shape

        # head token
        head_pos = self.pos_embed.expand(x.shape[0], -1, -1)
        ht = x.reshape(B, -1, self.num_heads, C//self.num_-heads).permute(0,2,1,3)
        ht = ht.mean(dim=2)
        ht = self.ht_proj(ht).reshape(B, -1, self.num_heads, C//self.num_heads)
        ht = self.act(self.ht_norm(ht)).flatten(2)
        ht = ht + head_pos
        x = torch.cat([x, ht], dim=1)

        # common MHSA
        qkv = self.qkv(x).reshape(B, N+self.num_heads, 3, 
                          self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N+self.num_heads, C)
        x = self.proj(x)

        # split, average and add
        cls, patch, ht = torch.split(x, [1, N-1, self.num_heads], dim=1)
        cls = cls + torch.mean(ht, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)
        return x


class DHVT(nn.Module):
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

def dhvt_s(pretrained=False, **kwargs):
    embed_dim= [96, 192, 384, 768]
    model = DHVT(patch_size=[4, 2, 2, 2], in_chans=3, base_embed=64, embed_dim=embed_dim, 
                  depth=[2, 2, 8, 2],
                 num_heads=[3, 6, 12, 16], mlp_ratio=[4, 4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 eta=1e-6, tokens_norm=True)
    return model

def dhvt_m(pretrained=False, **kwargs):
    embed_dim= [96, 192, 384, 768]
    model = DHVT(patch_size=[4, 2, 2, 2], in_chans=3, base_embed=256, embed_dim=embed_dim, 
                  depth=[3, 4, 8, 3],
                 num_heads=[3, 6, 12, 24], mlp_ratio=[4, 4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 eta=1e-6, tokens_norm=True)
    return model

def dhvt_l(pretrained=False, **kwargs):
    embed_dim= [128, 256, 512, 1024]
    model = DHVT(patch_size=[4, 2, 2, 2], in_chans=3, base_embed=256, embed_dim=embed_dim, 
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

def build_dhvt(args):
    backbone = Backbone(args.backbone_name, args.pre_trained)
    model = Joiner(backbone)
    return model