import torch
import math
import timm
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_, DropPath
from util.misc import NestedTensor
from .position_encoding import build_position_encoding

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for layer, x in enumerate(xs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['layer{}'.format(layer)] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 pretrain: bool):
        if name == 'convnext_tiny':
              backbone = timm.create_model('convnext_tiny_384_in22ft1k', 
                                        features_only=True,
                                        pretrained=pretrain)
              backbone.num_channels = [96, 192, 384, 768]
              
        elif name == 'convnextv2_tiny':
              backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', 
                                        features_only=True,
                                        pretrained=pretrain)
              backbone.num_channels = [96, 192, 384, 768]
              
        super().__init__(backbone, pretrain)
        
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        self.num_channels = backbone.num_channels
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

def build_convnext(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(args.backbone_name, args.pre_trained)
    model = Joiner(backbone,position_embedding)
    return model