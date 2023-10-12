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
                 pretrain: bool):
        if name == 'efficientnetv2_s':
              backbone = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', 
                                        features_only=True, pretrained=pretrain,
                                        out_indices=[2,3,4])
              backbone.num_channels=[64,160,256]
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

def build_efficientnetv2(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(args.backbone_name, args.pre_trained)
    model = Joiner(backbone,position_embedding)
    return model