import torch
import math
import timm
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_, DropPath
from util.misc import NestedTensor

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.num_channels = backbone.embed_dim

    def forward(self, tensor_list: NestedTensor):
        xs = []
        x = self.backbone.stem(tensor_list.tensors)
        for stage in self.backbone.stages:
            x = stage(x)
            xs.append(x)

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
        if name == 'maxvit_nano':
              backbone = timm.create_model('maxvit_rmlp_nano_rw_256', 
                                        pretrained=pretrain)
        super().__init__(backbone, pretrain)
        
class Joiner(nn.Sequential):
    def __init__(self, backbone):
        self.num_channels = list(backbone.num_channels)
        super().__init__(backbone)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        for name, x in xs.items():
            out.append(x)
        return out

def build_maxvit(args):
    backbone = Backbone(args.backbone_name, args.pre_trained)
    model = Joiner(backbone)
    return model