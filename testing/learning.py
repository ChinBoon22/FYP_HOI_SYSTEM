import torch
from torchvision.models.detection.image_list import ImageList
import pytest

import sys
sys.path.insert(0, '/home/cyphi02/MDS01/fyp-weapon-detection')

from methods.vidt.efficientnetv2 import Backbone, build_efficientnetv2, Joiner, build_position_encoding
from util.misc import NestedTensor
from test_helper_fns import *

BATCH = 1
HEIGHT = 64
WIDTH = 64
CHANNEL = 3

    
def backbone_output_shapes_with_pretrain():
    # Test the output shapes of the Backbone
    backbone = Backbone(name="efficientnetv2_s", pretrain=False)
    
    token = generate_x((BATCH, CHANNEL, HEIGHT, WIDTH))
    mask = torch.ones(BATCH, HEIGHT, WIDTH, dtype=torch.bool)  
    dummy_tensor_list = NestedTensor(token, mask)
    
    features = backbone(dummy_tensor_list)
    
    print(features.keys())
    assert len(features) == 3  # Three layers: layer0, layer1, layer2
    for layer_name, feature in features.items():
        print(feature.tensors.shape)
        print(feature.mask.shape)
        assert feature.tensors.shape[0] == 1
        assert feature.mask.shape[0] == 1
        assert feature.tensors.shape[2] == feature.tensors.shape[3]

backbone_output_shapes_with_pretrain()