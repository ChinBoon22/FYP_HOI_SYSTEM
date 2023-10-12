import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

import sys
sys.path.insert(0, '/home/cyphi02/MDS01/fyp-weapon-detection')

from methods.vidt.hybrid_encoder import CSPRepLayer
from test_helper_fns import *

# Define some constants for testing
IN_CHANNELS = [512, 1024, 2048]
FEAT_STRIDES = [8, 16, 32]
HIDDEN_DIM = 256
USE_ENCODER_IDX = [2]
NUM_ENCODER_LAYERS = 1
PE_TEMPERATURE = 10000
EXPANSION = 1.0
DEPTH_MULT = 1.0
ACT = 'silu'
TRT = False
EVAL_SIZE = None
NUM_BLOCKS = 3


BATCH = 4
HEIGHT = 64
WEIGHT = 64

# 1. Test the creation of CSP Rep Layer that will be used for feature fusion in HYBRID ENCODER
def test_create_csp_rep_layer():
    input_tensor = generate_x(BATCH, IN_CHANNELS[0], HEIGHT, WEIGHT)
    
    csp_rep_layer = CSPRepLayer(
                    HIDDEN_DIM * 2,
                    HIDDEN_DIM,
                    round(3 * DEPTH_MULT),
                    act=ACT,
                    expansion=EXPANSION) 
    
    output = csp_rep_layer(input_tensor)
    assert isinstance(csp_rep_layer.conv1, nn.Module)
    assert isinstance(csp_rep_layer.conv2, nn.Module)
    assert isinstance(csp_rep_layer.bottlenecks, nn.Sequential)
    assert len(csp_rep_layer.bottlenecks) == NUM_BLOCKS
    assert isinstance(csp_rep_layer.conv3, nn.Module)
    assert list(output.shape) == [BATCH, HIDDEN_DIM, HEIGHT, WEIGHT], f"Expected output shape {[BATCH, HIDDEN_DIM, HEIGHT, WEIGHT]}, but got {output.shape}"