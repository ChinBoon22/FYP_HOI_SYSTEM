import pytest
from torch import nn

import sys
sys.path.insert(0, '/home/cyphi02/MDS01/fyp-weapon-detection')

from methods.vidt.deformable_transformer import MLP
from test_helper_fns import generate_x

def test_MLP():
    INPUT_DIMENSION = 512
    HIDDEN_DIMENSION = OUTPUT_DIMENSION  = 256
    ACTIVATION = nn.ReLU()
    NUM_LAYER = 3
    B = 4
    x = generate_x((B, INPUT_DIMENSION))

    mlp = MLP(input_dim=INPUT_DIMENSION, hidden_dim=HIDDEN_DIMENSION, output_dim=OUTPUT_DIMENSION, num_layers=NUM_LAYER, activation=ACTIVATION)
    assert list(mlp(x).shape) == [B, HIDDEN_DIMENSION]
    
