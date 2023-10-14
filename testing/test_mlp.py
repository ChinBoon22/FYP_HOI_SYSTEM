import pytest

import sys
sys.path.insert(0, '/home/cyphi02/MDS01/fyp-weapon-detection')

from torch import nn
from methods.vidt.deformable_transformer import MLP
from test_helper_fns import generate_x

BATCH = 4
NUM_LAYER = 4
ACTIVATION = nn.ReLU()

def test_mlp_standard_input():
    INPUT_DIMENSION = 256
    HIDDEN_DIMENSION = OUTPUT_DIMENSION  = 256
    x = generate_x((BATCH, INPUT_DIMENSION))
    mlp = MLP(input_dim=INPUT_DIMENSION, hidden_dim=HIDDEN_DIMENSION, output_dim=OUTPUT_DIMENSION, num_layers=NUM_LAYER, activation=ACTIVATION)
    assert list(mlp(x).shape) == [BATCH, OUTPUT_DIMENSION]
    
def test_mlp_dims_smaller_hidden():
    INPUT_DIMENSION = 256
    HIDDEN_DIMENSION = 128
    OUTPUT_DIMENSION  = 256
    x=generate_x((BATCH, INPUT_DIMENSION))
    mlp = MLP(input_dim=INPUT_DIMENSION, hidden_dim=HIDDEN_DIMENSION, output_dim=OUTPUT_DIMENSION, num_layers=NUM_LAYER, activation=ACTIVATION)
    assert list(mlp(x).shape) == [BATCH, OUTPUT_DIMENSION]
    
def test_mlp_dims_larger_output():
    INPUT_DIMENSION = 256
    HIDDEN_DIMENSION = 256
    OUTPUT_DIMENSION  = 512
    x=generate_x((BATCH, INPUT_DIMENSION))
    mlp = MLP(input_dim=INPUT_DIMENSION, hidden_dim=HIDDEN_DIMENSION, output_dim=OUTPUT_DIMENSION, num_layers=NUM_LAYER, activation=ACTIVATION)
    assert list(mlp(x).shape) == [BATCH, OUTPUT_DIMENSION]
    
    
def test_mlp_dims_larger_hidden_smaller_output():
    INPUT_DIMENSION = 256
    HIDDEN_DIMENSION = 512
    OUTPUT_DIMENSION  = 256
    x=generate_x((BATCH, INPUT_DIMENSION))
    mlp = MLP(input_dim=INPUT_DIMENSION, hidden_dim=HIDDEN_DIMENSION, output_dim=OUTPUT_DIMENSION, num_layers=NUM_LAYER, activation=ACTIVATION)
    assert list(mlp(x).shape) == [BATCH, OUTPUT_DIMENSION]
    
    
def test_mlp_dims_activation_gelu():
    ACTIVATION=nn.GELU()
    INPUT_DIMENSION = 256
    HIDDEN_DIMENSION = OUTPUT_DIMENSION  = 256
    x=generate_x((BATCH, INPUT_DIMENSION))
    mlp = MLP(input_dim=INPUT_DIMENSION, hidden_dim=HIDDEN_DIMENSION, output_dim=OUTPUT_DIMENSION, num_layers=NUM_LAYER, activation=ACTIVATION)
    assert list(mlp(x).shape) == [BATCH, OUTPUT_DIMENSION]
    
def test_mlp_dims_activation_tanh():
    ACTIVATION=nn.Tanh()
    INPUT_DIMENSION = 256
    HIDDEN_DIMENSION = OUTPUT_DIMENSION  = 256
    x=generate_x((BATCH, INPUT_DIMENSION))
    mlp = MLP(input_dim=INPUT_DIMENSION, hidden_dim=HIDDEN_DIMENSION, output_dim=OUTPUT_DIMENSION, num_layers=NUM_LAYER, activation=ACTIVATION)
    assert list(mlp(x).shape) == [BATCH, OUTPUT_DIMENSION]
    
def test_mlp_many_hidden_layers():
    INPUT_DIMENSION = 256
    HIDDEN_DIMENSION = OUTPUT_DIMENSION  = 256
    LARGE_NUM_LAYERS = 10
    x= generate_x((BATCH, INPUT_DIMENSION))
    mlp = MLP(input_dim=INPUT_DIMENSION, hidden_dim=HIDDEN_DIMENSION, output_dim=OUTPUT_DIMENSION, num_layers=LARGE_NUM_LAYERS, activation=ACTIVATION)
    assert list(mlp(x).shape) == [BATCH, OUTPUT_DIMENSION]