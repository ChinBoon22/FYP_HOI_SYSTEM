import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

import sys
sys.path.insert(0, '/home/cyphi02/MDS01/fyp-weapon-detection')

from methods.vidt.hybrid_encoder import CSPRepLayer, HybridEncoder
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

# Fixture
@pytest.fixture
def csp_rep_layer():
    return CSPRepLayer(
                    HIDDEN_DIM * 2,
                    HIDDEN_DIM,
                    round(3 * DEPTH_MULT),
                    act=ACT,
                    expansion=EXPANSION) 
@pytest.fixture
def dummy_input():
    # Create dummy input features with the specified in_channels
    return [generate_x(BATCH, IN_CHANNELS[i], HEIGHT//2**i, WEIGHT//2**i) for i in range(len(IN_CHANNELS))]
    
# 1. Test the creation of CSP Rep Layer that will be used for feature fusion in HYBRID ENCODER
def test_create_csp_rep_layer(csp_rep_layer):
    input_tensor = generate_x(BATCH, IN_CHANNELS[0], HEIGHT, WEIGHT)
    output = csp_rep_layer(input_tensor)
    assert isinstance(csp_rep_layer.conv1, nn.Module)
    assert isinstance(csp_rep_layer.conv2, nn.Module)
    assert isinstance(csp_rep_layer.bottlenecks, nn.Sequential)
    assert len(csp_rep_layer.bottlenecks) == NUM_BLOCKS
    assert isinstance(csp_rep_layer.conv3, nn.Module)
    assert list(output.shape) == [BATCH, HIDDEN_DIM, HEIGHT, WEIGHT], f"Expected output shape {[BATCH, HIDDEN_DIM, HEIGHT, WEIGHT]}, but got {output.shape}"

    
# 2.Test Hybrid Encoder
def test_hybrid_encoder_forward_pass(dummy_input):
    """
    # print(dummy_input[0].shape) # torch.Size([4, 512, 64, 64])
    # print(dummy_input[1].shape) # torch.Size([4, 1024, 32, 32])
    # print(dummy_input[2].shape) # torch.Size([4, 2048, 16, 16])
    
    # print("output 1 shape: ", outputs[0].shape) # [4, 256, 64, 64]
    # print("output 2 shape: ", outputs[1].shape) # [4, 256, 32, 32]
    # print("output 3 shape: ", outputs[2].shape) # [4, 256, 16, 16]
    """
    hybrid_encoder = HybridEncoder(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=HIDDEN_DIM,
        use_encoder_idx=USE_ENCODER_IDX,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        pe_temperature=PE_TEMPERATURE,
        expansion=EXPANSION,
        depth_mult=DEPTH_MULT,
        act=ACT,
        trt=TRT,
        eval_size=EVAL_SIZE)    
        
    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    
    # Check if the outputs have the expected number of elements
    assert len(outputs) == len(IN_CHANNELS)
    assert list(outputs[0].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**0, WEIGHT//2**0]
    assert list(outputs[1].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**1, WEIGHT//2**1]
    assert list(outputs[2].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**2, WEIGHT//2**2]
    
# 2. 
def test_hybrid_encoder_larger_hidden_dimension(dummy_input):
    # Create dummy input features with the specified in_channels
    large_hidden_dimension = 512
    
    hybrid_encoder = HybridEncoder(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=large_hidden_dimension,
        use_encoder_idx=USE_ENCODER_IDX,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        pe_temperature=PE_TEMPERATURE,
        expansion=EXPANSION,
        depth_mult=DEPTH_MULT,
        act=ACT,
        trt=TRT,
        eval_size=EVAL_SIZE) 
     
    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    
    # Check if the outputs have the expected number of elements
    assert len(outputs) == len(IN_CHANNELS)
    assert list(outputs[0].shape) == [BATCH, large_hidden_dimension, HEIGHT//2**0, WEIGHT//2**0]
    assert list(outputs[1].shape) == [BATCH, large_hidden_dimension, HEIGHT//2**1, WEIGHT//2**1]
    assert list(outputs[2].shape) == [BATCH, large_hidden_dimension, HEIGHT//2**2, WEIGHT//2**2]

def test_hybrid_encoder_smaller_hidden_dimension(dummy_input):
    # Create dummy input features with the specified in_channels
    small_hidden_dimension = 128
    
    hybrid_encoder = HybridEncoder(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=small_hidden_dimension,
        use_encoder_idx=USE_ENCODER_IDX,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        pe_temperature=PE_TEMPERATURE,
        expansion=EXPANSION,
        depth_mult=DEPTH_MULT,
        act=ACT,
        trt=TRT,
        eval_size=EVAL_SIZE) 

    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    
    # Check if the outputs have the expected number of elements
    assert len(outputs) == len(IN_CHANNELS)
    assert list(outputs[0].shape) == [BATCH, small_hidden_dimension, HEIGHT//2**0, WEIGHT//2**0]
    assert list(outputs[1].shape) == [BATCH, small_hidden_dimension, HEIGHT//2**1, WEIGHT//2**1]
    assert list(outputs[2].shape) == [BATCH, small_hidden_dimension, HEIGHT//2**2, WEIGHT//2**2]


def test_hybrid_encoder_more_encoder_layers(dummy_input):
    # Create dummy input features with the specified in_channels
    more_encoder_layers = 3
    
    hybrid_encoder = HybridEncoder(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=HIDDEN_DIM,
        use_encoder_idx=USE_ENCODER_IDX,
        num_encoder_layers=more_encoder_layers,
        pe_temperature=PE_TEMPERATURE,
        expansion=EXPANSION,
        depth_mult=DEPTH_MULT,
        act=ACT,
        trt=TRT,
        eval_size=EVAL_SIZE) 

    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    
    # Check if the outputs have the expected number of elements
    assert len(outputs) == len(IN_CHANNELS)
    assert list(outputs[0].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**0, WEIGHT//2**0]
    assert list(outputs[1].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**1, WEIGHT//2**1]
    assert list(outputs[2].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**2, WEIGHT//2**2]
 
def test_hybrid_encoder_small_temperature(dummy_input):
    # Create dummy input features with the specified in_channels
    small_temperature = 5000
    
    hybrid_encoder = HybridEncoder(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=HIDDEN_DIM,
        use_encoder_idx=USE_ENCODER_IDX,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        pe_temperature=small_temperature,
        expansion=EXPANSION,
        depth_mult=DEPTH_MULT,
        act=ACT,
        trt=TRT,
        eval_size=EVAL_SIZE) 
     
    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    
    # Check if the outputs have the expected number of elements
    assert len(outputs) == len(IN_CHANNELS)
    assert list(outputs[0].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**0, WEIGHT//2**0]
    assert list(outputs[1].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**1, WEIGHT//2**1]
    assert list(outputs[2].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**2, WEIGHT//2**2]

def test_hybrid_encoder_small_temperature(dummy_input):
    # Create dummy input features with the specified in_channels
    small_temperature = 5000
    
    hybrid_encoder = HybridEncoder(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=HIDDEN_DIM,
        use_encoder_idx=USE_ENCODER_IDX,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        pe_temperature=small_temperature,
        expansion=EXPANSION,
        depth_mult=DEPTH_MULT,
        act=ACT,
        trt=TRT,
        eval_size=EVAL_SIZE) 
     
    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    
    # Check if the outputs have the expected number of elements
    assert len(outputs) == len(IN_CHANNELS)
    assert list(outputs[0].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**0, WEIGHT//2**0]
    assert list(outputs[1].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**1, WEIGHT//2**1]
    assert list(outputs[2].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**2, WEIGHT//2**2]

def test_hybrid_encoder_larger_expansion(dummy_input):
    # Create dummy input features with the specified in_channels
    large_expansion = 2
    
    hybrid_encoder = HybridEncoder(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=HIDDEN_DIM,
        use_encoder_idx=USE_ENCODER_IDX,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        pe_temperature=PE_TEMPERATURE,
        expansion=large_expansion,
        depth_mult=DEPTH_MULT,
        act=ACT,
        trt=TRT,
        eval_size=EVAL_SIZE) 
     
    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    
    # Check if the outputs have the expected number of elements
    assert len(outputs) == len(IN_CHANNELS)
    assert list(outputs[0].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**0, WEIGHT//2**0]
    assert list(outputs[1].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**1, WEIGHT//2**1]
    assert list(outputs[2].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**2, WEIGHT//2**2]

def test_hybrid_encoder_larger_depth_mult(dummy_input):
    """
    Larger Model Depth, double the layers in the network
    """
    large_depth_mult = 2
    
    hybrid_encoder = HybridEncoder(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=HIDDEN_DIM,
        use_encoder_idx=USE_ENCODER_IDX,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        pe_temperature=PE_TEMPERATURE,
        expansion=EXPANSION,
        depth_mult=large_depth_mult,
        act=ACT,
        trt=TRT,
        eval_size=EVAL_SIZE) 
     
    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    
    # Check if the outputs have the expected number of elements
    assert len(outputs) == len(IN_CHANNELS)
    assert list(outputs[0].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**0, WEIGHT//2**0]
    assert list(outputs[1].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**1, WEIGHT//2**1]
    assert list(outputs[2].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**2, WEIGHT//2**2]

def test_hybrid_encoder_with_trt(dummy_input):
    """
    trt is a boolean parameter that stands for TensorRT, which is a deep learning inference optimizer and runtime developed by NVIDIA.
    """
    
    hybrid_encoder = HybridEncoder(
        in_channels=IN_CHANNELS,
        feat_strides=FEAT_STRIDES,
        hidden_dim=HIDDEN_DIM,
        use_encoder_idx=USE_ENCODER_IDX,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        pe_temperature=PE_TEMPERATURE,
        expansion=EXPANSION,
        depth_mult=DEPTH_MULT,
        act=ACT,
        trt=True,
        eval_size=EVAL_SIZE) 
     
    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    
    # Check if the outputs have the expected number of elements
    assert len(outputs) == len(IN_CHANNELS)
    assert list(outputs[0].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**0, WEIGHT//2**0]
    assert list(outputs[1].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**1, WEIGHT//2**1]
    assert list(outputs[2].shape) == [BATCH, HIDDEN_DIM, HEIGHT//2**2, WEIGHT//2**2]