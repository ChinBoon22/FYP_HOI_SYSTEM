import pytest
import torch
import math

import sys
sys.path.insert(0, '/home/cyphi02/MDS01/fyp-weapon-detection')

from methods.vidt.position_encoding import PositionEmbeddingSine
from util.misc import NestedTensor
from test_helper_fns import *

BATCH = 4
HEIGHT = 64
WIDTH = 64
CHANNEL = 96

@pytest.fixture
def dummy_tensor_list():
    token = generate_x((BATCH, HEIGHT, WIDTH, CHANNEL))
    mask = torch.ones(BATCH, HEIGHT, WIDTH, dtype=torch.bool)  
    return NestedTensor(token, mask)
    
# Create a fixture to initialize the PositionEmbeddingSine instance with common parameters
@pytest.fixture
def position_embedding():
    return 
    
def test_sin_position_embedding_forward(position_embedding, dummy_tensor_list):
    """ 
    Purpose: Call the forward method of PositionEmbeddingSine with the dummy input.
    Expected Dimension = (BATCH, POS_FEATS * 2, HEIGHT, WIDTH)
    """ 
    pos_embedding = PositionEmbeddingSine(num_pos_feats=64, temperature=10000, normalize=False, scale=None)
    assert isinstance(position_embedding, PositionEmbeddingSine)
    assert position_embedding.num_pos_feats == 64
    assert position_embedding.temperature == 10000
    assert not position_embedding.normalize
    assert position_embedding.scale == 2 * math.pi
    
    pos_embedding = position_embedding(dummy_tensor_list)
    
    assert list(pos_embedding.shape) == [BATCH, position_embedding.num_pos_feats * 2, HEIGHT, WIDTH]
    
    
def test_sin_position_embedding_diff_pos_feats(dummy_tensor_list):
    pos_embedding =  PositionEmbeddingSine(num_pos_feats=128) 
    output = pos_embedding(dummy_tensor_list)  
    assert list(output.shape) == [BATCH, pos_embedding.num_pos_feats * 2, HEIGHT, WIDTH]

def test_sin_position_embedding_diff_temperature(dummy_tensor_list):
    pos_embedding =  PositionEmbeddingSine(temperature=5000)   
    output = pos_embedding(dummy_tensor_list)
    assert list(output.shape) == [BATCH, pos_embedding.num_pos_feats * 2, HEIGHT, WIDTH]

def test_sin_position_embedding_with_smmall_scale(dummy_tensor_list):
    pos_embedding =  PositionEmbeddingSine(scale=math.pi)   
    output = pos_embedding(dummy_tensor_list)
    assert list(output.shape) == [BATCH, pos_embedding.num_pos_feats * 2, HEIGHT, WIDTH]
    
def test_sin_position_embedding_with_norm(dummy_tensor_list):
    pos_embedding =  PositionEmbeddingSine(normalize=True)   
    output = pos_embedding(dummy_tensor_list)
    assert list(output.shape) == [BATCH, pos_embedding.num_pos_feats * 2, HEIGHT, WIDTH]
