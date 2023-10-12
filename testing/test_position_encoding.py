import pytest
import torch
import math

import sys
sys.path.insert(0, '/home/cyphi02/MDS01/fyp-weapon-detection')

from methods.vidt.position_encoding import PositionEmbeddingSine
from util.misc import NestedTensor

# Create a fixture to initialize the PositionEmbeddingSine instance with common parameters
@pytest.fixture
def position_embedding():
    return PositionEmbeddingSine(num_pos_feats=64, temperature=10000, normalize=False, scale=None)

def test_position_embedding_creation(position_embedding):
    # Ensure that the PositionEmbeddingSine instance is created correctly
    assert isinstance(position_embedding, PositionEmbeddingSine)
    assert position_embedding.num_pos_feats == 64
    assert position_embedding.temperature == 10000
    assert not position_embedding.normalize
    assert position_embedding.scale == 2 * math.pi
    
def test_sin_position_embedding_forward(position_embedding):
    # Create a dummy tensor and mask for testing
    dummy_data = torch.randn(4, 3, 64, 64)  
    dummy_mask = torch.ones(4, 64, 64, dtype=torch.bool)  
    dummy_tensor_list = NestedTensor(dummy_data, dummy_mask)

    # Call the forward method of PositionEmbeddingSine with the dummy input
    pos_embedding = position_embedding(dummy_tensor_list)
    assert pos_embedding.shape == torch.Size([4, 128, 64, 64])
