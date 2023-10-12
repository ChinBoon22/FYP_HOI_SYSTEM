import pytest
import torch
import math
import sys

from vidt.position_encoding import PositionEmbeddingSine

position_embedding = PositionEmbeddingSine(128, normalize=True)
print(position_embedding)