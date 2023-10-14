import pytest
import torch
from torch import nn

import sys
sys.path.insert(0, '/home/cyphi02/MDS01/fyp-weapon-detection')

from methods.vidt.deformable_transformer import DeformableTransformerDecoderLayer\
    , InteractionLayer, DeformableTransformerDecoder, DeformableTransformer, MLP
from test_helper_fns import *
from methods.vidt.hybrid_encoder import CSPRepLayer, HybridEncoder

# defaults
DIM_MODEL = 256
DIM_FFN = 1024
DIM_FEATS=256
DROPOUT=0.1
NLEVELS = 4
NHEADS = 8
NPOINTS = 4
B=4
DET_TOKEN_NUM=100
INTER_TOKEN_NUM=100
POS_DIM=256
ORI_H = 640
ORI_W = 640
NUM_DECODER_LAYERS=6

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
WIDTH = 64

    
def create_deform_trans_layer(d_model=DIM_MODEL, d_ffn=DIM_FFN,
                                dropout=DROPOUT,
                                n_levels=NLEVELS,
                                n_heads=NHEADS,
                                n_points=NPOINTS):
    return DeformableTransformerDecoderLayer(d_model=d_model,
                                            d_ffn=d_ffn,
                                            dropout=dropout,
                                            n_levels=n_levels,
                                            n_heads=n_heads,
                                            n_points=n_points)


def get_spatial_shapes_per_layer(ori_h=ORI_H, ori_w=ORI_W, nlevels=NLEVELS):
    return [[ori_h/(8<<i), ori_w/(8<<i)] for i in range(nlevels)]

def create_input(det_token_num=DET_TOKEN_NUM, model_dim=DIM_MODEL,
                ori_h=ORI_H, ori_w=ORI_W, nlevels=NLEVELS,ref_points_ver=4):
    """
    tgt = target
    query_pos = position embeddings
    spatial_dims_per_layer = Dimensions or shapes of feature maps at different levels
    
    """
    tgt = generate_x((B, det_token_num, model_dim)).to('cuda')
    query_pos = generate_x((B, det_token_num, model_dim)).to('cuda')
    
    if ref_points_ver == 4:  
        ref_points = generate_x(B, det_token_num, 4, 4).to('cuda') # 4 reference points
    else :
        ref_points = generate_x(B, det_token_num, 2).to('cuda') # 1 reference point
        
    spatial_dims_per_layer = get_spatial_shapes_per_layer(nlevels=nlevels, ori_h=ori_h, ori_w=ori_w)
    combined = int(sum([h*w for h, w in spatial_dims_per_layer]))
    
    src= generate_x((B, combined, model_dim)).to('cuda')
    
    src_spatial_shapes = torch.Tensor(spatial_dims_per_layer).type(torch.long).to('cuda')
    combined_tens = torch.Tensor([h*w for h, w in spatial_dims_per_layer])
    cumulative = combined_tens.cumsum(0)
    for i in range(cumulative.shape[-1]-1, 0, -1):
        cumulative[i]=cumulative[i-1]
    cumulative[0]=0
    level_start_idx = cumulative.type(torch.long).to('cuda')
    return tgt, query_pos, ref_points, src, src_spatial_shapes, level_start_idx


def test_deform_trans_decoder_layer():
    """
    Purpose: Test if the decoder layer works under normal conditions and parameters. 
    Condition: No special condition. 
    """
    layer = create_deform_trans_layer()
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    print(src.shape)
    print(src_spatial_shapes.shape)
    print(level_start_index.shape)
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    

def test_deform_trans_decoder_layer_large_dmodel():
    DIM_MODEL = 1024
    layer = create_deform_trans_layer(d_model=DIM_MODEL)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input(model_dim=DIM_MODEL)
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    
def test_deform_trans_decoder_layer_small_dmodel():
    DIM_MODEL = 16
    layer = create_deform_trans_layer(d_model=DIM_MODEL)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input(model_dim=DIM_MODEL)
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    print(tgt.shape)
    

def test_deform_trans_decoder_layer_large_dffn():
    DIM_FFN = 8192
    layer = create_deform_trans_layer(d_ffn=DIM_FFN)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    print(out.shape)
    
def test_deform_trans_decoder_layer_small_dffn():
    DIM_FFN = 16
    layer = create_deform_trans_layer(d_ffn=DIM_FFN)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    
def test_deform_trans_decoder_layer_dropout_all():
    """
    
    """
    layer = create_deform_trans_layer(dropout=1.0)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    
def test_deform_trans_decoder_layer_large_nheads():
    """
    Condition: Number of decoder head is 64, very large.
    """
    layer = create_deform_trans_layer(n_heads=64)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    
def test_deform_trans_decoder_layer_min_nheads():
    """
    Condition: Number of decoder head is 1, very small.
    """
    layer = create_deform_trans_layer(n_heads=1)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    
def create_decoder_input(det_token_num=DET_TOKEN_NUM, inter_token_num=INTER_TOKEN_NUM, 
                            model_dim=DIM_MODEL,
                ori_h=ORI_H, ori_w=ORI_W, nlevels=NLEVELS):
    tgt, query_pos, ref_points, src\
        , src_spatial_shapes, level_start_idx = create_input(det_token_num=det_token_num,
                                                                model_dim=model_dim,
                                                                ori_h=ori_h, ori_w=ori_w, nlevels=nlevels,
                                                                ref_points_ver=2)
    src_valid_ratios = generate_x((B, NLEVELS, 2)).cuda()
    inter_tgt = generate_x((B, inter_token_num, model_dim)).cuda()
    inter_query_pos = generate_x((B, inter_token_num, model_dim)).to('cuda')
    sub_ref_points = ref_points
    src_padding_mask = generate_mask_all0((B, src.shape[1])).bool().cuda()
    return tgt, inter_tgt, query_pos, inter_query_pos, ref_points, sub_ref_points\
        , src, src_spatial_shapes, level_start_idx, src_valid_ratios, src_padding_mask


def test_deform_trans_decoder():
    decoder_layer = create_deform_trans_layer()
    inter_decoder_layer = create_deform_trans_layer()
    decoder = DeformableTransformerDecoder(decoder_layer, inter_decoder_layer, None,
                                            num_layers=NUM_DECODER_LAYERS, single_branch=True).cuda()
    tgt, inter_tgt, query_pos, inter_query_pos, ref_points, sub_ref_points\
        , src, src_spatial_shapes, level_start_idx\
            , src_valid_ratios, src_padding_mask = create_decoder_input()
    
    out = decoder(tgt, inter_tgt, ref_points, sub_ref_points, src, src_spatial_shapes, 
                    level_start_idx, src_valid_ratios,
                    query_pos=query_pos,inter_query_pos=inter_query_pos,
                    src_padding_mask=src_padding_mask)
    assert len(out) == 4
    

def test_deform_trans_decoder():
    decoder_layer = create_deform_trans_layer()
    inter_decoder_layer = create_deform_trans_layer()
    decoder = DeformableTransformerDecoder(decoder_layer, inter_decoder_layer, None,
                                            num_layers=NUM_DECODER_LAYERS, single_branch=True).cuda()
    tgt, inter_tgt, query_pos, inter_query_pos, ref_points, sub_ref_points\
        , src, src_spatial_shapes, level_start_idx\
            , src_valid_ratios, src_padding_mask = create_decoder_input()
    
    
    out = decoder(tgt, inter_tgt, ref_points, sub_ref_points, ref_points, ref_points, src, src_spatial_shapes, level_start_idx, src_valid_ratios,
                    src_padding_mask=src_padding_mask)
    
    assert len(out) == 4
    det_out, inter_out, ref_points_out, sub_ref_points_out = out
    assert list(det_out.shape) == list(tgt.shape)
    assert list(inter_out.shape) == list(inter_tgt.shape)
    assert list(ref_points_out.shape) == list(ref_points.shape)
    assert list(sub_ref_points_out.shape) == list(sub_ref_points.shape)
    

def test_deform_trans_decoder_interm():
    decoder_layer = create_deform_trans_layer()
    inter_decoder_layer = create_deform_trans_layer()
    decoder = DeformableTransformerDecoder(decoder_layer, inter_decoder_layer, None,
                                            num_layers=NUM_DECODER_LAYERS, return_intermediate=True).cuda()
    tgt, inter_tgt, query_pos, inter_query_pos, ref_points, sub_ref_points\
        , src, src_spatial_shapes, level_start_idx\
            , src_valid_ratios, src_padding_mask = create_decoder_input()
    
    out = decoder(tgt, inter_tgt, ref_points, sub_ref_points, query_pos, inter_query_pos, src, src_spatial_shapes, level_start_idx, src_valid_ratios,
                    src_padding_mask=src_padding_mask)
    assert len(out) == 4
    det_out, inter_out, ref_points_out, sub_ref_points_out = out
    assert list(det_out.shape) == [NUM_DECODER_LAYERS+1] + list(tgt.shape)
    assert list(inter_out.shape) == [NUM_DECODER_LAYERS+1] + list(inter_tgt.shape)
    assert list(ref_points_out.shape) == [NUM_DECODER_LAYERS+1] + list(ref_points.shape)
    assert list(sub_ref_points_out.shape) == [NUM_DECODER_LAYERS+1] + list(sub_ref_points.shape)
    
    
def test_MLP():
    INPUT_DIMENSION = 512
    HIDDEN_DIMENSION = OUTPUT_DIMENSION  = 256
    ACTIVATION = nn.ReLU()
    NUM_LAYER = 3
    B = 4
    x = generate_x((B, INPUT_DIMENSION))

    mlp = MLP(input_dim=INPUT_DIMENSION, hidden_dim=HIDDEN_DIMENSION, output_dim=OUTPUT_DIMENSION, num_layers=NUM_LAYER, activation=ACTIVATION)
    assert list(mlp(x).shape) == [B, HIDDEN_DIMENSION]
    
    
    

def create_encoder():
    
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
    
    dummy_input = [generate_x(BATCH, IN_CHANNELS[i], HEIGHT//2**i, WIDTH//2**i) for i in range(len(IN_CHANNELS))]
    # Perform a forward pass
    outputs = hybrid_encoder(dummy_input)
    return outputs

def generate_inter_points(): 
    
    encoder =  create_encoder()
    decoder_layer = create_deform_trans_layer()
    inter_decoder_layer = create_deform_trans_layer()
    decoder = DeformableTransformerDecoder(decoder_layer, inter_decoder_layer, None,
                                            num_layers=NUM_DECODER_LAYERS, return_intermediate=True).cuda()
    tgt, inter_tgt, query_pos, inter_query_pos, ref_points, sub_ref_points\
        , src, src_spatial_shapes, level_start_idx\
            , src_valid_ratios, src_padding_mask = create_decoder_input()
    
    class_enc = decoder.class_embed[-1](encoder)
    verb_enc = decoder.verb_embed[-1](encoder)
    enc_outputs_class = class_enc.sigmoid() * verb_enc.sigmoid()
    _, topk_ind = torch.topk(enc_outputs_class.max(-1)[0], 100, axis=1)
    topk_ind = topk_ind // enc_outputs_class.shape[2]
            
    output = decoder.inter_bbox_embed[-1](encoder).sigmoid()
    print(output.shape)
    inter_reference_points = torch.gather(inter_reference_points, 1, 
                               topk_ind.unsqueeze(-1).repeat(1,1,inter_reference_points.shape[-1]))
    
    print(inter_reference_points.shape)