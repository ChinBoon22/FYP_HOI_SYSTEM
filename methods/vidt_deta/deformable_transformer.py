# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------

import copy
import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from util.misc import inverse_sigmoid
from ops.modules import MSDeformAttn
from methods.swin_w_ram import masked_sin_pos_encoding

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DeformableTransformer(nn.Module):
    """ A Deformable Transformer for the neck in a detector

    The transformer encoder is completely removed for ViDT
    Parameters:
        d_model: the channel dimension for attention [default=256]
        nhead: the number of heads [default=8]
        num_decoder_layers: the number of decoding layers [default=6]
        dim_feedforward: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        return_intermediate_dec: whether to return all the indermediate outputs [default=True]
        num_feature_levels: the number of scales for extracted features [default=4]
        dec_n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
        token_label: whether to use the token label loss for training [default=False]. This is an additional trick
            proposed in  https://openreview.net/forum?id=LhbD74dsZFL (ICLR'22) for further improvement
    """

    def __init__(self, d_model=256, nhead=8, use_encoder=False, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1, activation="relu", return_intermediate_dec=True, 
                 num_feature_levels=4, enc_n_points=4, dec_n_points=4, drop_path=0., token_label=False, instance_aware_attn=True,
                 single_branch=False, interleave=False, det_token_num=100, inter_token_num=100, use_bg=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.det_token_num = det_token_num
        self.inter_token_num = inter_token_num
        self.use_encoder = use_encoder
        self.interleave = interleave & use_encoder
        
        if use_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead,
                                                          enc_n_points)
            self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        
        
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points,
                                                        drop_path=drop_path)
        
        if single_branch:
            inter_decoder_layer = None
            instance_aware_attn = None
          
        else:
            inter_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, dec_n_points,
                                                            drop_path=drop_path)
            instance_aware_attn = InteractionLayer(d_model, d_model, dropout) if instance_aware_attn else None
            
        self.decoder = DeformableTransformerDecoder(decoder_layer, inter_decoder_layer, 
                                                    instance_aware_attn, num_decoder_layers, 
                                                    return_intermediate_dec, single_branch=single_branch)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.token_label = token_label

        if self.token_label:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            self.token_embed = nn.Linear(d_model, 91)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.token_embed.bias.data = torch.ones(91) * bias_value
            
        # [DET] and [REL] Token Generation
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, d_model))
        self.rel_det_token = nn.Parameter(torch.zeros(1, inter_token_num, d_model))

        self.det_attn_blocks = nn.Sequential(*nn.ModuleList([
            DetAttentionBlock(det_token_num+inter_token_num,
                dim=d_model, num_heads=8, tokens_norm=True)
            for _ in range(2)]))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        normal_(self.level_embed)
        self.det_token = trunc_normal_(self.det_token, std=.02)
        self.rel_det_token = trunc_normal_(self.rel_det_token, std=.02)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):
        """ The forward step of the decoder

        Parameters:
            srcs: [Patch] tokens from the last 4 layers of the Swin backbone, 
                    list of length 4 with shapes [B, 256, H/8, W/8],
                    [B, 256, H/16, W/16], [B, 256, H/32, W/32], [B, 256, H/64, W/64]
            masks: input padding mask from the last 4 layers of the Swin backbone,
                    list of length 4 with shapes [B, H/8, W/8], [B, H/16, W/16], ...
            tgt: [DET] tokens, shape: [B, 100, 256]
            inter_tgt: [INTER] tokens, shape: [B, 100, 256]
            query_pos: [DET] token pos encodings

        Returns:
            hs: calibrated [DET] tokens
            init_reference_out: init reference points
            inter_references_out: intermediate reference points for box refinement
            enc_token_class_unflat: info. for token labeling
        """

        # prepare input for the transformer encoder/decoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        # Interleave 
        if not self.interleave:
            # Encoder
            if self.use_encoder:
                
                memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
            else:
                memory = src_flatten
                
            # [DET] and [REL] Token Generation
            bs, _, c = memory.shape
            det_token = self.det_token.expand(bs, -1, -1) 
            rel_token = self.rel_det_token.expand(bs, -1, -1) 

            tokens = torch.cat([det_token, rel_token, memory], dim=1)
            tokens = self.det_attn_blocks(tokens)
            tgt = tokens[:,:self.det_token_num]
            inter_tgt = tokens[:,self.det_token_num:self.det_token_num+self.inter_token_num]

            # prepare input for token label
            if self.token_label:
                output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_token_class_unflat = None
            if self.token_label:
                enc_token_class = self.token_embed(output_memory)
                enc_token_class_unflat = []
                for st, (h, w) in zip(level_start_index, spatial_shapes):
                    enc_token_class_unflat.append(enc_token_class[:, st:st+h*w, :].view(bs, h, w, 91))

            reference_points = self.decoder.bbox_embed[-1](tgt).sigmoid()
            sub_reference_points = self.decoder.sub_bbox_embed[-1](tgt).sigmoid()
            inter_reference_points = self.decoder.inter_bbox_embed[-1](inter_tgt).sigmoid()
            inter_vector_points = self.decoder.inter_vector_embed[-1](inter_tgt).sigmoid()
            class_enc = self.decoder.class_embed[-1](tgt)
            verb_enc = self.decoder.verb_embed[-1](inter_tgt)

            init_reference_out = reference_points # query_pos -> reference point
            init_sub_reference_out = sub_reference_points
            init_inter_reference_out = inter_reference_points
            init_inter_vector_out = inter_vector_points

            # decoder
            hs, rels, references, sub_references, inter_references, inter_vectors = self.decoder(tgt, inter_tgt, reference_points, 
                                                                            sub_reference_points, inter_reference_points, 
                                                                            inter_vector_points, memory, 
                                                                            spatial_shapes, level_start_index, 
                                                                            valid_ratios, src_padding_mask=mask_flatten)

        else:
            # [DET] and [REL] Token Generation
            bs, _, c = src_flatten.shape
            det_token = self.det_token.expand(bs, -1, -1) 
            rel_token = self.rel_det_token.expand(bs, -1, -1) 

            tokens = torch.cat([det_token, rel_token, src_flatten], dim=1)
            tokens = self.det_attn_blocks(tokens)
            tgt = tokens[:,:self.det_token_num]
            inter_tgt = tokens[:,self.det_token_num:self.det_token_num+self.inter_token_num]

            # prepare input for token label
            if self.token_label:
                output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_token_class_unflat = None
            if self.token_label:
                enc_token_class = self.token_embed(output_memory)
                enc_token_class_unflat = []
                for st, (h, w) in zip(level_start_index, spatial_shapes):
                    enc_token_class_unflat.append(enc_token_class[:, st:st+h*w, :].view(bs, h, w, 91))

            reference_points = self.decoder.bbox_embed[-1](tgt).sigmoid()
            sub_reference_points = self.decoder.sub_bbox_embed[-1](tgt).sigmoid()
            inter_reference_points = self.decoder.inter_bbox_embed[-1](inter_tgt).sigmoid()
            inter_vector_points = self.decoder.inter_vector_embed[-1](inter_tgt).sigmoid()
            class_enc = self.decoder.class_embed[-1](tgt)
            verb_enc = self.decoder.verb_embed[-1](inter_tgt)

            init_reference_out = reference_points # query_pos -> reference point
            init_sub_reference_out = sub_reference_points
            init_inter_reference_out = inter_reference_points
            init_inter_vector_out = inter_vector_points
          
            # Encoder & Decoder Interleave
            src = src_flatten
            output = tgt # shape: [B, 100, 256]
            inter_output = inter_tgt # shape: [B, 100, 256]
            src_valid_ratios = valid_ratios
            src_spatial_shapes = spatial_shapes
            src_level_start_index = level_start_index
            src_padding_mask = mask_flatten
            
            intermediate = []
            intermediate_rel = []
            intermediate_reference_points = []
            intermediate_sub_reference_points = []
            intermediate_inter_reference_points = []
            intermediate_inter_vector_points = []
            
            enc_reference_points = self.encoder.get_reference_points(spatial_shapes, src_valid_ratios, device=src_flatten.device)
            for i in range(self.encoder.num_layers):
                src = self.encoder.layers[i](src, lvl_pos_embed_flatten, enc_reference_points, spatial_shapes, level_start_index, mask_flatten)
                
                instance_decoder_layer = self.decoder.layers[i]
                if not self.decoder.single_branch:
                    interaction_decoder_layer = self.decoder.inter_decoder_layers[i]
                    instance_aware_attn_layer = self.decoder.inter_layers[i]

                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                    sub_reference_points_input = sub_reference_points[:, :, None] \
                                            * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]  
                    inter_reference_points_input = inter_reference_points[:, :, None] \
                                            * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                    inter_vector_points_input = inter_vector_points[:, :, None] \
                                            * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                    sub_reference_points_input = sub_reference_points[:, :, None] * src_valid_ratios[:, None]
                    inter_reference_points_input = inter_reference_points[:, :, None] * src_valid_ratios[:, None]
                    inter_vector_points_input = inter_vector_points[:, :, None] \
                                            * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]                
                
                # Instance DAB
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
                raw_query_pos = self.decoder.hs_ref_point(query_sine_embed) # bs, nq, 256
                pos_scale = self.decoder.hs_query_scale(output) if i != 0 else 1
                query_pos = pos_scale * raw_query_pos

                output = instance_decoder_layer(output, query_pos, reference_points_input, src, src_spatial_shapes, 
                                                    src_level_start_index, src_padding_mask, sub_reference_points_input)    
            
                if not self.decoder.single_branch:
                    # Interaction DAB
                    inter_query_sine_embed = gen_sineembed_for_position(inter_reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
                    raw_query_pos = self.decoder.rel_hs_ref_point(inter_query_sine_embed) # bs, nq, 256
                    pos_scale = self.decoder.rel_hs_query_scale(inter_output) if i != 0 else 1
                    inter_query_pos = pos_scale * raw_query_pos
                    
                    inter_output = interaction_decoder_layer(inter_output, inter_query_pos, inter_reference_points_input, 
                                                            src, src_spatial_shapes, src_level_start_index, src_padding_mask)
                
                    output, inter_output = instance_aware_attn_layer(output, inter_output)
                else:
                    inter_output = output
                            
                # hack implementation for iterative bounding box refinement
                if self.decoder.bbox_embed is not None:
                    # reference_points = self.refine_sub_obj_bbox(i, output, reference_points)
                    tmp = self.decoder.bbox_embed[i](output)
                    reference_points = self.decoder.refine_bbox(tmp, reference_points)
                    
                if self.decoder.sub_bbox_embed is not None:
                    tmp_sub = self.decoder.sub_bbox_embed[i](output)
                    sub_reference_points = self.decoder.refine_bbox(tmp_sub, sub_reference_points)

                if self.decoder.inter_bbox_embed is not None:
                    tmp_inter = self.decoder.inter_bbox_embed[i](inter_output)
                    inter_reference_points = self.decoder.refine_bbox(tmp_inter, inter_reference_points)
                
                if self.decoder.inter_vector_embed is not None:
                    tmp_inter_vec = self.decoder.inter_vector_embed[i](inter_output)
                    inter_vector_points = self.decoder.refine_bbox(tmp_inter_vec, inter_vector_points)

                if self.decoder.return_intermediate:
                    intermediate.append(output)
                    intermediate_rel.append(inter_output)
                    intermediate_reference_points.append(reference_points)
                    intermediate_sub_reference_points.append(sub_reference_points)
                    intermediate_inter_reference_points.append(inter_reference_points)
                    intermediate_inter_vector_points.append(inter_vector_points)

            if self.decoder.return_intermediate:
                hs = torch.stack(intermediate)
                rels = torch.stack(intermediate_rel)
                references = torch.stack(intermediate_reference_points)
                sub_references = torch.stack(intermediate_sub_reference_points)
                inter_references = torch.stack(intermediate_inter_reference_points)
                inter_vectors = torch.stack(intermediate_inter_vector_points)

        return (hs, rels, init_reference_out, init_sub_reference_out, 
              init_inter_reference_out, init_inter_vector_out, 
              references, sub_references, inter_references, inter_vectors, 
              class_enc, verb_enc, enc_token_class_unflat)
        
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 eta=1e-5):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    @torch.jit.ignore
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod 
    @torch.jit.export
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @torch.jit.ignore
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_dropouts=4, n_norms = 3, drop_path=0.):
        super().__init__()
        self.d_model = d_model
        # self.d_ffn = d_ffn
        self.dropout = dropout
        self.activation=activation
        self.n_levels = n_levels
        self.n_heads = n_heads
        
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        
        self.activation = _get_activation_fn(activation)
        
        for norm_i in range(1, n_norms+1):
            setattr(self, f"norm{norm_i}", nn.LayerNorm(d_model))
            
        for drop_i in range(1, n_dropouts+1):
            setattr(self, f"dropout{drop_i}", nn.Dropout(dropout))
        
        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
        

class DeformableTransformerDecoderLayer(TransformerDecoderLayer):
    """ A decoder layer.

    Parameters:
        d_model: the channel dimension for attention [default=256]
        d_ffn: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        n_levels: the number of scales for extracted features [default=4]
        n_heads: the number of heads [default=8]
        n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
    """

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, drop_path=0.):
        super().__init__(d_model=d_model, d_ffn=d_ffn, 
                         dropout=dropout, activation=activation, n_levels=n_levels, 
                         n_heads=n_heads, drop_path=drop_path)

        # [DET x PATCH] deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # self.cross_attn = DeformableAttention2D(dim=d_model,heads=n_heads,downsample_factor=n_levels,
        #                                         offset_scale=n_points)
    

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, 
                src_padding_mask=None, sub_reference_points=None):

        # [DET] self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Multi-scale deformable cross-attention in Eq. (1) in the ViDT paper
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        
        if sub_reference_points is not None:
            tgt2_sub = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               sub_reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
            tgt2 = tgt2 + tgt2_sub

        if self.drop_path is None:
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # ffn
            tgt = self.forward_ffn(tgt)
        else:
            tgt = tgt + self.drop_path(self.dropout1(tgt2))
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
            tgt = tgt + self.drop_path(self.dropout4(tgt2))
            tgt = self.norm3(tgt)

        return tgt
class DeformableTransformerDecoder(nn.Module):
    """ A Decoder consisting of multiple layers

    Parameters:
        decoder_layer: a deformable decoding layer
        num_layers: the number of layers
        return_intermediate: whether to return intermediate resutls
    """

    def __init__(self, decoder_layer, inter_decoder_layer, 
                 inter_layer, num_layers, return_intermediate=False,
                 single_branch=False):
        super().__init__()
        d_model = decoder_layer.d_model
        activation = decoder_layer.activation
        self.layers = _get_clones(decoder_layer, num_layers)
        if single_branch:
            self.inter_decoder_layers = None
            self.inter_layers = None
        else:
            self.inter_decoder_layers = _get_clones(inter_decoder_layer, num_layers)
            self.inter_layers = _get_clones(inter_layer, num_layers) if inter_layer is not None else None
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        
        # DAB
        self.hs_ref_point = MLP(2 * d_model, d_model, d_model, 4, activation)
        self.hs_query_scale = MLP(d_model, d_model, d_model, 4, activation)
        self.rel_hs_ref_point = MLP(2 * d_model, d_model, d_model, 4, activation)
        self.rel_hs_query_scale = MLP(d_model, d_model, d_model, 4, activation)
        
        # hack implementation for iterative bounding box refinement
        self.bbox_embed = None
        self.sub_bbox_embed = None
        self.inter_bbox_embed = None
        self.inter_vector_embed = None
        self.class_embed = None
        self.verb_embed = None
        
        # whether decoder is single or double branched
        self.single_branch=single_branch
        
    def refine_bbox(self, bbox_inter_pred, reference_points):
        if reference_points.shape[-1] == 4:
            new_reference_points = bbox_inter_pred + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
        else:
            assert reference_points.shape[-1] == 2
            new_reference_points = bbox_inter_pred
            new_reference_points[..., :2] = bbox_inter_pred[..., :2] + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
        reference_points = new_reference_points.detach()
        return reference_points
    
    def refine_sub_obj_bbox(self, index, output, reference_points):
        # NOTE: not used, ignore
        tmp = self.bbox_embed[index](output)
        tmp_sub = self.sub_bbox_embed[index](output)
        
        obj_reference_points = self.refine_bbox(tmp, reference_points)
        sub_reference_points = self.refine_bbox(tmp_sub, reference_points)
        
        reference_points = (obj_reference_points + sub_reference_points) / 2
        return reference_points

    def forward(self, tgt, inter_tgt, reference_points, sub_reference_points, 
                inter_reference_points, inter_vector_points, src, src_spatial_shapes, 
                src_level_start_index, src_valid_ratios, src_padding_mask=None):
        """ The forwared step of the Deformable Decoder

        Parameters:
            tgt: [DET] tokens
            reference_poitns: reference points for deformable attention
            src: the [PATCH] tokens fattened into a 1-d sequence
            src_spatial_shapes: the spatial shape of each multi-scale feature map
            src_level_start_index: the start index to refer different scale inputs
            src_valid_ratios: the ratio of multi-scale feature maps
            query_pos: the pos encoding for [DET] tokens
            src_padding_mask: the input padding mask

        Returns:
            output: [DET] tokens calibrated (i.e., object embeddings)
            reference_points: A reference points

            If return_intermediate = True, output & reference_points are returned from all decoding layers
        """

        output = tgt # shape: [B, 100, 256]
        inter_output = inter_tgt # shape: [B, 100, 256]
        intermediate = []
        intermediate_rel = []
        intermediate_reference_points = []
        intermediate_sub_reference_points = []
        intermediate_inter_reference_points = []
        intermediate_inter_vector_points = []
            
        for i in range(self.num_layers):
            instance_decoder_layer = self.layers[i]
            if not self.single_branch:
                interaction_decoder_layer = self.inter_decoder_layers[i]
                instance_aware_attn_layer = self.inter_layers[i]

            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                sub_reference_points_input = sub_reference_points[:, :, None] \
                                        * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]  
                inter_reference_points_input = inter_reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                inter_vector_points_input = inter_vector_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                sub_reference_points_input = sub_reference_points[:, :, None] * src_valid_ratios[:, None]
                inter_reference_points_input = inter_reference_points[:, :, None] * src_valid_ratios[:, None]
                inter_vector_points_input = inter_vector_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]                

            # Instance DAB
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
            raw_query_pos = self.hs_ref_point(query_sine_embed) # bs, nq, 256
            pos_scale = self.hs_query_scale(output) if i != 0 else 1
            query_pos = pos_scale * raw_query_pos

            output = instance_decoder_layer(output, query_pos, reference_points_input, src, src_spatial_shapes, 
                                                src_level_start_index, src_padding_mask, sub_reference_points_input)    
        
            if not self.single_branch:
                # Interaction DAB
                inter_query_sine_embed = gen_sineembed_for_position(inter_reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
                raw_query_pos = self.rel_hs_ref_point(inter_query_sine_embed) # bs, nq, 256
                pos_scale = self.rel_hs_query_scale(inter_output) if i != 0 else 1
                inter_query_pos = pos_scale * raw_query_pos
                
                inter_output = interaction_decoder_layer(inter_output, inter_query_pos, inter_reference_points_input, 
                                                        src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            
                output, inter_output = instance_aware_attn_layer(output, inter_output)
            else:
                inter_output = output
                        
            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                # reference_points = self.refine_sub_obj_bbox(i, output, reference_points)
                tmp = self.bbox_embed[i](output)
                reference_points = self.refine_bbox(tmp, reference_points)
                
            if self.sub_bbox_embed is not None:
                tmp_sub = self.sub_bbox_embed[i](output)
                sub_reference_points = self.refine_bbox(tmp_sub, sub_reference_points)

            if self.inter_bbox_embed is not None:
                tmp_inter = self.inter_bbox_embed[i](inter_output)
                inter_reference_points = self.refine_bbox(tmp_inter, inter_reference_points)
            
            if self.inter_vector_embed is not None:
                tmp_inter_vec = self.inter_vector_embed[i](inter_output)
                inter_vector_points = self.refine_bbox(tmp_inter_vec, inter_vector_points)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_rel.append(inter_output)
                intermediate_reference_points.append(reference_points)
                intermediate_sub_reference_points.append(sub_reference_points)
                intermediate_inter_reference_points.append(inter_reference_points)
                intermediate_inter_vector_points.append(inter_vector_points)

        if self.return_intermediate:
            return (torch.stack(intermediate), torch.stack(intermediate_rel), 
                    torch.stack(intermediate_reference_points), torch.stack(intermediate_sub_reference_points),
                    torch.stack(intermediate_inter_reference_points),torch.stack(intermediate_inter_vector_points))

        return output, inter_output, reference_points, sub_reference_points

class InteractionLayer(nn.Module):
    """
    Instance-aware attention layer
    """
    def __init__(self, d_model, d_feature, dropout=0.1):
        # defaults
        #   d_model = 256;  d_feature = 256;  dropout = 0.1

        super().__init__()
        self.d_feature = d_feature

        self.det_tfm = nn.Linear(d_model, d_feature)
        self.rel_tfm = nn.Linear(d_model, d_feature)
        self.det_value_tfm = nn.Linear(d_model, d_feature)

        self.rel_norm = nn.LayerNorm(d_model)

        # if use dropout
        if dropout is not None:
            self.dropout = dropout
            self.det_dropout = nn.Dropout(dropout)
            self.rel_add_dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, det_in, rel_in):
        """
        det_in: output of instance branch's i-th decoder layer (shape: [B, 100, 256])
        rel_in: output of interaction branch's i-th decoder layre (shape: [B, 100, 256])
        """ 
        det_attn_in = self.det_tfm(det_in)
        rel_attn_in = self.rel_tfm(rel_in)
        det_value = self.det_value_tfm(det_in) # [B, 100, 256]

        # affinity score map (analogous to Q-K mapping)
        # result shape: [B, 100 (det), 100 (inter)]
        scores = torch.matmul(det_attn_in,
            rel_attn_in.transpose(1, 2)) / math.sqrt(self.d_feature)

        # softmax across the last dimension,
        det_weight = F.softmax(scores.transpose(1, 2), dim = -1) # shape: [B, 100 (inter), 100 (det)]

        if self.dropout is not None:
          det_weight = self.det_dropout(det_weight)
        
        rel_add = torch.matmul(det_weight, det_value) # shape: [B, 100 (inter), 256]
        rel_out = self.rel_add_dropout(rel_add) + rel_in
        rel_out = self.rel_norm(rel_out)

        return det_in, rel_out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.actf = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.actf(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DetAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(self, num_queries, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1., tokens_norm=False):
        super().__init__()
        self.num_queries = num_queries
        self.norm1 = norm_layer(dim)

        self.attn = DetAttn(num_queries,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        self.tokens_norm = tokens_norm

    def forward(self, x):
        x_det = x[:, :self.num_queries]
        x_det = x_det + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x_det = x_det + self.drop_path(self.gamma2 * self.mlp(self.norm2(x_det)))
        return torch.cat((x_det,x[:,self.num_queries:]),dim=1)
      
class DetAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, num_queries, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, :self.num_queries]).unsqueeze(1).reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_det = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_det = self.proj(x_det)
        x_det = self.proj_drop(x_det)
        return x_det
      
# class DetAttn(nn.Module):
#     # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     # with slight modifications to do CA 
#     def __init__(self, num_queries, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
#                  order=5):
#         super().__init__()
#         self.num_queries = num_queries
#         self.order = order
#         self.dims = [dim // 2 ** i for i in range(order)]
#         self.dims.reverse()
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.proj_in = nn.Linear(dim,2*dim)
#         self.dw_linear = nn.Linear(sum(self.dims),sum(self.dims))
        
#         self.pws = nn.ModuleList(
#             [nn.Linear(self.dims[i], self.dims[i+1]) for i in range(order-1)]
#         )
        
#         self.proj_out = nn.Linear(dim,dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
        
#         x = self.proj_in(x).permute(0,2,1)
#         pwa, abc = torch.split(x, (self.dims[0], sum(self.dims)), dim=1)
        
#         dw_abc = self.dw_linear(abc.permute(0,2,1)) * self.scale
#         dw_list = torch.split(dw_abc.permute(0,2,1), self.dims, dim=1) # B, C, N
#         x = (pwa * dw_list[0]).permute(0,2,1) # B, N, C
        
#         for i in range(self.order-1):
#             x = self.pws[i](x) * dw_list[i+1].permute(0,2,1)
            
#         x = self.proj_out(x)
#         x = self.proj_drop(x)
#         x = x[:,:self.num_queries]
#         return x

# class GlobalLocalFilter(nn.Module):
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         self.linear = nn.Linear(dim // 2, dim // 2)
#         self.pre_norm = nn.LayerNorm(dim)
#         self.post_norm = nn.LayerNorm(dim)
      
#     def forward(self,x):
#         x = self.pre_norm(x)
#         x1, x2 = torch.chunk(x, 2, dim=-1)
#         x1 = self.linear(x1) # B, N, C'
        
#         x2 = x2.permute(0,2,1)
#         B, C, N = x2.shape
#         x2 = torch.fft.rfft2(x2, dim=-1, norm='ortho')

    
def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_x, pos_y), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_x, pos_y, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "mish":
        return F.mish
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_deforamble_transformer(args):
    use_iaa = not args.without_iaa
    return DeformableTransformer(
        d_model=args.reduced_dim,
        nhead=args.nheads,
        use_encoder=args.with_encoder,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="gelu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        token_label=args.token_label,
        instance_aware_attn=use_iaa,
        single_branch=args.single_branch_decoder,
        interleave=args.interleave,
        det_token_num=args.det_token_num, 
        inter_token_num=args.inter_token_num,
        use_bg=args.bg_token_num is not None)


