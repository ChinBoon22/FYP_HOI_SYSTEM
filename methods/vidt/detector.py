# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------
"""Build a VIDT detector for object detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import (nested_tensor_from_tensor_list,
                        inverse_sigmoid, NestedTensor, Conv2d)
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessor import PostProcess, PostProcessSegm, PostProcessHOI
from .deformable_transformer import build_deforamble_transformer
from methods.vidt.fpn_fusion import FPNFusionModule
from methods.vidt.hybrid_encoder import HybridEncoder
import copy
import math
from .dct import ProcessorDCT
from .efficientnetv2 import build_efficientnetv2

def _get_clones(module, N):
    """ Clone a moudle N times """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Detector(nn.Module):
    """ This is a combination of "Swin with RAM" and a "Neck-free Deformable Decoder" """

    def __init__(self, backbone, transformer, num_classes, num_verbs, num_queries, num_inter_queries,
                 num_feature_levels, aux_loss=False, 
                #  with_box_refine=False,
                with_obj_box_refine=False, with_sub_box_refine=False,
                with_interaction_box_refine=False, with_interaction_vector_refine=False,
                 # The three additional techniques for ViDT+
                 epff=None,# (1) Efficient Pyramid Feature Fusion Module
                 with_vector=False, processor_dct=None, vector_hidden_dim=256,# (2) UQR Module
                 iou_aware=False, token_label=False, # (3) Additional losses
                 distil=False, predict_interaction_vector=False, 
                 weight_standardization=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries (i.e., det tokens). This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            epff: None or fusion module available
            iou_aware: True if iou_aware is to be used.
              see the original paper https://arxiv.org/abs/1912.05992
            token_label: True if token_label is to be used.
              see the original paper https://arxiv.org/abs/2104.10858
            distil: whether to use knowledge distillation with token matching
        """

        super().__init__()
        self.num_queries = num_queries
        self.num_inter_queries = num_inter_queries
        self.transformer = transformer
        self.num_feature_levels = num_feature_levels
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.verb_embed = nn.Linear(hidden_dim, num_verbs)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.interaction_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.interaction_vector_embed = MLP(hidden_dim, hidden_dim, 4, 3) if predict_interaction_vector else None
        
        self.backbone = backbone

        # two essential techniques used [default use]
        self.aux_loss = aux_loss
        # self.with_box_refine = with_box_refine
        self.with_obj_box_refine=with_obj_box_refine
        self.with_sub_box_refine=with_sub_box_refine

        self.conv_layer = self._get_conv_layer(weight_standardization)

        # For UQR module for ViDT+
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.processor_dct.n_keep, 3)

        ############ Modified for ViDT+
        # For two additional losses for ViDT+
        self.iou_aware = iou_aware
        self.token_label = token_label

        # distillation
        self.distil = distil

        # For EPFF module for ViDT+
        if epff is None:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            # for each of the multi-scale output features from the backbone
            # standardises the num channels which varies for each one to the transformer hidden dim, 256
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                  # This is 1x1 conv -> so linear layer
                #   nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                  self.conv_layer()(in_channels, hidden_dim, kernel_size=1),
                  
                  nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs + 1):
                input_proj_list.append(nn.Sequential(
                  # This is 1x1 conv -> so linear layer
                #   nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                  self.conv_layer()(in_channels, hidden_dim, kernel_size=1),
                  
                  nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)

            # initialize the projection layer for [PATCH] tokens
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            self.fusion = None
        else:
            # the cross scale fusion module has its own reduction layers
            self.fusion = epff
        ############

        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.verb_embed.bias.data = torch.ones(num_verbs) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.interaction_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.interaction_bbox_embed.layers[-1].bias.data, 0)

        ############ Added for UQR
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)
        ############
        
        if self.interaction_vector_embed:
            nn.init.constant_(self.interaction_vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.interaction_vector_embed.layers[-1].bias.data, 0)

        # the prediction is made for each decoding layers + the standalone detector (Swin with RAM)
        num_pred = transformer.decoder.num_layers + 1

        # set up all required nn.Module for additional techniques
        # if with_box_refine:
        if with_obj_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.class_embed = self.class_embed
            self.transformer.decoder.bbox_embed = self.bbox_embed
        if with_sub_box_refine:
            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred)
            nn.init.constant_(self.sub_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.sub_bbox_embed = self.sub_bbox_embed
        if with_interaction_box_refine:
            self.verb_embed = _get_clones(self.verb_embed, num_pred)
            self.interaction_bbox_embed = _get_clones(self.interaction_bbox_embed, num_pred)
            nn.init.constant_(self.interaction_bbox_embed[0].layers[-1].bias.data[2:], -2.0)            
            self.transformer.decoder.verb_embed = self.verb_embed
            self.transformer.decoder.inter_bbox_embed = self.interaction_bbox_embed
        if with_interaction_vector_refine:
            self.interaction_vector_embed = _get_clones(self.interaction_vector_embed, num_pred)
            nn.init.constant_(self.interaction_vector_embed[0].layers[-1].bias.data[2:], -2.0)            
            self.transformer.decoder.inter_vector_embed = self.interaction_vector_embed

        if not with_obj_box_refine and not with_sub_box_refine: # no bbox refinement
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
            
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.verb_embed = nn.ModuleList([self.verb_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
            self.transformer.decoder.sub_bbox_embed = None

        ############ Added for UQR
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])
        ############

        if self.iou_aware:
            self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            # if with_box_refine:
            if with_obj_box_refine:
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            else:
                self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])

    def _get_conv_layer(self, weight_standardization=False):
        def _conv_layer():
            return Conv2d if weight_standardization else nn.Conv2d
        return _conv_layer

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward step of ViDT

        Parameters:
            The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dictionary having the key and value pairs below:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
                            If iou_aware is True, "pred_ious" is also returns as one of the key in "aux_outputs"
            - "enc_tokens": If token_label is True, "enc_tokens" is returned to be used

            Note that aux_loss and box refinement is used in ViDT in default.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # return multi-scale [PATCH] tokens along with final [DET] and [INTER] tokens and their pos encodings
        features, pos = self.backbone(samples)
        
        # [PATCH] token projection
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            features[l] = src
            masks.append(mask)

        srcs = []
        if self.fusion is None:
            for l, src in enumerate(features):
                srcs.append(self.input_proj[l](src))
                
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1])
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    srcs.append(src)
                    masks.append(mask)   
                  
        else:
            # EPFF (multi-scale fusion) is used if fusion is activated
            srcs = self.fusion(features)
            # srcs.reverse()
        
        if self.num_feature_levels < len(srcs):
          feature_level = len(srcs)-self.num_feature_levels
          srcs = srcs[feature_level:]
          masks = masks[feature_level:]
          pos = pos[feature_level:]
          
        # return the output of the neck-free decoder
        # hs shape: [7, B, 100, 256]
        # rels shape: [7, B, 100, 256]
        hs, rels, init_reference, init_sub_reference, init_inter_reference, init_inter_vector, \
        references, sub_references, inter_references, inter_vectors, class_enc, verb_enc, enc_token_class_unflat = \
                    self.transformer(srcs, masks, pos)

        outputs_classes = []
        outputs_coords = []
        outputs_sub_coords = []
        outputs_verbs = []
        outputs_interaction_vectors = []
        outputs_interaction_coords = []

        # perform predictions via the detection head
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else references[lvl - 1]
            sub_reference = init_sub_reference if lvl == 0 else sub_references[lvl-1]
            inter_reference = init_inter_reference if lvl == 0 else inter_references[lvl - 1]
            inter_vector = init_inter_vector if lvl == 0 else inter_vectors[lvl - 1]
      
            # inter_reference
            reference = inverse_sigmoid(reference)
            sub_reference = inverse_sigmoid(sub_reference)
            inter_reference = inverse_sigmoid(inter_reference)
            inter_vector = inverse_sigmoid(inter_vector)

            outputs_class = self.class_embed[lvl](hs[lvl])
            verb_class = self.verb_embed[lvl](rels[lvl])
          
            ## bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp_sub = self.sub_bbox_embed[lvl](hs[lvl])
            tmp_inter = self.interaction_bbox_embed[lvl](rels[lvl])
            
            if reference.shape[-1] == 4:
                tmp += reference
                tmp_sub += sub_reference
                tmp_inter += inter_reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp_sub[..., :2] += sub_reference
                tmp_inter[..., :2] += inter_reference
            
            if self.interaction_vector_embed:
                tmp_inter_vec = self.interaction_vector_embed[lvl](rels[lvl])
                tmp_inter_vec += inter_vector
                outputs_inter_vec = tmp_inter_vec.sigmoid()
                outputs_interaction_vectors.append(outputs_inter_vec)

            outputs_coord = tmp.sigmoid()
            sub_coord = tmp_sub.sigmoid()
            outputs_interaction_coord = tmp_inter.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_verbs.append(verb_class)
            outputs_coords.append(outputs_coord)
            outputs_sub_coords.append(sub_coord)
            outputs_interaction_coords.append(outputs_interaction_coord)

        # stack all predictions made from each decoding layers
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_sub_coord = torch.stack(outputs_sub_coords)
        outputs_verb = torch.stack(outputs_verbs)
        outputs_interaction_coord = torch.stack(outputs_interaction_coords)
        if self.interaction_vector_embed:
            outputs_interaction_vector = torch.stack(outputs_interaction_vectors)

        ############ Added for UQR
        outputs_vector = None
        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)
        ############

        # final prediction is made the last decoding layer
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 
               'pred_verb_logits': outputs_verb[-1], 'pred_sub_boxes': outputs_sub_coord[-1],
               'pred_interaction_boxes': outputs_interaction_coord[-1]}
        
        # encoder loss
        out['enc_outputs'] = {'pred_logits': class_enc, 'pred_boxes': init_reference, 
                          'pred_verb_logits': verb_enc, 'pred_sub_boxes': init_sub_reference,
                          'pred_interaction_boxes': init_inter_reference}

        if self.interaction_vector_embed:
            out.update({'pred_interaction_vectors': outputs_interaction_vector[-1]})
            out['enc_outputs']['pred_interaction_vectors'] = init_inter_vector

        ############ Added for UQR
        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})

        ############

        # aux loss is defined by using the rest predictions
        if self.aux_loss and self.transformer.decoder.num_layers > 0:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_verb, 
                                                    outputs_sub_coord, outputs_vector, 
                                                    outputs_interaction_coord, outputs_interaction_vector)

        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.iou_aware:
            outputs_ious = []
            for lvl in range(hs.shape[0]):
                outputs_ious.append(self.iou_embed[lvl](hs[lvl]))
            outputs_iou = torch.stack(outputs_ious)
            out['pred_ious'] = outputs_iou[-1]

            if self.aux_loss:
                for i, aux in enumerate(out['aux_outputs']):
                    aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

        if self.distil:
            # 'patch_token': multi-scale patch tokens from each stage
            # 'body_det_token' and 'neck_det_tgt': the input det_token for multiple detection heads
            out['distil_tokens'] = {'patch_token': srcs, 'body_det_token': det_tgt, 'neck_det_token': hs}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_verb, outputs_sub_coord, outputs_vector=None, outputs_interaction_coord=None, outputs_interaction_vector=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        
        if outputs_vector is not None:
            raise NotImplementedError("not supposed to be activated")
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_interaction_boxes': c, 'pred_vectors': d}
                    for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_interaction_coord[:-1], outputs_vector[:-1])]
        elif len(outputs_interaction_vector) > 0:
            return [{'pred_logits': obj_cls, 'pred_boxes': obj_coord, 
                     'pred_verb_logits': rel_verb, 'pred_sub_boxes': sub_coord,
                     'pred_interaction_boxes': inter_coord, 'pred_interaction_vectors': inter_vec}
                    for obj_cls, obj_coord, rel_verb, sub_coord, inter_coord, inter_vec in zip(outputs_class[:-1], outputs_coord[:-1], 
                                          outputs_verb[:-1], outputs_sub_coord[:-1], outputs_interaction_coord[:-1],
                                          outputs_interaction_vector[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b, 
                     'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_interaction_boxes': e}
                    for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1], outputs_verb[:-1], 
                                              outputs_sub_coord[:-1], outputs_interaction_coord[:-1])]


class MLP(nn.Module):
  """ Very simple multi-layer perceptron (also called FFN)"""

  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
      super().__init__()
      self.num_layers = num_layers
      h = [hidden_dim] * (num_layers - 1)
      self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

  def forward(self, x):
      for i, layer in enumerate(self.layers):
          x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
      return x

def build(args, is_teacher=False):

    # distillation is deprecated
    if is_teacher:
        print('Token Distillation is deprecated in this version. Please use the previous version of ViDT.')
    assert is_teacher is False
    # a teacher model for distilation
    #if is_teacher:
    #    return build_teacher(args)
    ############################

    if args.dataset_file == 'coco':
        num_classes = 91

    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    if 'edgenext' in args.backbone_name:
        backbone = build_backbone(args)
    elif 'efficientnetv2' in args.backbone_name:
        backbone = build_efficientnetv2(args)
    elif 'container' in args.backbone_name:
        backbone = build_container(args)
    elif 'maxvit' in args.backbone_name:
        backbone = build_maxvit(args)
    elif 'hornet' in args.backbone_name:
        backbone = build_hornet(args)
    elif 'convnextv2' in args.backbone_name:
        backbone = build_convnextv2(args)
    elif 'convnext' in args.backbone_name:
        backbone = build_convnext(args)
    elif 'davit' in args.backbone_name:
        backbone = build_davit(args)
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')
    # print(next(backbone.parameters()).is_cuda)
    # backbone.finetune_det(method=args.method,
    #                       det_token_num=args.det_token_num,
    #                       inter_token_num=args.inter_token_num,
    #                       pos_dim=args.reduced_dim,
    #                       cross_indices=args.cross_indices,
    #                       inter_cross_indices=args.inter_cross_indices,
    #                       inter_patch_cross_indices=args.inter_patch_cross_indices,
    #                       bg_token_num=args.bg_token_num)

    epff = None
    if args.epff:
        # epff = FPNFusionModule(backbone.num_channels, fuse_dim=args.reduced_dim, n_block=4)
        epff = HybridEncoder(in_channels=backbone.num_channels, 
                             hidden_dim=args.reduced_dim)

    deform_transformers = build_deforamble_transformer(args)

    # Added for UQR module
    if args.with_vector:
        processor_dct = ProcessorDCT(args.n_keep, args.gt_mask_len)

    model = Detector(
        backbone,
        deform_transformers,
        num_classes=args.num_obj_classes,
        num_verbs = args.num_verb_classes,
        num_queries=args.det_token_num,
        num_inter_queries=args.inter_token_num,
        num_feature_levels=args.num_feature_levels,
        # two essential techniques used in ViDT
        aux_loss=args.aux_loss,
        # with_box_refine=args.with_box_refine,
        with_obj_box_refine=args.with_obj_box_refine,
        with_sub_box_refine=args.with_sub_box_refine, 
        with_interaction_box_refine=args.with_obj_box_refine,
        with_interaction_vector_refine=args.with_sub_box_refine, 
        # an epff module for ViDT+
        epff=epff,
        # an UQR module for ViDT+
        with_vector=args.with_vector,
        processor_dct=processor_dct if args.with_vector else None,
        # two additional losses for VIDT+
        iou_aware=args.iou_aware,
        token_label=args.token_label,
        vector_hidden_dim=args.vector_hidden_dim,
        # distil
        distil=False if args.distil_model is None else True,
        predict_interaction_vector=args.predict_interaction_vector,
        weight_standardization=args.weight_standardization,
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_obj_ce': args.obj_cls_loss_coef, 'loss_verb_ce': args.verb_cls_loss_coef,
                   'loss_sub_bbox': args.bbox_loss_coef, 'loss_obj_bbox': args.bbox_loss_coef,
                   'loss_sub_giou': args.giou_loss_coef, 'loss_obj_giou': args.giou_loss_coef, 
                   'loss_verb_bbox': args.bbox_loss_coef, 'loss_verb_giou': args.giou_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.predict_interaction_vector:
        weight_dict['loss_verb_vec'] = args.verb_vec_loss_coef
    ##
    if args.iou_aware:
        weight_dict['loss_iouaware'] = args.iouaware_loss_coef

    if args.token_label:
        weight_dict['loss_token_focal'] = args.token_loss_coef
        weight_dict['loss_token_dice'] = args.token_loss_coef

    # For UQR module
    if args.masks:
        weight_dict["loss_vector"] = 1

    if args.distil_model is not None:
        weight_dict['loss_distil'] = args.distil_loss_coef

    # aux decoding loss
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1 + 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'verbs', 'interaction_boxes']
    if args.predict_interaction_vector:
        # FIXME: once interaction vector loss has been implemented, uncomment below
        losses += ["interaction_vector"]
        pass
    
    if args.iou_aware:
        losses += ['iouaware']

    # For UQR
    if args.masks:
        losses += ["masks"]

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(args.num_obj_classes, matcher, weight_dict, losses,
                             focal_alpha=args.focal_alpha,
                             # For UQR
                             with_vector=args.with_vector,
                             processor_dct=processor_dct if args.with_vector else None,
                             vector_loss_coef=args.vector_loss_coef,
                             no_vector_loss_norm=args.no_vector_loss_norm,
                             vector_start_stage=args.vector_start_stage)
    criterion.to(device)
    # HER -=> post도 고쳐야함.
    # postprocessors = {'bbox': PostProcess(processor_dct=processor_dct if (args.with_vector) else None)}
    # if args.masks:
    #     postprocessors['segm'] = PostProcessSegm(processor_dct=processor_dct if args.with_vector else None)
    postprocessor = PostProcessHOI(args.subject_category_id)
    return model, criterion, postprocessor

''' deprecated
def build_teacher(args):

    if args.dataset_file == 'coco':
        num_classes = 91

    if args.dataset_file == "coco_panoptic":
        num_classes = 250

    if args.distil_model == 'vidt_nano':
        backbone, hidden_dim = swin_nano()
    elif args.distil_model == 'vidt_tiny':
        backbone, hidden_dim = swin_tiny()
    elif args.distil_model == 'vidt_small':
        backbone, hidden_dim = swin_small()
    elif args.distil_model == 'vidt_base':
        backbone, hidden_dim = swin_base_win7()
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    backbone.finetune_det(method=args.method,
                          det_token_num=args.det_token_num,
                          pos_dim=args.reduced_dim,
                          cross_indices=args.cross_indices)

    cross_scale_fusion = None
    if args.cross_scale_fusion:
        cross_scale_fusion = FPNFusionModule(backbone.num_channels, fuse_dim=args.reduced_dim, all=args.cross_all_out)

    deform_transformers = build_deforamble_transformer(args)

    model = Detector(
        backbone,
        deform_transformers,
        num_classes=num_classes,
        num_queries=args.det_token_num,
        # two essential techniques used in ViDT
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        # three additional techniques (optionally)
        cross_scale_fusion=cross_scale_fusion,
        iou_aware=args.iou_aware,
        token_label=args.token_label,
        # distil
        distil=False if args.distil_model is None else True,
    )

    return model
'''