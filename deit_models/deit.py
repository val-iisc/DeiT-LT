# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_
from .vision_transformer import VisionTransformer

try:
    from timm.models._registry import register_model
except:
    from timm.models.registry import register_model


__all__ = [
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
    "deit_tiny_distilled_patch16_224",
    "deit_small_distilled_patch16_224",
    "deit_base_distilled_patch16_224",
    "deit_base_patch16_384",
    "deit_base_distilled_patch16_384",
    "deit_large_distilled_patch16_224",
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, crt=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = (
            nn.Linear(self.embed_dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)

    def prepare_tokens(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        # --> New pos encoding
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        return x

    def forward_features(self, x):
        # x = [self.prepare_tokens(x) for x in x_list]
        x = self.prepare_tokens(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)

        cls_token = x
        distill_token = x_dist

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        sim_12 = cos(cls_token, distill_token)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)

        # --> Attention Diversification loss
        # Thresholding from ADL --> top-k for our use-case
        # attn_res = torch.zeros_like(attn_vectors)
        # attn_topk, attn_topk_idx = torch.topk(attn_vectors, k=100, dim=-1)
        # attn_res = attn_res.scatter(-1, attn_topk_idx, attn_topk)

        # Cos-sim b/w attn vectors for CLS and DIS tokens
        # cls_attn = attn_res[:, :, :, 0, :]
        # dis_attn = attn_res[:, :, :, 1, :]
        # avg_attn = torch.mean(attn_res, dim=3)

        # Cos-emb loss w/ margin to avoid -ve sim scores
        # margin = 0.5
        # adl_loss_cls = F.cosine_similarity(cls_attn, avg_attn, dim=-1) - (torch.ones_like(cls_attn[:, :, :, 0]) * margin)
        # adl_loss_dis = F.cosine_similarity(dis_attn, avg_attn, dim=-1) - (torch.ones_like(dis_attn[:, :, :, 0]) * margin)
        # adl_loss_cls[adl_loss_cls < 0] = 0
        # adl_loss_dis[adl_loss_dis < 0] = 0

        # ? Previous loss --> cos-sim, results in -ve sim scores
        # adl_loss_cls = F.cosine_similarity(cls_attn, avg_attn, dim=-1)
        # adl_loss_dis = F.cosine_similarity(dis_attn, avg_attn, dim=-1)
        # adl_loss_heads = torch.sum((adl_loss_cls + adl_loss_dis) / 2, dim=-1)
        # adl_loss = torch.mean(torch.mean(adl_loss_heads, dim=-1), dim=-1)
        adl_loss = 0

        if self.training:
            return x, x_dist, sim_12, adl_loss  # (batch, 10)
        else:
            return x, x_dist

    def copy_dist_qkv(self):
        for i in range(len(self.blocks)):
            # self.blocks[i].attn.qkv_dist = self.blocks[i].attn.qkv
            self.blocks[i].attn.qkv_dist.weight.data = (
                self.blocks[i].attn.qkv.weight.data.clone().detach()
            )
            self.blocks[i].attn.qkv_dist.bias.data = (
                self.blocks[i].attn.qkv.bias.data.clone().detach()
            )
            self.blocks[i].attn.qkv_dist.weight.requires_grad = False
            self.blocks[i].attn.qkv_dist.bias.requires_grad = False


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_large_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model