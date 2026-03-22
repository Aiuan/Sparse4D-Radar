# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)

try:
    from ..ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
    "DeformableFeatureFusion",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
            self,
            embed_dims: int = 256,
            num_groups: int = 8,
            num_levels: int = 4,
            num_cams: int = 6,
            proj_drop: float = 0.0,
            attn_drop: float = 0.0,
            kps_generator: dict = None,
            temporal_fusion_module=None,
            use_temporal_anchor_embed=True,
            use_deformable_func=False,
            use_camera_embed=False,
            residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
            self,
            instance_feature: torch.Tensor,
            anchor: torch.Tensor,
            anchor_embed: torch.Tensor,
            feature_maps: List[torch.Tensor],
            metas: dict,
            **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                    1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
            feature_maps: List[torch.Tensor],
            key_points: torch.Tensor,
            projection_mat: torch.Tensor,
            image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
            self,
            features: torch.Tensor,
            weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_depth_layers=1,
            equal_focal=100,
            max_depth=60,
            loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                        error
                        / max(1.0, len(gt) * len(depth_preds))
                        * self.loss_weight
                )
            loss = loss + _loss
        return loss


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
            self,
            in_channels=None,
            pre_norm=None,
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
            ffn_drop=0.0,
            dropout_layer=None,
            add_identity=True,
            init_cfg=None,
            **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)


@ATTENTION.register_module()
class DeformableFeatureFusion(BaseModule):
    def __init__(
            self,
            #  ==============img==================
            img_feat_dims: int = 256,
            img_num_groups: int = 8,
            img_num_levels: int = 4,
            num_cams: int = 6,
            use_camera_embed=False,
            #  ==============pts==================
            pts_feat_dims: int = 512,
            pts_num_groups: int = 8,
            pts_num_levels: int = 1,
            pts_range: List[float] = None,
            #  ===================================
            embed_dims: int = 256,
            proj_drop: float = 0.0,
            attn_drop: float = 0.0,
            kps_generator: dict = None,
            use_deformable_func=False,
            residual_mode="add",
            #  ===================================
            refine_layer: dict = None,
            fusion_layer: dict = None,
    ):
        super(DeformableFeatureFusion, self).__init__()

        if img_feat_dims % img_num_groups != 0:
            raise ValueError(
                f"img_feat_dims must be divisible by img_num_groups, "
                f"but got {img_feat_dims} and {img_num_groups}"
            )

        if pts_feat_dims % pts_num_groups != 0:
            raise ValueError(
                f"pts_feat_dims must be divisible by pts_num_groups, "
                f"but got {pts_feat_dims} and {pts_num_groups}"
            )

        assert pts_range is not None, "point_cloud_range must be provided."

        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."

        self.img_feat_dims = img_feat_dims
        self.img_num_groups = img_num_groups
        self.img_group_dims = int(img_feat_dims / img_num_groups)
        self.img_num_levels = img_num_levels
        self.num_cams = num_cams
        self.use_camera_embed = use_camera_embed

        self.pts_feat_dims = pts_feat_dims
        self.pts_num_groups = pts_num_groups
        self.pts_group_dims = int(pts_feat_dims / pts_num_groups)
        self.pts_num_levels = pts_num_levels
        self.pts_range = pts_range

        self.embed_dims = embed_dims
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop

        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts

        self.use_deformable_func = use_deformable_func
        self.residual_mode = residual_mode

        img_weights_dims = img_num_groups * img_num_levels * self.num_pts
        pts_weights_dims = pts_num_groups * pts_num_levels * self.num_pts

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
        else:
            self.camera_encoder = None
            img_weights_dims *= num_cams

        self.img_weights = Linear(embed_dims, img_weights_dims)
        self.pts_weights = Linear(embed_dims, pts_weights_dims)

        out_dims = img_feat_dims + pts_feat_dims

        if refine_layer is not None:
            assert not use_deformable_func, "use_deformable_func is not supported when refine_layer is not None."
            from .additional_module import ADDITIONAL_MODULE
            self.refine_layer = build_from_cfg(refine_layer, ADDITIONAL_MODULE)

        if fusion_layer is not None:
            from .additional_module import ADDITIONAL_MODULE
            self.fusion_layer = build_from_cfg(fusion_layer, ADDITIONAL_MODULE)
            out_dims = self.fusion_layer.out_channels

        self.output_proj = Linear(out_dims, embed_dims)

    def init_weight(self):
        constant_init(self.img_weights, val=0.0, bias=0.0)
        constant_init(self.pts_weights, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
            self,
            instance_feature: torch.Tensor,
            anchor: torch.Tensor,
            anchor_embed: torch.Tensor,
            img_feats: List[torch.Tensor],
            pts_feats: List[torch.Tensor],
            metas: dict,
            **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        weights_img, weights_pts = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights_img = (
                weights_img.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.img_num_levels,
                    self.img_num_groups,
                )
            )
            features_img = DAF(*img_feats, points_2d, weights_img).reshape(
                bs, num_anchor, self.img_feat_dims
            )

            points_bev = (
                self.splat_points(
                    key_points,
                    self.pts_range,
                ).unsqueeze(1)  # add dummy view_dim
                .permute(0, 2, 3, 1, 4)
                .view(bs, num_anchor, self.num_pts, 1, 2)
            )
            weights_pts = (
                weights_pts.unsqueeze(2).permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    1,
                    self.pts_num_levels,
                    self.pts_num_groups,
                )
            )
            features_pts = DAF(*pts_feats, points_bev, weights_pts).reshape(
                bs, num_anchor, self.pts_feat_dims
            )
        else:
            features_img = self.img_feature_sampling(
                img_feats,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features_img = self.multi_view_level_fusion(features_img, weights_img)
            features_img = features_img.sum(dim=2)  # fuse multi-point features

            features_pts = self.bev_feature_sampling(
                pts_feats,
                key_points,
                self.pts_range,
            )
            features_pts = self.multi_level_fusion(features_pts, weights_pts)

            if hasattr(self, 'refine_layer'):
                features_pts = self.refine_layer(features_pts, key_points, anchor, metas['points'])

            features_pts = features_pts.sum(dim=2)

        if hasattr(self, 'fusion_layer'):
            features = self.fusion_layer(feat_img=features_img, feat_pts=features_pts)
        else:
            features = torch.cat([features_img, features_pts], dim=-1)

        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed

        shape_pts = [self.pts_num_levels, self.num_pts]
        shape_mask_pts = [1, self.num_pts]

        weights_pts = (
            self.pts_weights(feature)
            .reshape(bs, num_anchor, -1, self.pts_num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                *shape_pts,
                self.pts_num_groups,
            )
        )

        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, *shape_mask_pts, 1,
                device=weights_pts.device, dtype=weights_pts.dtype
            )
            weights_pts = ((mask > self.attn_drop) * weights_pts) / (
                    1 - self.attn_drop
            )

        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights_img = (
            self.img_weights(feature)
            .reshape(bs, num_anchor, -1, self.img_num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.img_num_levels,
                self.num_pts,
                self.img_num_groups,
            )
        )

        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1,
                device=weights_img.device, dtype=weights_img.dtype
            )
            weights_img = ((mask > self.attn_drop) * weights_img) / (
                    1 - self.attn_drop
            )
        return weights_img, weights_pts

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def img_feature_sampling(
            feature_maps: List[torch.Tensor],
            key_points: torch.Tensor,
            projection_mat: torch.Tensor,
            image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureFusion.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        # add valid_mask to consist with CUDA ops version
        # valid_mask = (points_2d[..., 0] > -1) & (points_2d[..., 0] < 1) & \
        #                 (points_2d[..., 1] > -1) & (points_2d[..., 1] < 1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )  # * valid_mask.unsqueeze(1)
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, img_feat_dims

        return features

    def multi_view_level_fusion(
            self,
            features: torch.Tensor,
            weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.img_num_groups, self.img_group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.img_feat_dims
        )
        return features

    @staticmethod
    def splat_points(key_points, point_cloud_range):
        points_bev = key_points[..., :2].clone()  # clone to avoid modifying input
        points_bev[..., 0] = (points_bev[..., 0] - point_cloud_range[0]) / (
                point_cloud_range[3] - point_cloud_range[0])
        points_bev[..., 1] = (points_bev[..., 1] - point_cloud_range[1]) / (
                point_cloud_range[4] - point_cloud_range[1])
        return points_bev

    @staticmethod
    def bev_feature_sampling(
            feature_maps: List[torch.Tensor],
            key_points: torch.Tensor,
            point_cloud_range: List[float],
    ):
        points_bev = DeformableFeatureFusion.splat_points(
            key_points, point_cloud_range
        )

        points_bev = points_bev * 2 - 1

        # add valid_mask to consist with CUDA ops version
        # valid_mask = (points_bev[..., 0] > -1) & (points_bev[..., 0] < 1) & \
        #                 (points_bev[..., 1] > -1) & (points_bev[..., 1] < 1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm, points_bev
                )  # * valid_mask.unsqueeze(1)
            )
        features = torch.stack(features, dim=1)
        features = features.permute(
            0, 3, 1, 4, 2
        )  # bs, num_anchor, num_levels, num_pts, pts_feat_dims

        return features

    def multi_level_fusion(
            self,
            features: torch.Tensor,
            weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.pts_num_groups, self.pts_group_dims)
        )
        features = features.sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.pts_feat_dims
        )

        return features
