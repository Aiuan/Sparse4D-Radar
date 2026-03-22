import torch
import torch.nn as nn

from mmcv.ops import QueryAndGroup
from mmcv.cnn import Linear
from mmcv.runner.base_module import Sequential
from mmdet.models.builder import MODELS

ADDITIONAL_MODULE = MODELS


def build_additional_module(cfg):
    """Build scatter encoder."""
    return ADDITIONAL_MODULE.build(cfg)


@ADDITIONAL_MODULE.register_module()
class VelocityConsistencySampling(nn.Module):
    def __init__(self, default_similarity=0.5, max_radius=0.5, sample_num=16, embed_dims=256):
        super().__init__()
        self.default_similarity = default_similarity
        self.max_radius = max_radius
        self.sample_num = sample_num

        self.grouper = QueryAndGroup(max_radius, sample_num)

        self.velocity_encoder = Sequential(
            *linear_relu_ln(embed_dims, 1, 2, 2)
        )

    def forward(self, features_pts, key_points, anchor, points):
        bs, num_anchor, num_keypoints = key_points.shape[:3]

        kps_velocity = anchor[:, :, None, 8:10].repeat(1, 1, num_keypoints, 1)  # vx, vy
        kps_velocity_embed = self.velocity_encoder(kps_velocity)

        grouped_velocity = []
        grouped_valid = []
        for i in range(bs):
            this_points_xyz = points[i][:, :3].unsqueeze(0)
            this_centers_xyz = key_points[i, :, :, :3].view(1, -1, 3)
            this_features = points[i][:, 3:5].permute(1, 0).unsqueeze(0)  # vx, vy

            grouped_features = self.grouper(
                this_points_xyz.contiguous(),
                this_centers_xyz.contiguous(),
                this_features.contiguous()
            )

            mask = torch.norm(
                grouped_features[:, :3, :, :], dim=1
            ) < self.grouper.max_radius

            grouped_features = grouped_features * mask.unsqueeze(1).float()

            velocity_mean = grouped_features[0, 3:5].sum(dim=-1) / mask[0].sum(dim=-1).clamp(min=1.0)
            velocity_mean = velocity_mean.permute(1, 0).view(num_anchor, num_keypoints, 2)
            grouped_velocity.append(velocity_mean)

            veloctiy_valid = mask[0].any(dim=-1)
            veloctiy_valid = veloctiy_valid.view(num_anchor, num_keypoints)
            grouped_valid.append(veloctiy_valid)

        grouped_velocity = torch.stack(grouped_velocity, dim=0)
        grouped_velocity_embed = self.velocity_encoder(grouped_velocity)

        grouped_valid = torch.stack(grouped_valid, dim=0)

        similarity = kps_velocity_embed * grouped_velocity_embed
        similarity = similarity.sum(dim=-1) / (similarity.shape[-1] ** 0.5)
        similarity = similarity.sigmoid()

        similarity_refined = similarity * grouped_valid + self.default_similarity * (~grouped_valid)

        features_pts = features_pts * similarity_refined.unsqueeze(-1)

        return features_pts


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


@ADDITIONAL_MODULE.register_module()
class AdaptiveModalityGating(nn.Module):
    def __init__(self, img_dim, pts_dim, out_dim):
        super().__init__()
        self.img_dim = img_dim
        self.pts_dim = pts_dim
        self.out_dim = out_dim

        self.fc_img = nn.Linear(img_dim, out_dim)
        self.fc_pts = nn.Linear(pts_dim, out_dim)

        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )

        self.fc_out = nn.Linear(out_dim, out_dim)

    @property
    def out_channels(self):
        return self.out_dim

    def forward(self, feat_img, feat_pts):
        f_img = self.fc_img(feat_img)
        f_pts = self.fc_pts(feat_pts)

        gate_map = self.gate(torch.cat([f_img, f_pts], dim=-1))
        fused_feat = gate_map * f_img + (1 - gate_map) * f_pts
        output = self.fc_out(fused_feat)
        return output
