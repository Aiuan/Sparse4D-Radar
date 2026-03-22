import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmcv.ops import Voxelization
from mmdet.models import DETECTORS, build_backbone, build_neck
from .sparse4d import Sparse4D
from .voxel_encoder import build_voxel_encoder
from .middle_encoder import build_middle_encoder

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["Sparse4DRadar"]


@DETECTORS.register_module()
class Sparse4DRadar(Sparse4D):
    def __init__(
            self,
            img_backbone,
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_backbone,
            head,
            pts_scatter_encoder=None,
            pts_neck=None,
            img_neck=None,
            init_cfg=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            use_grid_mask=True,
            use_deformable_func=False,
            depth_branch=None,
    ):
        super(Sparse4DRadar, self).__init__(
            img_backbone=img_backbone,
            head=head,
            img_neck=img_neck,
            init_cfg=init_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            use_grid_mask=use_grid_mask,
            use_deformable_func=use_deformable_func,
            depth_branch=depth_branch)

        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.pts_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder = build_middle_encoder(pts_middle_encoder)
        self.pts_backbone = build_backbone(pts_backbone)
        if pts_neck:
            self.pts_neck = build_neck(pts_neck)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @force_fp32(apply_to=("img", "points"))
    def forward(self, img, points, **data):
        if self.training:
            return self.forward_train(img, points, **data)
        else:
            return self.forward_test(img, points, **data)

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_img_feat(self, img, return_depth=False, metas=None):
        """Extract features of images."""
        return super(Sparse4DRadar, self).extract_feat(img, return_depth, metas)

    @auto_fp16(apply_to=("points",), out_fp32=True)
    def extract_pts_feat(self, pts):
        """Extract features of points."""
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = x.float()  # for mixed precision training
        x = self.pts_backbone(x)
        if self.pts_neck is not None:
            x = self.pts_neck(x)
        if self.use_deformable_func:
            x = [feat.unsqueeze(1) for feat in x]  # add dummy view_dim
            x = feature_maps_format(x)
        return x

    def extract_feat(self, img, points, data):
        """Extract features from images and points."""
        img_feats, depths = self.extract_img_feat(img, True, data)
        pts_feats = self.extract_pts_feat(points)
        return (img_feats, depths, pts_feats)

    def forward_train(self, img, points, **data):
        img_feats, depths, pts_feats = self.extract_feat(img, points, data)

        data['img'] = img
        data['points'] = points

        model_outs = self.head(img_feats, pts_feats, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, points, **data):
        if isinstance(img, list):
            return self.aug_test(img, points, **data)
        else:
            return self.simple_test(img, points, **data)

    def simple_test(self, img, points, **data):
        img_feats, depths, pts_feats = self.extract_feat(img, points, data)

        data['img'] = img
        data['points'] = points

        model_outs = self.head(img_feats, pts_feats, data)
        results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, points, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], points[0], **data)
