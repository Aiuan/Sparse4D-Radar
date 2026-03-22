# ================ base config ===================
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

total_batch_size = 32
num_gpus = 8
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
num_epochs = 100
checkpoint_epoch_interval = 20

checkpoint_config = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)
log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)
load_from = None
resume_from = None
workflow = [("train", 1)]
# fp16 = dict(loss_scale=32.0)

tracking_test = False
tracking_threshold = 0.2

input_shape = (960, 544)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

point_cloud_range = [-60.0, -40.0, -3.0, 60.0, 40.0, 5.0]
voxel_size = [0.25, 0.25, 8]

# ================== model ========================
class_names = ['car', 'pedestrian', 'rider', 'large_vehicle']

num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = False  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
decouple_attn = True
with_quality_estimation = True

model = dict(
    type="Sparse4DRadar",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    # =================for points feature extract=========================
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=8,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[320, 480]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    # ====================================================================
    head=dict(
        type="Sparse4DRadarHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="omnihd_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=3,
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=(
            [
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * num_single_frame_decoder
            + [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * (num_decoder - num_single_frame_decoder)
        )[2:],
        temp_graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        )
        if temporal
        else None,
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LN"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableFeatureFusion",
            img_feat_dims=embed_dims,
            img_num_groups=num_groups,
            img_num_levels=num_levels,
            num_cams=6,
            use_camera_embed=True,
            pts_feat_dims=384,
            pts_num_groups=num_groups,
            pts_num_levels=1,
            pts_range=point_cloud_range,
            embed_dims=embed_dims,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
            refine_layer=dict(
                type='VelocityConsistencySampling',
                default_similarity=0.5,
                max_radius=0.5,
                sample_num=16,
                embed_dims=embed_dims,
            ),
            fusion_layer=dict(
                type="AdaptiveModalityGating",
                img_dim=embed_dims,
                pts_dim=384,
                out_dim=embed_dims,
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="L1Loss", loss_weight=0.25),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
            loss_yawness=dict(type="GaussianFocalLoss"),
        ),
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

# ================== data ========================
dataset_type = "Omnihd3DDetTrackDataset"
data_root = "data/omnihd/"
anno_root = "data/omnihd_anno_pkls/"
file_client_args = dict(backend="disk")

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles_omnihd", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
        max_depth=60,
    ),
    dict(
        type='LoadRadarPointsMultiSweeps_omnihd',
        load_dim=8,
        sweeps_num=3,
        use_dim=[0, 1, 2, 3, 4, 5, 6, 7],
        file_client_args=file_client_args,
        max_num=40000,
        pc_range=point_cloud_range
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[60] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type='PointShuffle'),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "points",
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_depth",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles_omnihd", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(
        type='LoadRadarPointsMultiSweeps_omnihd',
        load_dim=8,
        sweeps_num=3,
        use_dim=[0, 1, 2, 3, 4, 5, 6, 7],
        file_client_args=file_client_args,
        max_num=40000,
        pc_range=point_cloud_range
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "points",
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 1080,
    "W": 1920,
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    # workers_per_gpu=0,  # single thread for debug
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "omnihd_infos_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "omnihd_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_test,
        tracking_threshold=tracking_threshold,
        # load_interval=1000,  # mini val for debug
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "omnihd_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_test,
        tracking_threshold=tracking_threshold,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=6e-5,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
            "pts_backbone": dict(lr_mult=0.5),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
vis_pipeline = [
    dict(type="LoadMultiViewImageFromFiles_omnihd", to_float32=True),
    dict(
        type='LoadRadarPointsMultiSweeps_omnihd',
        load_dim=8,
        sweeps_num=3,
        use_dim=[0, 1, 2, 3, 4, 5, 6, 7],
        file_client_args=file_client_args,
        max_num=40000,
        pc_range=point_cloud_range
    ),
    dict(
        type="Collect",
        keys=["img", "points"],
        meta_keys=["timestamp", "lidar2img"],
    ),
]
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval,
    pipeline=vis_pipeline,
    # out_dir="./vis",  # for visualization
)
