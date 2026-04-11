norm_cfg = dict(type='GN', requires_grad=True, num_groups=32)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/deit_base_patch16_224-b5f2ef4d.pth',
    backbone=dict(
        type='ViTAdapter',
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[
            False, False, False, False, False, False, False, False, False,
            False, False, False
        ],
        window_size=[
            None, None, None, None, None, None, None, None, None, None, None,
            None
        ]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='GN', requires_grad=True, num_groups=32),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='CustomDiceLoss', loss_weight=3.0)
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='GN', requires_grad=True, num_groups=32),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='CustomDiceLoss', loss_weight=3.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
dataset_type = 'SklearnMetricsDataset'
data_root = 'data_medical/27919209/MSLesSeg3C'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='SklearnMetricsDataset',
        data_root='data_medical/27919209/MSLesSeg3C',
        img_dir='images',
        ann_dir='masks_fix',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split=
        '/home/jacopo/Git/ViT-Adapter/segmentation/data_medical/27919209/MSLesSeg3C/fold/fold_4/train.txt',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    val=dict(
        type='SklearnMetricsDataset',
        data_root='data_medical/27919209/MSLesSeg3C',
        img_dir='images',
        ann_dir='masks_fix',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split=
        '/home/jacopo/Git/ViT-Adapter/segmentation/data_medical/27919209/MSLesSeg3C/fold/fold_4/val.txt',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    test=dict(
        type='SklearnMetricsDataset',
        data_root='data_medical/27919209/MSLesSeg3C',
        img_dir='images',
        ann_dir='masks_fix',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    persistent_workers=False)
log_config = dict(
    interval=200, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=1)
evaluation = dict(
    interval=2000,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
    work_dir='work_dirs_Lesion_copy/medseg_custom/fold_4')
pretrained = 'pretrained/deit_base_patch16_224-b5f2ef4d.pth'
albu_train_transforms = [
    dict(type='Rotate', limit=60, p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='ElasticTransform',
                alpha=120,
                sigma=6.0,
                alpha_affine=3.5999999999999996,
                p=1.0),
            dict(type='GridDistortion', p=1.0),
            dict(
                type='OpticalDistortion',
                distort_limit=1,
                shift_limit=0.5,
                p=1.0)
        ],
        p=0.3),
    dict(
        type='ColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.5)
]
fp16 = dict(loss_scale=dict(init_scale=512))
custom_imports = dict(imports=['custom_dice'], allow_failed_imports=False)
work_dir = 'work_dirs_Lesion_copy/medseg_custom/fold_4'
gpu_ids = range(0, 1)
auto_resume = False
device = 'cuda'
