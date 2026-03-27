# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
pretrained = 'pretrained/deit_base_patch16_224-b5f2ef4d.pth'
model = dict(
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
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
        window_attn=[False] * 12,
        window_size=[None] * 12),
    decode_head=dict(num_classes=150, in_channels=[768, 768, 768, 768]),
    auxiliary_head=dict(num_classes=150, in_channels=768),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# =========================================================
# Custom Code
# =========================================================
albu_train_transforms = [
    dict(
        type='Rotate',
        limit=60, 
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ElasticTransform', alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            dict(type='GridDistortion', p=1.0),
            dict(type='OpticalDistortion', distort_limit=1, shift_limit=0.5, p=1.0)
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

# =========================================================
# 2. Definisci la TUA nuova Train Pipeline
# =========================================================
# Nota: img_scale=(512, 512) deve corrispondere alla dimensione che vuoi usare
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False), # Metti True se 0 è background da ignorare
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomFlip', prob=0.5),
    
    # QUI INSERIAMO ALBUMENTATIONS
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=None,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# =========================================================
# 3. Collega la nuova pipeline alla configurazione dei dati
# =========================================================
# Questo passaggio è FONDAMENTALE. Senza questo, definisci la pipeline ma non la usi.
data = dict(
    train=dict(
        pipeline=train_pipeline
    )
)
# =========================================================
#
# =========================================================

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2,
          val=dict(pipeline=test_pipeline),
          test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
fp16 = dict(loss_scale=dict(init_scale=512))
