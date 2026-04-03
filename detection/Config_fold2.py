# mask_rcnn_vit_adapter_tiny_airleish.py

_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', # Inizia con 1x (12 epoche) per testare
    '../_base_/default_runtime.py'
]

# 1. PESI PREADDRESTRATI (Assicurati che il file sia in questa cartella)
pretrained = 'pretrained/deit_tiny_patch16_224-a1311bcf.pth'

# 2. CONFIGURAZIONE MODELLO (Specifico per Tiny)
model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        patch_size=16,
        embed_dim=192,         # Cambiato per Tiny (Base era 768)
        depth=12,
        num_heads=3,           # Cambiato per Tiny (Base era 12)
        mlp_ratio=4,
        drop_path_rate=0.1,    # Ridotto per modelli piccoli
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[True, True, False, True, True, False,
                     True, True, False, True, True, False],
        window_size=[14, 14, None, 14, 14, None,
                     14, 14, None, 14, 14, None],
        
        pretrained=pretrained),
    neck=dict(
        type='FPN',
        in_channels=[192, 192, 192, 192], # Deve corrispondere a embed_dim
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(num_classes=3),
        mask_head=dict(num_classes=3)
    )
)

# 3. METADATI DATASET
classes = ('AM', 'HC', 'NU')

# 4. PIPELINES (Ottimizzate per microscopia)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True), # Ridotto per stabilitÃ 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# 5. CONFIGURAZIONE DATA (Set1 per Train, Set2 per Val)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=classes,
        ann_file='dataset/AIR_LEISH_dataset/Set2/_annotations.coco.json',
        img_prefix='dataset/AIR_LEISH_dataset/Set2/Images/',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        classes=classes,
        ann_file='dataset/AIR_LEISH_dataset/Set1/_annotations.coco.json',
        img_prefix='dataset/AIR_LEISH_dataset/Set1/Images/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        classes=classes,
        ann_file='dataset/AIR_LEISH_dataset/Set2/_annotations.coco.json',
        img_prefix='dataset/AIR_LEISH_dataset/Set2/Images/',
        pipeline=test_pipeline)
)

# 6. OTTIMIZZATORE
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'level_embed': dict(decay_mult=0.),
            'pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'bias': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)

# 7. RUNTIME & EVALUATION
evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best='auto')
checkpoint_config = dict(interval=1, max_keep_ckpts=3, save_last=True)
dist_params = dict(backend='gloo')