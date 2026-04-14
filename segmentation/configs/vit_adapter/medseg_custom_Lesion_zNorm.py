# --- EREDITARIETÀ ---
_base_ = [
    './upernet_deit_adapter_base_512_160k_ade20k.py'
]

# 1. IMPORTIAMO IL NUOVO SCRIPT ASSIEME ALLA TUA DICE LOSS
custom_imports = dict(imports=['custom_dice', 'zNormlization'], allow_failed_imports=False)

# --- CONFIGURAZIONE DATASET ---
dataset_type = 'SklearnMetricsDataset'
data_root = 'data_medical/27919209/MSLesSeg3C'

# 2. RIMUOVIAMO LA NORMALIZZAZIONE IMAGENET
# img_norm_cfg = dict(mean=[123.675...]) -> CANCELLATO!

# 3. AGGIORNIAMO LA TRAIN PIPELINE
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=15, pad_val=0, seg_pad_val=255),
    
    # LA NUOVA NORMALIZZAZIONE MEDICA VA QUI!
    dict(type='MRILocalZNormalize', clip_values=True),
    
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# 4. AGGIORNIAMO LA TEST PIPELINE
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            
            # LA NUOVA NORMALIZZAZIONE MEDICA VA ANCHE QUI!
            dict(type='MRILocalZNormalize', clip_values=True),
            
            dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    # --- IMPOSTAZIONI CRITICHE PER LAPTOP ---
    samples_per_gpu=1,        # Batch size 1 per evitare crash della memoria (OOM)
    workers_per_gpu=1,        # 1 worker per evitare l'errore ">0"
    persistent_workers=False, # Disattivato per evitare conflitti di memoria
    # ----------------------------------------

    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images100',
        ann_dir='masks_fix100',
        split='fold100/fold_1/train.txt',
        img_suffix='.png',      # Cerca i PNG
        seg_map_suffix='.png',  # Maschere PNG
        pipeline=train_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images100',
        ann_dir='masks_fix100',
        split='fold100/fold_1/val.txt',
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
        
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images100',
        ann_dir='masks_fix100',
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
)

# --- MODIFICA MODELLO (GROUP NORM) ---
# Usiamo GroupNorm (GN) invece di BatchNorm (BN) perché con batch_size=1
# la BatchNorm fallisce matematicamente (divisione per zero/errore statistico).
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    decode_head=dict(
        num_classes=2,      # 2 classi: sfondo + lesione
        norm_cfg=norm_cfg,  # Usa Group Norm
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.1, 0.9]),
            dict(type='CustomDiceLoss', loss_weight=3.0, class_weight=[0.0, 1.0]) # Il peso 3 forza il modello a cercare la lesione
        ]
    
    ),
    auxiliary_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,   # Usa Group Norm
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.1, 0.9]),
            dict(type='CustomDiceLoss', loss_weight=3.0, class_weight=[0.0, 1.0]) # Il peso 3 forza il modello a cercare la lesione
        ]
    )
)

# --- IMPOSTAZIONI RUNNER ---
# Training più breve (20k iterazioni invece di 160k)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000) # Salva ogni 2000 iters
evaluation = dict(interval=2000, metric='mIoU') # Valuta ogni 2000 iters

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])