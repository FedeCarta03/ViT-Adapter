# --- EREDITARIETÀ ---
_base_ = [
    './upernet_deit_adapter_base_512_160k_ade20k.py'
]

# 1. IMPORTIAMO LA DICE LOSS E LA NUOVA PIPELINE PER I .NPY
# (Abbiamo rimosso zNormlization, non serve più!)
custom_imports = dict(imports=['custom_dice', 'custom_pipeline'], allow_failed_imports=False)

# --- CONFIGURAZIONE DATASET ---
dataset_type = 'SklearnMetricsDataset'
# ATTENZIONE: Assicurati che questo sia il percorso dove lo script precedente ha salvato i dati
data_root = '/home/jacopo/Git/ViT-Adapter/segmentation/data_medical/processed_dataset/train'

# 2. AGGIORNIAMO LA TRAIN PIPELINE
train_pipeline = [
    # Usiamo il nostro loader personalizzato per i file .npy
    dict(type='LoadNumpyFromFile', to_float32=True), 
    dict(type='LoadAnnotations', reduce_zero_label=False),
    
    # Manteniamo le proporzioni corrette senza distorcere l'MRI
    dict(type='Resize', img_scale=(218, 182), keep_ratio=True), 
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=15, pad_val=0, seg_pad_val=255),
    
    # LA NORMALIZZAZIONE È STATA FATTA IN 3D SUI DATI GREZZI, QUINDI QUI NON C'È PIÙ NULLA!
    
    # Riempie i bordi di nero
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255), 
    
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadNumpyFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(218, 182), 
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255), 
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    persistent_workers=False,

    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks_fix',
        split='fold/fold_1/train.txt',
        img_suffix='.npy',      # <--- LEGGE I FILE .NPY
        seg_map_suffix='.png',  # <--- LE MASCHERE RESTANO .PNG
        pipeline=train_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks_fix',
        split='fold/fold_1/val.txt',
        img_suffix='.npy',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
        
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks_fix',
        # Modifica lo split o rimuovilo in fase di test vero e proprio
        split='fold/fold_1/val.txt', 
        img_suffix='.npy',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
)

# --- MODIFICA MODELLO (GROUP NORM) ---
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    decode_head=dict(
        num_classes=2,      
        norm_cfg=norm_cfg,  
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 10.0]),
            dict(type='CustomDiceLoss', loss_weight=1.0, class_weight=[1.0, 10.0]) 
        ]
    ),
    auxiliary_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,   
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 10.0]),
            dict(type='CustomDiceLoss', loss_weight=1.0, class_weight=[1.0, 10.0]) 
        ]
    )
)

# --- IMPOSTAZIONI RUNNER ---
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

# --- OTTIMIZZATORE E LEARNING RATE ---
optimizer = dict(
    type='AdamW', 
    lr=1e-4,  
    weight_decay=0.01
)

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', 
    cumulative_iters=8, 
    grad_clip=dict(max_norm=1.0, norm_type=2)
)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=1e-6,
    by_epoch=False
)