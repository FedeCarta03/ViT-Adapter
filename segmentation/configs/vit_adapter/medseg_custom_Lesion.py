# --- EREDITARIETÀ ---
# Usa il modello base DeiT (i pesi che hai scaricato)
_base_ = [
    './upernet_deit_adapter_base_512_160k_ade20k.py'
]

# --- CONFIGURAZIONE DATASET ---
dataset_type = 'SklearnMetricsDataset'
data_root = 'data_medical/27919209/MSLesSeg3C'   # Assicurati che la cartella sia qui

# Normalizzazione standard (la stessa di ImageNet)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Pipeline di Training
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Resize e Crop a 512x512

    # Ridimensiona l'immagine tra metà e raddoppio
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),

    # Fai un ritaglio casuale, ma analizza i pixel. Se una singola classe (ad esempio lo sfondo nero) 
    # occupa più del 75% del tuo ritaglio, buttalo via e fai un nuovo tentativo da un'altra parte
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# Pipeline di Test/Validazione
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
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
        img_dir='images',
        ann_dir='masks_fix',
        split='fold/fold_1/train.txt',
        img_suffix='.png',      # Cerca i PNG
        seg_map_suffix='.png',  # Maschere PNG
        pipeline=train_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks_fix',
        split='fold/fold_1/val.txt',
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
        
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks_fix',
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
    
    ),
    auxiliary_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,   # Usa Group Norm
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