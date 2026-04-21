# --- EREDITARIETÀ ---
_base_ = [
    './upernet_deit_adapter_base_512_160k_ade20k.py'
]

# 1. IMPORTIAMO IL NUOVO SCRIPT ASSIEME ALLA TUA DICE LOSS
custom_imports = dict(imports=['custom_dice', 'zNormlization'], allow_failed_imports=False)

# --- CONFIGURAZIONE DATASET ---
dataset_type = 'SklearnMetricsDataset'
data_root = 'data_medical/27919209/MSLesSeg_FLAIR'

# 2. RIMUOVIAMO LA NORMALIZZAZIONE IMAGENET
# img_norm_cfg = dict(mean=[123.675...]) -> CANCELLATO!

# 3. AGGIORNIAMO LA TRAIN PIPELINE
train_pipeline = [
    # Fondamentale: Carica l'immagine fingendo sia a colori (3 canali uguali)
    dict(type='LoadImageFromFile', color_type='color'), 
    dict(type='LoadAnnotations', reduce_zero_label=False),
    
    # Manteniamo le proporzioni corrette senza distorcere l'MRI
    dict(type='Resize', img_scale=(218, 182), keep_ratio=True), 
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=15, pad_val=0, seg_pad_val=255),
    
    # LA TUA Z-NORM E' SALVA E RIMANE QUI
    dict(type='MRILocalZNormalize', clip_values=True),
    
    # Riempie i bordi di nero
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255), 
    
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'), # 3 canali anche in test!
    dict(
        type='MultiScaleFlipAug',
        img_scale=(218, 182), 
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            
            # LA TUA Z-NORM ANCHE QUI
            dict(type='MRILocalZNormalize', clip_values=True),
            
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255), 
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
        num_classes=2,      
        norm_cfg=norm_cfg,  
        loss_decode=[
            # Pesi neutri: la Cross Entropy non deve sbilanciare nulla
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 10.0]),
            # La Dice Loss pensa alle forme
            dict(type='CustomDiceLoss', loss_weight=1.0, class_weight=[1.0, 10.0]) 
        ]
    ),
    auxiliary_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,   
        loss_decode=[
            # Pesi neutri: la Cross Entropy non deve sbilanciare nulla
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 10.0]),
            # La Dice Loss pensa alle forme
            dict(type='CustomDiceLoss', loss_weight=1.0, class_weight=[1.0, 10.0]) 
        ]
    )
)

# --- IMPOSTAZIONI RUNNER ---
# Training più breve (20k iterazioni invece di 160k)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=1000) # Salva ogni 2000 iters
evaluation = dict(interval=1000, metric='mIoU') # Valuta ogni 2000 iters

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

# --- OTTIMIZZATORE E LEARNING RATE (SOVRASCRITTURA) ---
# Usiamo AdamW. Abbassato ma non troppo (6e-5 invece di 1e-5) per 
# compensare l'accumulo dei gradienti e le 20k iterazioni.
optimizer = dict(
    type='AdamW', 
    lr=1e-4,  
    weight_decay=0.01
)

# Configurazione base per non far esplodere i gradienti
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', 
    cumulative_iters=8, 
    grad_clip=dict(max_norm=1.0, norm_type=2)
)

# --- POLICY DEL LEARNING RATE ---
# Invece di tenere il LR fisso, lo facciamo scendere gradualmente 
# verso lo zero (policy 'poly') man mano che si avvicina a 20.000 iterazioni.
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=3000, # Alzato da 1500 a 3000 iterazioni di riscaldamento
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=1e-6,
    by_epoch=False
)