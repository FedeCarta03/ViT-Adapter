#!/bin/bash

# Definisci il file di configurazione base (inserisci il percorso corretto se non è nella root)
CONFIG="Config_fold2.py"

# Definisci i percorsi base (quelli generati dagli script precedenti)
IMG_DIR="dataset/MergeDataset/Immagini"
ANN_DIR="dataset/MergeDataset/KFold_Annotations"

# Ciclo for da 1 a 5 per i nostri Fold
for FOLD in {1..5}
do
    echo "=========================================================="
    echo "INIZIO TRAINING FOLD $FOLD"
    echo "=========================================================="
    
    # Cartella dove verranno salvati i pesi e i log di questo fold
    WORK_DIR="work_dirs_2/kfoldTrain/fold_${FOLD}"
    
    # Eseguiamo il training utilizzando torch.distributed.launch per evitare il crash
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CV_NUM_THREADS=0 \
    python -m torch.distributed.launch --nproc_per_node=1 train.py $CONFIG --launcher pytorch \
        --work-dir $WORK_DIR \
        --cfg-options data.train.ann_file="${ANN_DIR}/fold${FOLD}_train.json" \
                      data.train.img_prefix="${IMG_DIR}" \
                      data.val.ann_file="${ANN_DIR}/fold${FOLD}_val.json" \
                      data.val.img_prefix="${IMG_DIR}" \
                      data.test.ann_file="${ANN_DIR}/fold${FOLD}_val.json" \
                      data.test.img_prefix="${IMG_DIR}"
                      
    echo "=========================================================="
    echo "FINE TRAINING FOLD $FOLD"
    echo "=========================================================="
    echo ""
done

echo "🎉 TUTTI E 5 I FOLD SONO STATI ADDESTRATI!"