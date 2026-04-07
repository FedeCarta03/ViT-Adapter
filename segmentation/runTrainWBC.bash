#!/bin/bash

# Definisci il file di configurazione base
CONFIG="configs/vit_adapter/medseg_custom_copy.py"

# Ciclo for da 1 a 5 per i nostri Fold
for FOLD in {1..5}
do
    echo "=========================================================="
    echo "INIZIO TRAINING FOLD $FOLD"
    echo "=========================================================="
    
    # Cartella dove verranno salvati i pesi e i log di questo fold
    WORK_DIR="work_dirs_WBC/medseg_custom/fold_${FOLD}"
    
    # Eseguiamo il training sovrascrivendo i file di split e la work_dir
    # Eseguiamo il training sovrascrivendo i file di split e la work_dir
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CV_NUM_THREADS=0 python -u train.py $CONFIG \
        --work-dir $WORK_DIR \
        --cfg-options data.train.split="$PWD/data_medical/WbcMSBench_Merged/stratified_folds/fold_${FOLD}/train.txt" \
                      data.val.split="$PWD/data_medical/WbcMSBench_Merged/stratified_folds/fold_${FOLD}/val.txt" \
                      evaluation.work_dir=$WORK_DIR
                      
    echo "=========================================================="
    echo "FINE TRAINING FOLD $FOLD"
    echo "=========================================================="
    echo ""
done

echo "TUTTI E 5 I FOLD SONO STATI ADDESTRATI!"