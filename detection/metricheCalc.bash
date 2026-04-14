#!/bin/bash

# Ciclo sui 5 fold
for i in {1..5}
do
    echo "=================================================="
    echo " INIZIO VALUTAZIONE FOLD $i"
    echo "=================================================="
    
    # Adatta questi percorsi alla struttura delle tue cartelle
    CONFIG_FILE="Config_fold2.py"
    CHECKPOINT_FILE="work_dirs/kfoldTrain/fold_${i}/latest.pth"
    OUTPUT_PKL="work_dirs/kfoldTrainMet/fold_${i}/results.pkl"

    python test.py \
        $CONFIG_FILE \
        $CHECKPOINT_FILE \
        --out $OUTPUT_PKL \
        --eval bbox segm \
        --eval-options classwise=True
        
    echo "Valutazione Fold $i completata. Predizioni salvate in $OUTPUT_PKL"
done