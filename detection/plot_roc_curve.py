import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

# --- IMPORT CORRETTI PER MMDETECTION ---
import mmdet_custom 
from mmdet.apis import init_detector, inference_detector

def calcola_e_disegna_roc_multifold():
    # ==========================================
    # 1. CONFIGURAZIONE PERCORSI
    # ==========================================
    config_file = 'Config_fold2.py' # Assicurati che questo config vada bene
    
    # I TUOI PERCORSI REALI
    img_dir = 'dataset/MergeDataset/Immagini'
    json_base_dir = 'dataset/MergeDataset/KFold_Annotations'
    
    colori_fold = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    # ==========================================
    # 2. CICLO SUI 5 FOLD
    # ==========================================
    for i in range(1, 6):
        fold_num = i
        ckpt_path = f'work_dirs/kfoldTrain/fold_{fold_num}/latest.pth'
        json_path = f'{json_base_dir}/fold{fold_num}_val.json'
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Saltato FOLD {fold_num}: Checkpoint non trovato")
            continue
        if not os.path.exists(json_path):
            print(f"⚠️ Saltato FOLD {fold_num}: JSON non trovato ({json_path})")
            continue
            
        print(f"\n>>> Elaborazione FOLD {fold_num}...")
        
        # Inizializza il modello MMDetection
        model = init_detector(config_file, ckpt_path, device='cuda:0')
        
        # Carica il Ground Truth per questo fold
        coco = COCO(json_path)
        img_ids = coco.getImgIds()
        
        y_true_fold = []
        y_scores_fold = []

        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            nome_file = img_info['file_name']
            img_path = os.path.join(img_dir, nome_file)
            
            if not os.path.exists(img_path):
                print(f"    ⚠️ Immagine mancante: {nome_file}")
                continue

            h, w = img_info['height'], img_info['width']
            
            # --- 1. CREA MAPPA GROUND TRUTH (0 = Sfondo, 1 = Cellula) ---
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                m = coco.annToMask(ann)
                gt_mask = np.maximum(gt_mask, m) 
                
            gt_flat = gt_mask.flatten()
            
            # --- 2. INFERENZA CON MMDET ---
            # Questo restituisce maschere e confidenze
            result = inference_detector(model, str(img_path))
            
            # Creiamo una mappa di probabilità vuota
            prob_map = np.zeros((h, w), dtype=np.float32)
            
            # Estrazione per MMDetection 2.x (tuple)
            if isinstance(result, tuple) and len(result) == 2:
                img_bboxes, img_masks = result
                for class_idx in range(len(img_masks)):
                    bboxes_classe = img_bboxes[class_idx]
                    maschere_classe = img_masks[class_idx]
                    
                    for idx_mask, mask in enumerate(maschere_classe):
                        score = bboxes_classe[idx_mask][4] # Prende la confidenza
                        
                        if isinstance(mask, dict) and 'counts' in mask:
                            m_np = maskUtils.decode(mask).astype(bool)
                        else:
                            m_np = np.array(mask, dtype=bool)
                            
                        # Assegna il livello di confidenza ai pixel della maschera
                        prob_map[m_np] = np.maximum(prob_map[m_np], score)
            
            # Estrazione per MMDetection 3.x
            elif hasattr(result, 'pred_instances'):
                masks = result.pred_instances.masks
                scores = result.pred_instances.scores
                for m, score in zip(masks, scores):
                    m_np = m.cpu().numpy().astype(bool)
                    prob_map[m_np] = np.maximum(prob_map[m_np], score.item())

            prob_flat = prob_map.flatten()

            # Sottocampionamento per non saturare la RAM
            y_true_fold.append(gt_flat[::50])
            y_scores_fold.append(prob_flat[::50])

        # --- 3. CALCOLO CURVA ROC DEL FOLD ---
        y_true_all = np.concatenate(y_true_fold)
        y_scores_all = np.concatenate(y_scores_fold)
        
        # Forza y_true a essere binario (0 o 1)
        y_true_all = np.where(y_true_all > 0, 1, 0)
        
        fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)
        
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        plt.plot(fpr, tpr, color=colori_fold[fold_num-1], lw=1.5, alpha=0.7, 
                 label=f'Fold {fold_num} (AUC = {roc_auc:.4f})')
        
        del model
        torch.cuda.empty_cache()

    # ==========================================
    # 3. DISEGNO MEDIA E SALVATAGGIO
    # ==========================================
    if not tprs:
        print("Errore: Nessun fold elaborato. Controlla i path.")
        return

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Random Guess (AUC = 0.50)')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='black', lw=3, label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})')

    std_tpr = np.std(tprs, axis=0)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), 
                     color='gray', alpha=0.2, label='$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 0.25]) # Zoom
    plt.ylim([0.75, 1.01])  

    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Multi-Fold ROC Comparison (Detection)', fontsize=14)
    plt.legend(loc="lower right", fontsize=9, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig('roc_final_kfold.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Grafico salvato con successo: 'roc_final_kfold.png'")

if __name__ == '__main__':
    calcola_e_disegna_roc_multifold()