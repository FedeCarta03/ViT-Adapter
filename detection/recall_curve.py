import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

# --- IMPORT CORRETTI PER MMDETECTION ---
import mmdet_custom 
from mmdet.apis import init_detector, inference_detector

def calcola_e_disegna_pr_multifold():
    # ==========================================
    # 1. CONFIGURAZIONE PERCORSI
    # ==========================================
    config_file = 'Config_fold2.py' # Assicurati che vada bene per tutti
    
    # I TUOI PERCORSI REALI
    img_dir = 'dataset/MergeDataset/Immagini'
    json_base_dir = 'dataset/MergeDataset/KFold_Annotations'
    
    colori_fold = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    precisions = []
    aps = []
    # Asse di Recall fisso per l'interpolazione (da 0 a 1 per uniformità)
    mean_recall = np.linspace(0, 1, 100)

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
            
        print(f"\n>>> Elaborazione PR per FOLD {fold_num}...")
        
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
            result = inference_detector(model, str(img_path))
            
            # Creiamo una mappa di probabilità vuota per le predizioni
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

            # Sottocampionamento per gestire la memoria RAM
            y_true_fold.append(gt_flat[::50])
            y_scores_fold.append(prob_flat[::50])

        # --- 3. CALCOLO CURVA PR DEL FOLD ---
        y_true_all = np.concatenate(y_true_fold)
        y_scores_all = np.concatenate(y_scores_fold)
        
        # Forza y_true a essere binario
        y_true_all = np.where(y_true_all > 0, 1, 0)
        
        precision, recall, _ = precision_recall_curve(y_true_all, y_scores_all)
        ap = average_precision_score(y_true_all, y_scores_all)
        aps.append(ap)
        
        # Interpolazione: siccome recall è decrescente, dobbiamo invertirlo per np.interp
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)

        # Disegno curva singola
        plt.plot(recall, precision, color=colori_fold[fold_num-1], lw=1.5, alpha=0.5, 
                 label=f'Fold {fold_num} (AP = {ap:.4f})')
        
        del model
        torch.cuda.empty_cache()

    # ==========================================
    # 4. MEDIA, DEVIAZIONE E SALVATAGGIO
    # ==========================================
    if not precisions:
        print("Errore: Nessun fold elaborato.")
        return

    # Calcolo Baseline (Prevalenza della classe positiva nel test set)
    baseline = np.sum(y_true_all) / len(y_true_all)
    plt.axhline(y=baseline, color='gray', linestyle='--', lw=2, label=f'Baseline ({baseline:.3f})')

    # Media Precision
    mean_precision = np.mean(precisions, axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    plt.plot(mean_recall, mean_precision, color='black', lw=3, 
             label=f'Mean PR (mAP = {mean_ap:.3f} $\pm$ {std_ap:.3f})')

    # Ombra Deviazione Standard
    std_precision = np.std(precisions, axis=0)
    plt.fill_between(mean_recall, np.maximum(mean_precision - std_precision, 0), 
                     np.minimum(mean_precision + std_precision, 1), 
                     color='gray', alpha=0.2, label='$\pm$ 1 std. dev.')

    # Formattazione
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Multi-Fold Precision-Recall Comparison', fontsize=14)
    plt.legend(loc="lower left", fontsize=9, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02]) # Impostato da 0 a 1.02 per vedere l'intera scala. Puoi rimettere 0.4 se preferisci lo zoom.

    plt.savefig('pr_final_kfold.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Grafico salvato come: 'pr_final_kfold.png'")

if __name__ == '__main__':
    calcola_e_disegna_pr_multifold()