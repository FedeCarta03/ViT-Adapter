import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

# Import specifici per ViT-Adapter
import mmseg_custom 
from mmseg.apis import init_segmentor
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

def calcola_e_disegna_pr_multifold():
    # --- 1. CONFIGURAZIONE ---
    config_file = 'configs/vit_adapter/medseg_custom.py'
    folds_paths = [f'work_dirs/medseg_custom/fold_{i}/latest.pth' for i in range(1, 6)]
    
    img_dir = 'data_medical/PolypGen/img_dir/test'
    mask_dir = 'data_medical/PolypGen/masks_dir/test'
    
    immagini_test = list(Path(img_dir).glob('*.png'))
    colori_fold = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    precisions = []
    aps = []
    # Creiamo un asse di Recall fisso per l'interpolazione (da 1 a 0)
    mean_recall = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    # --- 2. CICLO SUI 5 FOLD ---
    for idx, ckpt_path in enumerate(folds_paths):
        fold_num = idx + 1
        if not os.path.exists(ckpt_path):
            print(f"Salto FOLD {fold_num}: pesi non trovati.")
            continue
            
        print(f"Elaborazione PR per FOLD {fold_num}...")
        model = init_segmentor(config_file, ckpt_path, device='cuda:0')
        test_pipeline = Compose(model.cfg.data.test.pipeline)
        
        y_true_fold = []
        y_scores_fold = []

        for img_path in immagini_test:
            mask_path = os.path.join(mask_dir, img_path.name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_flat = np.where(gt_mask > 0, 1, 0).flatten()
            
            data = dict(img_info=dict(filename=str(img_path)), img_prefix=None)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            data = scatter(data, ['cuda:0'])[0]

            with torch.no_grad():
                logits = model.encode_decode(data['img'][0], data['img_metas'][0])
                probs = torch.softmax(logits, dim=1)
                prob_lesion = probs[0, 1, :, :].cpu().numpy().flatten()
            
            if prob_lesion.shape != gt_flat.shape:
                 prob_lesion_2d = prob_lesion.reshape(probs.shape[2], probs.shape[3])
                 prob_lesion_2d = cv2.resize(prob_lesion_2d, (gt_mask.shape[1], gt_mask.shape[0]))
                 prob_lesion = prob_lesion_2d.flatten()

            # Sottocampionamento per gestire la memoria
            y_true_fold.append(gt_flat[::50])
            y_scores_fold.append(prob_lesion[::50])

        y_true_all = np.concatenate(y_true_fold)
        y_scores_all = np.concatenate(y_scores_fold)
        
        # Calcolo Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true_all, y_scores_all)
        ap = average_precision_score(y_true_all, y_scores_all)
        aps.append(ap)
        
        # Interpolazione: siccome recall è decrescente nella funzione, dobbiamo invertirlo per interp
        # Usiamo l'asse mean_recall per rendere le curve confrontabili
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)

        # Disegno curva singola
        plt.plot(recall, precision, color=colori_fold[idx], lw=1.5, alpha=0.5, 
                 label=f'Fold {fold_num} (AP = {ap:.4f})')
        
        del model
        torch.cuda.empty_cache()

    # --- 3. MEDIA E DEVIAZIONE ---
    # Calcolo Baseline (Prevalenza della classe positiva nel test set)
    # y_true_all dell'ultimo fold va bene se il test set è fisso
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

    # --- 4. FORMATTAZIONE ---
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Multi-Fold Precision-Recall Comparison - ViT-Adapter', fontsize=14)
    plt.legend(loc="lower left", fontsize=9, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Zoom per focalizzarsi sull'area di performance alta
    plt.xlim([0.0, 1.0])
    plt.ylim([0.4, 1.02]) # Adattato per vedere bene le differenze in alto

    plt.savefig('pr_final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Grafico salvato come: 'pr_final_comparison.png'")

if __name__ == '__main__':
    calcola_e_disegna_pr_multifold()