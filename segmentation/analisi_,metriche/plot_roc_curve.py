import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Import specifici per ViT-Adapter
import mmseg_custom 
from mmseg.apis import init_segmentor
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

def calcola_e_disegna_roc_multifold():
    # --- 1. CONFIGURAZIONE ---
    config_file = 'configs/vit_adapter/medseg_custom.py'
    # Genera i percorsi per i 5 fold
    folds_paths = [f'work_dirs/medseg_custom/fold_{i}/latest.pth' for i in range(1, 6)]
    
    img_dir = 'data_medical/PolypGen/img_dir/test'
    mask_dir = 'data_medical/PolypGen/masks_dir/test'
    
    immagini_test = list(Path(img_dir).glob('*.png'))
    
    # Palette colori ad alto contrasto
    colori_fold = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'] # Rosso, Blu, Verde, Viola, Arancione
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    # --- 2. CICLO SUI 5 FOLD ---
    for idx, ckpt_path in enumerate(folds_paths):
        fold_num = idx + 1
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"Elaborazione FOLD {fold_num}...")
        model = init_segmentor(config_file, ckpt_path, device='cuda:0')
        test_pipeline = Compose(model.cfg.data.test.pipeline)
        
        y_true_fold = []
        y_scores_fold = []

        for img_path in immagini_test:
            # Caricamento Ground Truth
            mask_path = os.path.join(mask_dir, img_path.name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_flat = np.where(gt_mask > 0, 1, 0).flatten()
            
            # Inferenza
            data = dict(img_info=dict(filename=str(img_path)), img_prefix=None)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            data = scatter(data, ['cuda:0'])[0]

            with torch.no_grad():
                logits = model.encode_decode(data['img'][0], data['img_metas'][0])
                probs = torch.softmax(logits, dim=1)
                prob_lesion = probs[0, 1, :, :].cpu().numpy().flatten()
            
            # Resize
            if prob_lesion.shape != gt_flat.shape:
                 prob_lesion_2d = prob_lesion.reshape(probs.shape[2], probs.shape[3])
                 prob_lesion_2d = cv2.resize(prob_lesion_2d, (gt_mask.shape[1], gt_mask.shape[0]))
                 prob_lesion = prob_lesion_2d.flatten()

            y_true_fold.append(gt_flat[::50])
            y_scores_fold.append(prob_lesion[::50])

        # Calcolo metriche fold
        y_true_all = np.concatenate(y_true_fold)
        y_scores_all = np.concatenate(y_scores_fold)
        fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)
        
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        # DISEGNO CURVA FOLD
        plt.plot(fpr, tpr, color=colori_fold[idx], lw=1.5, alpha=0.7, 
                 label=f'Fold {fold_num} (AUC = {roc_auc:.4f})')
        
        del model
        torch.cuda.empty_cache()

    # --- 3. MEDIA, DEVIAZIONE E RANDOM GUESS ---
    # La linea di Random Guess (Diagonale)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Random Guess (AUC = 0.50)')

    # Calcolo Media
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Disegno Media (Nera spessa per contrasto)
    plt.plot(mean_fpr, mean_tpr, color='black', lw=3, label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})')

    # Ombra Deviazione Standard
    std_tpr = np.std(tprs, axis=0)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), 
                     color='gray', alpha=0.2, label='$\pm$ 1 std. dev.')

    # --- 4. ZOOM E FORMATTAZIONE ---
    # Zoom per vedere meglio le curve "schiacciate" in alto
    plt.xlim([-0.01, 0.25]) 
    plt.ylim([0.75, 1.01])  

    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Multi-Fold ROC Comparison - ViT-Adapter Segmentation', fontsize=14)
    plt.legend(loc="lower right", fontsize=9, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig('roc_final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Grafico salvato come: 'roc_final_comparison.png'")

if __name__ == '__main__':
    calcola_e_disegna_roc_multifold()