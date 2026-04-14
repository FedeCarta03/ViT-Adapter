import numpy as np
import pickle
import csv
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

def calcola_metriche_pixel(coco_gt_path, mmdet_pkl_path):
    coco = COCO(coco_gt_path)
    with open(mmdet_pkl_path, 'rb') as f:
        predictions = pickle.load(f)
        
    img_ids = coco.getImgIds()
    y_true_all = []
    y_pred_all = []
    
    for idx, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']
        
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            m = coco.annToMask(ann)
            gt_mask = np.maximum(gt_mask, m * ann['category_id']) 
            
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        
        if isinstance(predictions[idx], dict) and 'pred_instances' in predictions[idx]:
            masks = predictions[idx]['pred_instances']['masks']
            labels = predictions[idx]['pred_instances']['labels']
            scores = predictions[idx]['pred_instances']['scores']
            for m, label, score in zip(masks, labels, scores):
                if score > 0.5:
                    m_np = m.cpu().numpy() if hasattr(m, 'cpu') else m
                    pred_mask = np.maximum(pred_mask, m_np * (label + 1))
        else:
            _, img_masks = predictions[idx]

        y_true_all.append(gt_mask.flatten())
        y_pred_all.append(pred_mask.flatten())
        
    y_true_flat = np.concatenate(y_true_all)
    y_pred_flat = np.concatenate(y_pred_all)
    
    # --- TUTTE LE METRICHE DELLA TUA TABELLA ---
    pixel_acc = accuracy_score(y_true_flat, y_pred_flat)
    dice_f1   = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    miou      = jaccard_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    recall    = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    
    return pixel_acc, dice_f1, miou, precision, recall

# ============================================================
# NUOVA PARTE: RACCOLTA DATI E SALVATAGGIO
# ============================================================

risultati_finali = []

for i in range(1, 6):
    print(f"\n>>> Analisi Fold {i} in corso...")
    gt_path = f'dataset/MergeDataset/KFold_Annotations/fold{i}_val.json'
    pkl_path = f'work_dirs/kfoldTrainMet/fold_{i}/results.pkl'
    
    try:
        acc, dice, iou, prec, rec = calcola_metriche_pixel(gt_path, pkl_path)
        
        riga = {
            'Fold': i,
            'Pixel_Accuracy': round(acc, 4),
            'Dice_F1': round(dice, 4),
            'mIoU': round(iou, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4)
        }
        risultati_finali.append(riga)
        print(f"Completato Fold {i} -> Acc: {acc:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
        
    except Exception as e:
        print(f"Errore nel Fold {i}: {e}")

# Calcolo della Media Finale
if risultati_finali:
    mean_acc  = np.mean([r['Pixel_Accuracy'] for r in risultati_finali])
    mean_dice = np.mean([r['Dice_F1'] for r in risultati_finali])
    mean_iou  = np.mean([r['mIoU'] for r in risultati_finali])
    mean_prec = np.mean([r['Precision'] for r in risultati_finali])
    mean_rec  = np.mean([r['Recall'] for r in risultati_finali])
    
    media_row = {
        'Fold': 'MEDIA FINALE',
        'Pixel_Accuracy': round(mean_acc, 4),
        'Dice_F1': round(mean_dice, 4),
        'mIoU': round(mean_iou, 4),
        'Precision': round(mean_prec, 4),
        'Recall': round(mean_rec, 4)
    }
    risultati_finali.append(media_row)

# SCRITTURA SU FILE CSV
nome_file_csv = "report_metriche_kfold_completo.csv"
colonne = ['Fold', 'Pixel_Accuracy', 'Dice_F1', 'mIoU', 'Precision', 'Recall']

with open(nome_file_csv, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=colonne)
    writer.writeheader()
    writer.writerows(risultati_finali)

print(f"\nOperazione completata!")
print(f"Il file con TUTTE le metriche è stato salvato qui: {nome_file_csv}")