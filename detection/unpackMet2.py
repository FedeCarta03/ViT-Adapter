import numpy as np
import pickle
import csv
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

# ==========================================
# CONFIGURAZIONE
# ==========================================
NOME_FILE_CSV = "report_metriche_indipendenti.csv"

def calcola_metriche_pixel(coco_gt_path, mmdet_pkl_path):
    coco = COCO(coco_gt_path)
    with open(mmdet_pkl_path, 'rb') as f:
        predictions = pickle.load(f)
        
    cat_ids = coco.getCatIds()
    
    # Creiamo dei contatori separati per ogni singola classe
    # Invece di sovrapporre tutto, contiamo i pixel giusti (intersezione) e totali (unione)
    stats = {cid: {'inter': 0, 'union': 0, 'gt_sum': 0, 'pred_sum': 0} for cid in cat_ids}
    
    img_ids = coco.getImgIds()
    
    for idx, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']
        
        # --- 1. MASCHERE GROUND TRUTH (Un livello per classe) ---
        gt_masks = {cid: np.zeros((h, w), dtype=bool) for cid in cat_ids}
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            m = coco.annToMask(ann).astype(bool)
            # Aggiungiamo la maschera solo al livello della sua classe
            gt_masks[ann['category_id']] = np.logical_or(gt_masks[ann['category_id']], m)
            
        # --- 2. MASCHERE PREDIZIONE (Un livello per classe) ---
        pred_masks = {cid: np.zeros((h, w), dtype=bool) for cid in cat_ids}
        
        # Estrazione per MMDetection 3.x
        if isinstance(predictions[idx], dict) and 'pred_instances' in predictions[idx]:
            masks = predictions[idx]['pred_instances']['masks']
            labels = predictions[idx]['pred_instances']['labels']
            scores = predictions[idx]['pred_instances']['scores']
            for m, label, score in zip(masks, labels, scores):
                if score > 0.5:
                    m_np = m.cpu().numpy() if hasattr(m, 'cpu') else m
                    id_reale_classe = cat_ids[label]
                    pred_masks[id_reale_classe] = np.logical_or(pred_masks[id_reale_classe], m_np.astype(bool))
                    
        # Estrazione per MMDetection 2.x
        else:
            img_bboxes, img_masks = predictions[idx]
            for class_idx in range(len(img_masks)):
                bboxes_classe = img_bboxes[class_idx]
                maschere_classe = img_masks[class_idx]
                id_reale_classe = cat_ids[class_idx]
                
                for i, mask in enumerate(maschere_classe):
                    score = bboxes_classe[i][4]
                    if score > 0.5:
                        if isinstance(mask, dict) and 'counts' in mask:
                            m_np = maskUtils.decode(mask).astype(bool)
                        else:
                            m_np = np.array(mask, dtype=bool)
                        pred_masks[id_reale_classe] = np.logical_or(pred_masks[id_reale_classe], m_np)

        # --- 3. CONTEGGIO PIXEL PER CLASSE ---
        for cid in cat_ids:
            gt = gt_masks[cid]
            pred = pred_masks[cid]
            
            # Quanti pixel si sovrappongono esattamente? (True Positive)
            inter = np.logical_and(gt, pred).sum()
            # Quanti pixel occupa l'area totale disegnata da modello + realtà?
            union = np.logical_or(gt, pred).sum()
            
            stats[cid]['inter'] += inter
            stats[cid]['union'] += union
            stats[cid]['gt_sum'] += gt.sum()
            stats[cid]['pred_sum'] += pred.sum()
            
    # --- 4. CALCOLO METRICHE MATEMATICHE ---
    risultati_fold = {}
    for i, cid in enumerate(cat_ids):
        inter = stats[cid]['inter']
        union = stats[cid]['union']
        gt_sum = stats[cid]['gt_sum']
        pred_sum = stats[cid]['pred_sum']
        
        # Formule esatte
        iou = inter / union if union > 0 else 0.0
        dice = (2 * inter) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0.0
        prec = inter / pred_sum if pred_sum > 0 else 0.0
        rec = inter / gt_sum if gt_sum > 0 else 0.0
        
        # Salviamo usando C1, C2, C3...
        risultati_fold[f'C{i+1}'] = {'Dice': dice, 'IoU': iou, 'Prec': prec, 'Rec': rec}
        
    return risultati_fold, len(cat_ids)

# ============================================================
# RACCOLTA DATI E SALVATAGGIO
# ============================================================
risultati_finali = []

for i in range(1, 6):
    print(f"\n>>> Analisi Fold {i} in corso...")
    # SOSTITUISCI CON I TUOI PATH REALI
    gt_path = f'dataset/MergeDataset/KFold_Annotations/fold{i}_val.json'
    pkl_path = f'work_dirs/kfoldTrainMet/fold_{i}/results.pkl'
    
    try:
        risultati_classi, n_classi = calcola_metriche_pixel(gt_path, pkl_path)
        
        riga = {'Fold': i}
        
        for c in range(1, n_classi + 1):
            prefisso = f'C{c}'
            riga[f'Dice_{prefisso}'] = round(risultati_classi[prefisso]['Dice'], 4)
            riga[f'IoU_{prefisso}'] = round(risultati_classi[prefisso]['IoU'], 4)
            riga[f'Prec_{prefisso}'] = round(risultati_classi[prefisso]['Prec'], 4)
            riga[f'Rec_{prefisso}'] = round(risultati_classi[prefisso]['Rec'], 4)
            
        risultati_finali.append(riga)
        print(f"Completato Fold {i} con successo!")
        
    except Exception as e:
        print(f"Errore nel Fold {i}: {e}")

# Media Finale
if risultati_finali:
    media_row = {'Fold': 'MEDIA FINALE'}
    chiavi = [k for k in risultati_finali[0].keys() if k != 'Fold']
    for chiave in chiavi:
        valori_colonna = [r[chiave] for r in risultati_finali]
        media_row[chiave] = round(np.mean(valori_colonna), 4)
    risultati_finali.append(media_row)

colonne_csv = list(risultati_finali[0].keys())

with open(NOME_FILE_CSV, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=colonne_csv)
    writer.writeheader()
    writer.writerows(risultati_finali)

print(f"\nOperazione completata! Controlla il file: {NOME_FILE_CSV}")