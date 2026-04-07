import os
import torch
import numpy as np
from pathlib import Path
import torchvision # Necessario per la funzione NMS
import mmcv_custom  
import mmdet_custom 
from mmdet.apis import init_detector, inference_detector
import mmcv

def esegui_ensemble_detection():
    # ==========================================
    # 1. CONFIGURAZIONE PERCORSI
    # ==========================================
    cartella_test = 'dataset/MergeDataset/Immagini' # La tua cartella immagini
    cartella_output = 'risultati_ensemble_detection'
    os.makedirs(cartella_output, exist_ok=True)

    num_classes = 3
    custom_palette = [(255, 0, 0), (0, 150, 0), (0, 0, 255)]
    iou_threshold = 0.5 
    score_thr = 0.5

    # Opzione B adattata: Usiamo i percorsi che sappiamo esistere dalla tua immagine
    base_dir = 'work_dirs/kfoldTrain' # Assicurati che questo sia corretto rispetto a dove lanci lo script
    
    immagini_test = list(Path(cartella_test).glob('*.png'))
    if not immagini_test:
        print(f"Nessuna immagine trovata in {cartella_test}")
        return

    print(f"Inizio Ensemble Memory-Safe su {len(immagini_test)} immagini...")

    # ==========================================
    # 2. CARICAMENTO DI TUTTI I MODELLI
    # ==========================================
    # Invece di caricarli e cancellarli, li teniamo tutti in GPU.
    # Questo richiede abbastanza VRAM. Se la GPU finisce la memoria, dovrai usare la CPU
    # o caricarli uno alla volta (molto più lento).
    modelli = []
    print("Pre-caricamento dei 5 modelli in GPU...")
    for i in range(1, 6):
        fold_dir = f'{base_dir}/fold_{i}'
        config_path = f'{fold_dir}/Config_fold2.py'
        # Usiamo latest.pth come indicato nel tuo screenshot dell'errore
        ckpt_path = f'{fold_dir}/latest.pth' 
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Saltato Fold {i}: file non trovato {ckpt_path}")
            modelli.append(None)
            continue
            
        print(f"  Caricamento Fold {i}...")
        try:
            model = init_detector(config_path, ckpt_path, device='cuda:0')
            modelli.append(model)
        except Exception as e:
            print(f"Errore caricando Fold {i}: {e}")
            modelli.append(None)

    modelli_validi = [m for m in modelli if m is not None]
    if not modelli_validi:
        print("Nessun modello caricato con successo. Esco.")
        return

    # Useremo il primo modello caricato per la funzione di disegno
    disegnatore = modelli_validi[0]

    # ==========================================
    # 3. ELABORAZIONE IMMAGINE PER IMMAGINE (Safe)
    # ==========================================
    for idx_img, img_path in enumerate(immagini_test):
        print(f"Elaborazione {idx_img+1}/{len(immagini_test)}: {img_path.name}")
        
        risultati_correnti = []
        
        # 3.1 Inferenza sui 5 modelli per QUESTA singola immagine
        for model in modelli_validi:
            with torch.no_grad(): # Aiuta a risparmiare VRAM durante l'inferenza
                res = inference_detector(model, str(img_path))
                risultati_correnti.append(res)
                
        # 3.2 Applicazione della NMS per questa immagine
        final_bboxes = []
        final_masks = []

        for class_idx in range(num_classes):
            all_bboxes_class = []
            all_masks_class = []
            
            for res in risultati_correnti:
                if isinstance(res, tuple):
                    bboxes, masks = res
                    all_masks_class.extend(masks[class_idx])
                else:
                    bboxes = res
                
                if len(bboxes[class_idx]) > 0:
                    all_bboxes_class.append(bboxes[class_idx])

            if len(all_bboxes_class) == 0:
                final_bboxes.append(np.zeros((0, 5), dtype=np.float32))
                if 'masks' in locals(): final_masks.append([])
                continue

            all_bboxes_class = np.vstack(all_bboxes_class)
            
            boxes = torch.from_numpy(all_bboxes_class[:, :4]).cuda()
            scores = torch.from_numpy(all_bboxes_class[:, 4]).cuda()
            
            keep = torchvision.ops.nms(boxes, scores, iou_threshold)
            
            final_bboxes.append(all_bboxes_class[keep.cpu().numpy()])
            
            if len(all_masks_class) > 0:
                masks_keep = [all_masks_class[i] for i in keep.cpu().numpy()]
                final_masks.append(masks_keep)

        # 3.3 Disegno e salvataggio
        ensemble_result = (final_bboxes, final_masks) if final_masks else final_bboxes
        path_salvataggio = os.path.join(cartella_output, img_path.name)
        
        disegnatore.show_result(
            str(img_path),
            ensemble_result,
            score_thr=score_thr,
            show=False,
            out_file=path_salvataggio,
            bbox_color=custom_palette,
            mask_color=custom_palette,
            text_color=(255, 255, 255)
        )
        
        # Svuota esplicitamente i risultati accumulati per questa immagine
        del risultati_correnti
        del ensemble_result
        torch.cuda.empty_cache() # Pulisce frammenti di VRAM

    print(f"\n✅ ENSEMBLE COMPLETATO! Risultati in: {cartella_output}")

if __name__ == '__main__':
    esegui_ensemble_detection()