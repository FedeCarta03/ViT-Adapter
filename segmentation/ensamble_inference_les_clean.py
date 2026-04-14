import os
import torch
import numpy as np
import cv2  # <-- AGGIUNTO PER LA PULIZIA (Connected Components)
from pathlib import Path

# --- L'IMPORT MAGICO CHE RISOLVE IL KEYERROR ---
import mmseg_custom 
from mmseg.apis import init_segmentor, inference_segmentor

def esegui_ensemble():
    # --- CONFIGURAZIONE PERCORSI (AGGIORNA QUESTI!) ---
    config_file = 'configs/vit_adapter/medseg_custom_Lesion_copy.py' 
    
    cartella_test = 'data_medical/27919209/MSLesSeg3C/test/images'
    
    cartella_output = './risultati_visivi_ensemble_les_copy_100_puliti' # <-- Cambiato nome cartella
    
    nome_pesi = 'latest.pth' 
    
    checkpoints = [f'work_dirs_Lesion_copy/medseg_custom100/fold_{i}/{nome_pesi}' for i in range(1, 6)]
    
    os.makedirs(cartella_output, exist_ok=True)
    
    immagini_test = list(Path(cartella_test).glob('*.png'))
    
    if not immagini_test:
        print(f"Nessuna immagine trovata in {cartella_test}")
        return

    print(f"Inizio Ensemble su {len(immagini_test)} immagini trovate...")
    
    voti_cumulativi = {str(img): None for img in immagini_test}

    # --- CICLO SUI MODELLI ---
    model = None
    fold_validi = 0 

    for fold_idx, ckpt_path in enumerate(checkpoints, 1):
        if not os.path.exists(ckpt_path):
            print(f"ATTENZIONE: Non trovo i pesi in {ckpt_path}. Salto questo fold.")
            continue
            
        print(f"\n--- Caricamento Modello Fold {fold_idx} ---")
        model = init_segmentor(config_file, ckpt_path, device='cuda:0')
        fold_validi += 1
        
        for img_path in immagini_test:
            percorso_str = str(img_path)
            risultato = inference_segmentor(model, percorso_str)
            maschera_predetta = risultato[0].astype(np.uint8) 
            
            if voti_cumulativi[percorso_str] is None:
                voti_cumulativi[percorso_str] = maschera_predetta
            else:
                voti_cumulativi[percorso_str] += maschera_predetta

        # Puliamo la memoria
        if fold_idx < len(checkpoints):
            del model
            torch.cuda.empty_cache()
            print(f"Fold {fold_idx} completato. Memoria GPU liberata.")
        else:
            print(f"Fold {fold_idx} completato. Mantengo l'ultimo modello per generare le immagini.")

    if fold_validi == 0:
        print("Errore: Nessun modello caricato. Controlla i percorsi dei checkpoints.")
        return

    # --- CALCOLO DELLA MAGGIORANZA E DISEGNO OVERLAY ---
    soglia_maggioranza = 2
    print(f"\nGenerazione delle immagini finali pulite (Soglia maggioranza: {soglia_maggioranza}/{fold_validi})...")
    
    # SOGLIA PIXEL PER LA PULIZIA
    min_pixel_size = 3
    
    for img_path in immagini_test:
        percorso_str = str(img_path)
        somma_voti = voti_cumulativi[percorso_str]
        
        if somma_voti is None:
            continue
            
        # 1. Applica il filtro della maggioranza (Grezzo)
        maschera_ensemble = np.where(somma_voti >= soglia_maggioranza, 1, 0).astype(np.uint8)
        
        # ========================================================
        # 2. INIZIO PULIZIA (Connected Components) SULLA MASCHERA FINALE
        # ========================================================
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(maschera_ensemble, connectivity=8)
        
        maschera_pulita = np.zeros_like(maschera_ensemble)
        
        # Ripopoliamo solo le "isole" più grandi della soglia
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_pixel_size:
                maschera_pulita[labels == i] = 1
                
        # Sostituiamo la maschera grezza con quella pulita
        maschera_ensemble = maschera_pulita.astype(np.uint8)
        # ========================================================
        # FINE PULIZIA
        # ========================================================

        nome_file = img_path.name
        path_salvataggio = os.path.join(cartella_output, nome_file)
        
        # 3. Usa la funzione ufficiale di mmseg per creare l'overlay pulito
        model.show_result(
            percorso_str,
            [maschera_ensemble],
            out_file=path_salvataggio,
            opacity=0.8
        )

    print(f"\n✅ FINITO! Le immagini con l'overlay (PULITO) sono in: '{cartella_output}'")

if __name__ == '__main__':
    esegui_ensemble()