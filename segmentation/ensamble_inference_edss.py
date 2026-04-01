import os
import torch
import numpy as np
from pathlib import Path

# --- L'IMPORT MAGICO CHE RISOLVE IL KEYERROR ---
# Questo dice a MMSegmentation di caricare l'architettura ViT-Adapter
import mmseg_custom 

from mmseg.apis import init_segmentor, inference_segmentor

def esegui_ensemble():
    # --- CONFIGURAZIONE PERCORSI ---
    config_file = 'configs/vit_adapter/medseg_custom.py'
    cartella_test = 'data_medical/27919209/MSLesSeg_Dataset_PNG_Fixed/test'
    cartella_output = 'risultati_visivi_edss' # Nuova cartella per gli overlay
    
    # Assicurati che il nome corrisponda all'ultimo salvataggio del tuo training
    nome_pesi = 'latest.pth' # Oppure 'iter_10000.pth'
    checkpoints = [f'work_dirs/medseg_custom/fold_{i}/{nome_pesi}' for i in range(1, 6)]
    
    os.makedirs(cartella_output, exist_ok=True)
    immagini_test = list(Path(cartella_test).glob('*.png'))
    
    if not immagini_test:
        print(f"Nessuna immagine trovata in {cartella_test}")
        return

    print(f"Inizio Ensemble su {len(immagini_test)} immagini...")
    voti_cumulativi = {img.name: None for img in immagini_test}

    # --- CICLO SUI 5 MODELLI ---
    model = None
    for fold_idx, ckpt_path in enumerate(checkpoints, 1):
        if not os.path.exists(ckpt_path):
            print(f"ERRORE: Non trovo i pesi in {ckpt_path}. Salto questo fold.")
            continue
            
        print(f"\n--- Caricamento Modello Fold {fold_idx} ---")
        model = init_segmentor(config_file, ckpt_path, device='cuda:0')
        
        for img_path in immagini_test:
            risultato = inference_segmentor(model, str(img_path))
            maschera_predetta = risultato[0].astype(np.uint8) 
            
            if voti_cumulativi[img_path.name] is None:
                voti_cumulativi[img_path.name] = maschera_predetta
            else:
                voti_cumulativi[img_path.name] += maschera_predetta

        # Puliamo la memoria della GPU, TRANNE per l'ultimo modello
        # (L'ultimo ci serve per usare la sua funzione di disegno!)
        if fold_idx < len(checkpoints):
            del model
            torch.cuda.empty_cache()
            print(f"Fold {fold_idx} completato. Memoria GPU liberata.")
        else:
            print(f"Fold {fold_idx} completato. Mantengo l'ultimo modello per generare le immagini.")

    # --- CALCOLO DELLA MAGGIORANZA E DISEGNO OVERLAY ---
    print("\nGenerazione delle immagini finali con opacità 0.8...")
    
    for img_path in immagini_test:
        somma_voti = voti_cumulativi[img_path.name]
        if somma_voti is None:
            continue
            
        # Voto a maggioranza: >= 3 modelli dicono che è un polipo
        maschera_ensemble = np.where(somma_voti >= 3, 1, 0).astype(np.uint8)
        
        path_salvataggio = os.path.join(cartella_output, img_path.name)
        
        # Usiamo la funzione ufficiale di mmseg per creare l'overlay (come il tuo comando test.py!)
        # Passiamo la nostra maschera "perfetta" generata dall'ensemble
        model.show_result(
            str(img_path),
            [maschera_ensemble],
            out_file=path_salvataggio,
            opacity=0.8
        )

    print(f"\n✅ FINITO! Le immagini con l'overlay sono in: '{cartella_output}'")

if __name__ == '__main__':
    esegui_ensemble()