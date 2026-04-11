import os
import torch
import numpy as np
from pathlib import Path

# --- L'IMPORT MAGICO CHE RISOLVE IL KEYERROR ---
import mmseg_custom 
from mmseg.apis import init_segmentor, inference_segmentor

def esegui_ensemble():
    # --- CONFIGURAZIONE PERCORSI (AGGIORNA QUESTI!) ---
    config_file = 'configs/vit_adapter/medseg_custom_Lesion_copy.py' # Il tuo config
    
    # La cartella dove hai salvato i PNG multi-modali di test
    cartella_test = 'data_medical/27919209/MSLesSeg3C/test/images'
    
    # Dove verranno salvate le immagini finali con l'overlay colorato
    cartella_output = './risultati_visivi_ensemble_les_copy' 
    
    nome_pesi = 'latest.pth' 
    
    # Controlla che i path dei tuoi checkpoint siano corretti
    # Esempio: 'work_dirs/il_tuo_config/fold_1/latest.pth'
    checkpoints = [f'work_dirs_Lesion_copy/medseg_custom/fold_{i}/{nome_pesi}' for i in range(1, 6)]
    
    os.makedirs(cartella_output, exist_ok=True)
    
    # MODIFICA 1: Ora la cartella è "piatta", cerchiamo tutti i PNG direttamente 
    # (senza rglob per le sottocartelle FLAIR)
    immagini_test = list(Path(cartella_test).glob('*.png'))
    
    if not immagini_test:
        print(f"Nessuna immagine trovata in {cartella_test}")
        return

    print(f"Inizio Ensemble su {len(immagini_test)} immagini trovate...")
    
    # Usiamo il path convertito a stringa come chiave
    voti_cumulativi = {str(img): None for img in immagini_test}

    # --- CICLO SUI MODELLI ---
    model = None
    fold_validi = 0 # Tiene traccia di quanti modelli troviamo realmente

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

        # Puliamo la memoria della GPU, TRANNE per l'ultimo modello (ci serve per l'overlay)
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
    # MODIFICA 2: Calcolo dinamico della maggioranza (es. se ho 5 modelli, la maggioranza è >= 3)
    soglia_maggioranza = (fold_validi // 2) + 1 
    print(f"\nGenerazione delle immagini finali (Soglia maggioranza: {soglia_maggioranza}/{fold_validi}) con opacità 0.8...")
    
    for img_path in immagini_test:
        percorso_str = str(img_path)
        somma_voti = voti_cumulativi[percorso_str]
        
        if somma_voti is None:
            continue
            
        # Applica il filtro della maggioranza
        maschera_ensemble = np.where(somma_voti >= soglia_maggioranza, 1, 0).astype(np.uint8)
        
        # MODIFICA 3: Visto che non abbiamo sottocartelle, salviamo direttamente col nome del file
        # (es. da "./dataset_vitadapter/test/images/P1_T1_slice_045.png" prende "P1_T1_slice_045.png")
        nome_file = img_path.name
        path_salvataggio = os.path.join(cartella_output, nome_file)
        
        # Usiamo la funzione ufficiale di mmseg per creare l'overlay
        model.show_result(
            percorso_str,
            [maschera_ensemble],
            out_file=path_salvataggio,
            opacity=0.8
        )

    print(f"\n✅ FINITO! Le immagini con l'overlay sono in: '{cartella_output}'")

if __name__ == '__main__':
    esegui_ensemble()