import os
import torch
import numpy as np
from pathlib import Path
import mmseg_custom 
from mmseg.apis import init_segmentor, inference_segmentor

def esegui_ensemble_3_classi():
    # --- CONFIGURAZIONE PERCORSI ---
    config_file = 'configs/vit_adapter/medseg_custom_WBC.py' # Il tuo nuovo config a 3 classi
    # Usa la cartella dove tieni i test originali o quella mergiata
    cartella_test = 'test_images_C1' 
    cartella_output = 'risultati_ensemble_wbc'
    
    # Percorsi dei pesi per i 5 fold (assicurati che il percorso sia corretto)
    checkpoints = [f'work_dirs_WBC/medseg_custom/fold_{i}/latest.pth' for i in range(1, 6)]
    
    os.makedirs(cartella_output, exist_ok=True)
    
    # Cerchiamo tutte le immagini (png o jpg)
    immagini_test = list(Path(cartella_test).rglob('*.png'))
    if not immagini_test:
        immagini_test = list(Path(cartella_test).rglob('*.jpg'))

    if not immagini_test:
        print(f"Nessuna immagine trovata in {cartella_test}")
        return

    print(f"Inizio Ensemble a 3 CLASSI su {len(immagini_test)} immagini...")

    # MODIFICA 1: Dizionario per memorizzare i conteggi di ogni classe per ogni pixel
    # Struttura: { 'percorso_img': array_di_voti(H, W, 3) }
    voti_classe = {str(img): None for img in immagini_test}

    # --- CICLO SUI 5 MODELLI ---
    model = None
    for fold_idx, ckpt_path in enumerate(checkpoints, 1):
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Salto Fold {fold_idx}: {ckpt_path} non trovato.")
            continue
            
        print(f"\n--- [FOLD {fold_idx}/5] Caricamento Modello ---")
        model = init_segmentor(config_file, ckpt_path, device='cuda:0')
        
        for img_path in immagini_test:
            p_str = str(img_path)
            risultato = inference_segmentor(model, p_str)
            maschera = risultato[0].astype(np.uint8) # Valori: 0, 1, 2
            
            # Inizializziamo l'array dei voti (H, W, 3) alla prima iterazione
            if voti_classe[p_str] is None:
                h, w = maschera.shape
                voti_classe[p_str] = np.zeros((h, w, 3), dtype=np.uint8)
            
            # MODIFICA 2: Incrementiamo il conteggio per la classe predetta in ogni pixel
            # Per ogni classe (0, 1, 2), aggiungiamo 1 dove il modello ha predetto quella classe
            for c in range(3):
                voti_classe[p_str][:, :, c] += (maschera == c).astype(np.uint8)

        # Pulizia GPU
        if fold_idx < len(checkpoints):
            del model
            torch.cuda.empty_cache()

    # --- CALCOLO VINCITORE E SALVATAGGIO ---
    print("\nCalcolo voto a maggioranza e generazione overlay...")
    
    for img_path in immagini_test:
        p_str = str(img_path)
        conteggi = voti_classe[p_str] # Array (H, W, 3)
        
        if conteggi is None: continue
            
        # MODIFICA 3: Argmax sull'ultimo asse per trovare la classe più votata per ogni pixel
        # Se i voti sono [0:1, 1:4, 2:0], argmax restituisce 1 (la classe con 4 voti)
        maschera_finale = np.argmax(conteggi, axis=-1).astype(np.uint8)
        
        # Gestione percorsi output
        percorso_relativo = img_path.relative_to(Path(cartella_test).parent)
        path_salvataggio = os.path.join(cartella_output, percorso_relativo)
        os.makedirs(os.path.dirname(path_salvataggio), exist_ok=True)
        
        # Mostra il risultato usando la palette a 3 colori del config
        model.show_result(
            p_str,
            [maschera_finale],
            out_file=path_salvataggio,
            opacity=1 # Ridotto un po' per vedere meglio nucleo/citoplasma
        )

    print(f"\n✅ ENSEMBLE COMPLETATO! Risultati in: '{cartella_output}'")

if __name__ == '__main__':
    esegui_ensemble_3_classi()