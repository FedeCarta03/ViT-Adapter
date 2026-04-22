import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def genera_split_medici(cartella_immagini, cartella_maschere, cartella_output, num_folds=5):
    # Crea la cartella base "fold" se non esiste
    os.makedirs(cartella_output, exist_ok=True)
    
    # 1. Leggi tutti i file .npy (le nostre fette normalizzate)
    file_immagini = [f for f in os.listdir(cartella_immagini) if f.endswith('.npy')]
    
    if not file_immagini:
        print(f"Errore: Nessun file .npy trovato in {cartella_immagini}")
        return

    print("Analisi delle maschere per la stratificazione in corso...")
    
    # 2. Estrai le info dal nome e la label dalla maschera
    dati = []
    for nome_file in file_immagini:
        nome_senza_ext = nome_file.replace('.npy', '')
        
        # Formato: P14_T4_slice_015 -> il paziente è la prima parte ('P14')
        paziente_id = nome_senza_ext.split('_')[0]
        
        # Leggiamo la maschera per capire se c'è una lesione
        percorso_maschera = os.path.join(cartella_maschere, f"{nome_senza_ext}.png")
        
        if not os.path.exists(percorso_maschera):
            print(f" [!] Maschera mancante per {nome_file}, salto la fetta.")
            continue
            
        maschera = cv2.imread(percorso_maschera, cv2.IMREAD_GRAYSCALE)
        
        # Etichetta: 1 se c'è almeno un pixel > 0 (lesione), altrimenti 0 (sana)
        etichetta = 1 if np.any(maschera > 0) else 0
            
        dati.append({
            'filename': nome_senza_ext,
            'patient_id': paziente_id,
            'label': etichetta
        })
        
    df = pd.DataFrame(dati)
    print(f"Trovate {len(df)} fette valide. Suddivisione in {num_folds} fold...")
    
    # 3. Creazione Stratified Group K-Fold (con shuffle sui pazienti)
    sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    X = df['filename']
    y = df['label']
    groups = df['patient_id']
    
    # 4. Generazione e salvataggio dei file .txt nelle sottocartelle
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        train_files = df.iloc[train_idx]['filename'].tolist()
        val_files = df.iloc[val_idx]['filename'].tolist()
        
        # --- CREAZIONE SOTTOCARTELLA ---
        # Crea il percorso: es. processed_dataset/train/fold/fold_1
        cartella_fold = os.path.join(cartella_output, f'fold_{fold}')
        os.makedirs(cartella_fold, exist_ok=True)
        
        # Salva train.txt dentro la sottocartella
        with open(os.path.join(cartella_fold, 'train.txt'), 'w') as f:
            f.write('\n'.join(train_files))
            
        # Salva val.txt dentro la sottocartella
        with open(os.path.join(cartella_fold, 'val.txt'), 'w') as f:
            f.write('\n'.join(val_files))
            
        # Log di conferma
        pazienti_val = df.iloc[val_idx]['patient_id'].unique()
        print(f"\nCreata cartella: {cartella_fold}")
        print(f"   -> train.txt : {len(train_files)} fette")
        print(f"   -> val.txt   : {len(val_files)} fette (Pazienti isolati per validazione: {list(pazienti_val)})")

# ====================
# ESECUZIONE
# ====================
if __name__ == "__main__":
    # Aggiorna i percorsi puntando alla tua nuova cartella processata
    # (Usa i percorsi reali che hai generato con lo script precedente)
    CARTELLA_BASE = "data_medical/27919209/NormDataset/train"
    
    cartella_con_i_npy = os.path.join(CARTELLA_BASE, "images")
    cartella_con_le_maschere = os.path.join(CARTELLA_BASE, "masks_fix")
    cartella_dove_salvare_i_txt = os.path.join(CARTELLA_BASE, "fold")
    
    genera_split_medici(
        cartella_immagini=cartella_con_i_npy, 
        cartella_maschere=cartella_con_le_maschere, 
        cartella_output=cartella_dove_salvare_i_txt, 
        num_folds=5
    )