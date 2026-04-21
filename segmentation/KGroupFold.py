import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def genera_split_medici(cartella_immagini, cartella_output, num_folds=5):
    # Crea la cartella base "fold" se non esiste
    os.makedirs(cartella_output, exist_ok=True)
    
    # 1. Leggi tutti i file PNG
    file_immagini = [f for f in os.listdir(cartella_immagini) if f.endswith('.png')]
    
    if not file_immagini:
        print(f"Errore: Nessun file PNG trovato in {cartella_immagini}")
        return

    # 2. Estrai le informazioni dal nome del file
    dati = []
    for nome_file in file_immagini:
        nome_senza_ext = nome_file.replace('.png', '')
        paziente_id = nome_senza_ext.split('_')[0]
        
        # Etichetta: 0 = Sano (finisce con 0), 1 = Lesione
        if nome_senza_ext[-1] == '0':
            etichetta = 0 
        else:
            etichetta = 1 
            
        dati.append({
            'filename': nome_senza_ext,
            'patient_id': paziente_id,
            'label': etichetta
        })
        
    df = pd.DataFrame(dati)
    
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
        # Crea il percorso: es. data_medical/.../fold/fold_1
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
        print(f"Creata cartella: {cartella_fold}")
        print(f"   -> train.txt : {len(train_files)} fette")
        print(f"   -> val.txt   : {len(val_files)} fette (Pazienti isolati: {list(pazienti_val)})\n")

# ====================
# ESECUZIONE
# ====================
if __name__ == "__main__":
    # Percorsi base
    cartella_con_i_png = "data_medical/27919209/MSLesSeg_FLAIR/images"
    cartella_dove_salvare_i_txt = "data_medical/27919209/MSLesSeg_FLAIR/fold"
    
    genera_split_medici(cartella_con_i_png, cartella_dove_salvare_i_txt, num_folds=5)