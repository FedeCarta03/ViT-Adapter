import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def genera_split_medici(cartella_immagini, cartella_output, num_folds=5):
    os.makedirs(cartella_output, exist_ok=True)
    
    # 1. Leggi tutti i file PNG
    file_immagini = [f for f in os.listdir(cartella_immagini) if f.endswith('.png')]
    
    if not file_immagini:
        print(f"Errore: Nessun file PNG trovato in {cartella_immagini}")
        return

    # 2. Estrai le informazioni dal nome del file
    dati = []
    for nome_file in file_immagini:
        # Togliamo l'estensione per il file txt del ViT-Adapter
        nome_senza_ext = nome_file.replace('.png', '')
        
        # Estraiamo il Paziente (es. da "P1_T1_slice_045" prende "P1")
        paziente_id = nome_senza_ext.split('_')[0]
        
        # Capiamo se è una fetta sana o malata basandoci sulla regola dello zero finale
        # (Se finisce con 0 è la fetta sana aggiunta ogni 10 step)
        if nome_senza_ext[-1] == '0':
            etichetta = 0 # Sano
        else:
            etichetta = 1 # Lesione
            
        dati.append({
            'filename': nome_senza_ext,
            'patient_id': paziente_id,
            'label': etichetta
        })
        
    df = pd.DataFrame(dati)
    
    # 3. Creazione Stratified Group K-Fold
    # Bilancia sani/malati (Stratified) MA tiene uniti i pazienti (Group)
    sgkf = StratifiedGroupKFold(n_splits=num_folds)
    
    X = df['filename']
    y = df['label']
    groups = df['patient_id']
    
    # 4. Generazione e salvataggio dei file .txt
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        train_files = df.iloc[train_idx]['filename'].tolist()
        val_files = df.iloc[val_idx]['filename'].tolist()
        
        # Calcola quanti pazienti unici ci sono nel validation
        pazienti_val = df.iloc[val_idx]['patient_id'].unique()
        
        # Salva Train
        with open(os.path.join(cartella_output, f'fold_{fold}_train.txt'), 'w') as f:
            f.write('\n'.join(train_files))
            
        # Salva Validation
        with open(os.path.join(cartella_output, f'fold_{fold}_val.txt'), 'w') as f:
            f.write('\n'.join(val_files))
            
        print(f"✅ Fold {fold} creato:")
        print(f"   -> Train: {len(train_files)} fette")
        print(f"   -> Val:   {len(val_files)} fette (Pazienti isolati nel Val: {list(pazienti_val)})\n")

# ====================
# ESECUZIONE
# ====================
if __name__ == "__main__":
    # Inserisci i tuoi percorsi reali
    cartella_con_i_png = "data_medical/27919209/MSLesSeg3C/images"
    cartella_dove_salvare_i_txt = "data_medical/27919209/MSLesSeg3C/fold"
    
    genera_split_medici(cartella_con_i_png, cartella_dove_salvare_i_txt, num_folds=5)