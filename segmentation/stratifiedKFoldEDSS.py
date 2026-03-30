import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def genera_kfold_stratificato_edss(percorso_dataset, file_excel, k=5, estensioni_valide=('.jpg', '.jpeg', '.png'), dir_output="fold_edss"):
    dataset_path = Path(percorso_dataset)

    if not dataset_path.exists():
        print(f"Errore: La cartella {percorso_dataset} non esiste.")
        return

    # 1. Carichiamo l'Excel e creiamo il dizionario Paziente -> EDSS
    print(f"Lettura del file clinico: {file_excel}...")
    df = pd.read_csv(file_excel, sep=";") # Usa read_csv se il file è .csv
    df_pazienti = df.drop_duplicates(subset=['Patient']).copy()
    
    # Rimuoviamo eventuali pazienti che non hanno il valore EDSS
    df_pazienti = df_pazienti.dropna(subset=['EDSS'])
    
    dizionario_etichette = dict(zip(df_pazienti['Patient'].astype(str), df_pazienti['EDSS'].astype(str)))

    # 2. Troviamo le cartelle principali (i Pazienti)
    cartelle_pazienti = [d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    cartelle_pazienti.sort()

    # 3. Mappiamo le cartelle con le etichette
    pazienti_validi = []
    etichette = []
    for p in cartelle_pazienti:
        if p in dizionario_etichette:
            pazienti_validi.append(p)
            etichette.append(dizionario_etichette[p])
        else:
            print(f"  -> Avviso: Paziente {p} saltato (non trovato nel file Excel o EDSS mancante).")

    if not pazienti_validi:
        print("Errore: Nessun paziente valido per la stratificazione.")
        return

    print(f"Trovati {len(pazienti_validi)} pazienti validi. Inizio la divisione in {k} fold Stratificati per EDSS...")

    # Convertiamo in numpy array per StratifiedKFold
    pazienti_array = np.array(pazienti_validi)
    etichette_array = np.array(etichette)

    # 4. Inizializziamo il K-Fold Stratificato
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    def raccogli_file_da_pazienti(lista_pazienti):
        file_totali = []
        for paziente in lista_pazienti:
            path_paziente = dataset_path / paziente
            for root, _, files in os.walk(path_paziente):
                for file in files:
                    if file.lower().endswith(estensioni_valide) and not file.startswith('.'):
                        file_totali.append(os.path.join(root, file))
        return file_totali

    # 5. Creiamo la struttura
    for fold_idx, (train_index, val_index) in enumerate(skf.split(pazienti_array, etichette_array)):
        fold_num = fold_idx + 1
        fold_dir = os.path.join(dir_output, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        pazienti_train = pazienti_array[train_index]
        pazienti_val = pazienti_array[val_index]

        train_files = raccogli_file_da_pazienti(pazienti_train)
        val_files = raccogli_file_da_pazienti(pazienti_val)

        with open(os.path.join(fold_dir, "train.txt"), 'w') as f_train:
            f_train.write('\n'.join(train_files))
            
        with open(os.path.join(fold_dir, "val.txt"), 'w') as f_val:
            f_val.write('\n'.join(val_files))

        print(f"✓ {fold_dir} creata -> Pazienti (Train: {len(pazienti_train)}, Val: {len(pazienti_val)}) | File salvati (train: {len(train_files)}, val: {len(val_files)})")
    
    print(f"\nOperazione completata! Output in: '{dir_output}'")

# --- ESECUZIONE ---
if __name__ == "__main__":
    PERCORSO_DATASET = "data_medical/27919209/MSLesSeg_Dataset_PNG/train" 
    FILE_CLINICO = "data_medical/27919209/MSLesSeg_Dataset/MSLesSeg_Dataset/info_dataset/clinical_data.csv" 
    
    genera_kfold_stratificato_edss(
        percorso_dataset=PERCORSO_DATASET, 
        file_excel=FILE_CLINICO,
        k=5, 
        estensioni_valide=('.png', '.jpg', '.jpeg'),
        dir_output="mslesseg_folds_edss" # <--- Cambiato il nome della cartella di output
    )