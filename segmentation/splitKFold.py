import os
from pathlib import Path
from sklearn.model_selection import KFold

def genera_kfold_pazienti_txt(percorso_dataset, k=5, estensioni_valide=('.jpg', '.jpeg', '.png'), dir_output="fold"):
    """
    Legge le cartelle dei pazienti, le divide in k-fold (Patient-Level)
    e salva i percorsi dei file in file txt dentro sottocartelle separate.
    """
    dataset_path = Path(percorso_dataset)

    if not dataset_path.exists():
        print(f"Errore: La cartella {percorso_dataset} non esiste.")
        return

    # 1. Troviamo le cartelle principali (i Pazienti, es: P54, P55)
    # Ignoriamo i file nascosti o file singoli messi per errore
    pazienti = [d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    pazienti.sort() # Li ordiniamo per avere riproducibilità

    if not pazienti:
        print(f"Errore: Nessuna cartella paziente trovata in {percorso_dataset}.")
        return

    print(f"Trovati {len(pazienti)} pazienti. Inizio la divisione in {k} fold (senza mischiare i pazienti!)...")

    # 2. Inizializziamo il K-Fold sui PAZIENTI (non sui file singoli)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Funzione di supporto: dato un elenco di pazienti, raccoglie tutti i loro file
    def raccogli_file_da_pazienti(lista_pazienti):
        file_totali = []
        for paziente in lista_pazienti:
            path_paziente = dataset_path / paziente
            # os.walk esplorerà eventuali sottocartelle (T1, T2, ecc.) di quel paziente
            for root, _, files in os.walk(path_paziente):
                for file in files:
                    if file.lower().endswith(estensioni_valide) and not file.startswith('.'):
                        percorso_completo = os.path.join(root, file)
                        file_totali.append(percorso_completo)
        return file_totali

    # 3. Creiamo la struttura delle cartelle e i file txt
    for fold_idx, (train_index, val_index) in enumerate(kf.split(pazienti)):
        fold_num = fold_idx + 1
        
        # Creiamo la cartella specifica per questo fold (es: fold/fold_1)
        fold_dir = os.path.join(dir_output, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        # Estraiamo i NOMI dei pazienti per train e val
        pazienti_train = [pazienti[i] for i in train_index]
        pazienti_val = [pazienti[i] for i in val_index]

        # Ora raccogliamo tutti i file png di quei specifici pazienti
        train_files = raccogli_file_da_pazienti(pazienti_train)
        val_files = raccogli_file_da_pazienti(pazienti_val)

        # Definiamo i percorsi di output per i txt
        txt_train = os.path.join(fold_dir, "train.txt")
        txt_val = os.path.join(fold_dir, "val.txt")

        # Scriviamo i file
        with open(txt_train, 'w') as f_train:
            f_train.write('\n'.join(train_files))
            
        with open(txt_val, 'w') as f_val:
            f_val.write('\n'.join(val_files))

        print(f"✓ {fold_dir} creata -> Pazienti (Train: {len(pazienti_train)}, Val: {len(pazienti_val)}) " 
              f"| File salvati (train.txt: {len(train_files)}, val.txt: {len(val_files)})")
    
    print(f"\nOperazione completata! Tutta la struttura è dentro la cartella: '{dir_output}'")


# --- COME USARLO ---
if __name__ == "__main__":
    # Inserisci il percorso della cartella 'train' che contiene i file PNG generati in precedenza
    PERCORSO_DATASET = "data_medical/27919209/MSLesSeg_Dataset_PNG/train" 
    
    genera_kfold_pazienti_txt(
        percorso_dataset=PERCORSO_DATASET, 
        k=5, 
        estensioni_valide=('.jpg', '.png', '.jpeg'),
        dir_output="mslesseg_folds"
    )