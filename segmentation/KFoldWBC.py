import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def genera_stratified_kfold_wbcmsbench(percorso_dataset, k=5, estensioni_valide=('.jpg', '.jpeg', '.png'), dir_output="stratified_folds"):
    """
    Esegue il merge delle cartelle train_images_Cx e val_images_Cx,
    estrae la classe (C1, C2, ecc.) e crea gli split per una Stratified K-Fold.
    """
    dataset_path = Path(percorso_dataset)
    file_totali = []
    classi_totali = [] # Questa lista ci serve per la stratificazione

    if not dataset_path.exists():
        print(f"Errore: La cartella base {percorso_dataset} non esiste.")
        return

    # 1. Raccogliamo i file e le rispettive classi ("C1", "C2", etc.)
    for cartella in dataset_path.iterdir():
        if cartella.is_dir():
            nome_cartella = cartella.name
            
            # Controlliamo se la cartella è una di quelle che ci interessano
            if nome_cartella.startswith("train_images_C") or nome_cartella.startswith("val_images_C"):
                
                # Estraiamo la parte finale del nome (es. "C1", "C2")
                classe_corrente = nome_cartella.split('_')[-1]
                
                # Iteriamo sui file all'interno della cartella
                for file in cartella.iterdir():
                    if file.suffix.lower() in estensioni_valide and not file.name.startswith('.'):
                        percorso_completo = str(file.absolute())
                        file_totali.append(percorso_completo)
                        classi_totali.append(classe_corrente)

    if not file_totali:
        print(f"Errore: Nessun file trovato nelle cartelle train/val in {percorso_dataset}.")
        return

    print(f"Trovati {len(file_totali)} file totali da unire.")
    
    # Check distribuzione classi
    distribuzione = {c: classi_totali.count(c) for c in set(classi_totali)}
    print(f"Distribuzione delle classi per la stratificazione: {distribuzione}")
    print(f"Inizio la divisione in {k} fold stratificati...\n")

    # 2. Inizializziamo la Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # 3. Creiamo la struttura delle cartelle e i file txt
    # Passiamo sia le features (file_totali) che i target (classi_totali) al metodo split
    for fold_idx, (train_index, val_index) in enumerate(skf.split(file_totali, classi_totali)):
        fold_num = fold_idx + 1
        
        # Creiamo la cartella specifica per questo fold
        fold_dir = os.path.join(dir_output, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        # Estraiamo i percorsi in base agli indici forniti dallo split
        train_files = [file_totali[i] for i in train_index]
        val_files = [file_totali[i] for i in val_index]

        # Definiamo i percorsi di output per i txt
        txt_train = os.path.join(fold_dir, "train.txt")
        txt_val = os.path.join(fold_dir, "val.txt")

        # Scriviamo i file
        with open(txt_train, 'w') as f_train:
            f_train.write('\n'.join(train_files))
            
        with open(txt_val, 'w') as f_val:
            f_val.write('\n'.join(val_files))

        print(f"✓ {fold_dir} creata -> train.txt ({len(train_files)} file) | val.txt ({len(val_files)} file)")
    
    print(f"\nOperazione completata con successo! Output salvato in: '{dir_output}'")


# --- COME USARLO ---
if __name__ == "__main__":
    # Inserisci il percorso della cartella PRINCIPALE che contiene 
    # le varie sottocartelle (train_images_C1, val_images_C1, ecc.)
    PERCORSO_BASE_DATASET = "data_medical/WBC512" 
    
    genera_stratified_kfold_wbcmsbench(
        percorso_dataset=PERCORSO_BASE_DATASET, 
        k=5, 
        estensioni_valide=('.jpg', '.png', '.jpeg')
    )