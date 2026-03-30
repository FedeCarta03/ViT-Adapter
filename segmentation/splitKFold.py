import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

def genera_kfold_stratificato_ms_type(file_csv, cartella_output, n_splits=5):
    print("\n--- INIZIO ELABORAZIONE ---")
    print(f"1. Lettura del file clinico: {file_csv}")

    # 💡 TRUCCO MAGICO PANDAS: sep=None e engine='python' fanno sì che Pandas
    # capisca da solo se il CSV usa la virgola o il punto e virgola!
    try:
        df = pd.read_csv(file_csv, sep=None, engine='python')
    except Exception as e:
        print(f"❌ ERRORE FATALE: Impossibile leggere il file CSV. Dettagli: {e}")
        return

    # Stampiamo le colonne così vedi esattamente come si chiamano
    colonne_trovate = df.columns.tolist()
    print(f"✅ Colonne trovate nel CSV: {colonne_trovate}")

    # --- IMPOSTA I NOMI DELLE TUE COLONNE QUI ---
    # Se lo script fallisce, controlla la lista stampata sopra e correggi questi due nomi!
    colonna_paziente = 'Patient' 
    colonna_target = 'MS_Type' 

    # Controllo di sicurezza: le colonne esistono?
    if colonna_paziente not in df.columns or colonna_target not in df.columns:
        print(f"❌ ERRORE: Impossibile trovare '{colonna_paziente}' o '{colonna_target}'.")
        print("Controlla i nomi esatti nella lista delle colonne stampata qui sopra!")
        return

    # Rimuoviamo i duplicati e le righe con target mancante (NaN)
    df_pazienti = df.drop_duplicates(subset=[colonna_paziente]).copy()
    df_pazienti = df_pazienti.dropna(subset=[colonna_target])
    
    print(f"2. Filtraggio: Trovati {len(df_pazienti)} pazienti unici e validi.")

    if len(df_pazienti) == 0:
        print("❌ ERRORE: I dati si sono svuotati! Impossibile procedere con lo split.")
        return

    # --- FASE DI SPLIT (K-FOLD STRATIFICATO) ---
    print(f"3. Generazione dei {n_splits} Fold Stratificati in corso...")
    
    # Prepariamo X (gli ID dei pazienti) e y (le etichette per bilanciare i fold)
    X = df_pazienti[colonna_paziente].values
    y = df_pazienti[colonna_target].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Crea la cartella di output se non esiste
    os.makedirs(cartella_output, exist_ok=True)

    fold = 1
    # skf.split restituisce gli INDICI, non i valori reali
    for train_index, val_index in skf.split(X, y):
        # Recuperiamo gli ID reali usando gli indici
        train_pazienti = X[train_index]
        val_pazienti = X[val_index]

        print(f"   -> Salvataggio Fold {fold}: {len(train_pazienti)} Train, {len(val_pazienti)} Validation")

        # Prepariamo i percorsi dei file (salviamo in formato .txt, puoi cambiare in .csv se preferisci)
        percorso_train = os.path.join(cartella_output, f'train_fold_{fold}.txt')
        percorso_val = os.path.join(cartella_output, f'val_fold_{fold}.txt')

        # Scriviamo i file
        with open(percorso_train, 'w') as f:
            for p in train_pazienti:
                f.write(f"{p}\n")

        with open(percorso_val, 'w') as f:
            for p in val_pazienti:
                f.write(f"{p}\n")

        fold += 1

    print(f"🎉 Finito! I file non sono vuoti. Controlla la cartella: '{cartella_output}'.")


# --- ZONA DI ESECUZIONE DELLO SCRIPT ---
if __name__ == "__main__":
    
    # ⚠️ INSERISCI QUI I TUOI PERCORSI REALI
    PERCORSO_CSV = "data_medical/27919209/MSLesSeg_Dataset/MSLesSeg_Dataset/info_dataset/clinical_data.csv"
    
    # Cartella dove vuoi che vengano creati i file txt/csv divisi
    CARTELLA_OUTPUT = "folds_generati/" 
    
    # Scegli quanti fold vuoi (di solito 5)
    NUMERO_FOLD = 5

    genera_kfold_stratificato_ms_type(PERCORSO_CSV, CARTELLA_OUTPUT, n_splits=NUMERO_FOLD)