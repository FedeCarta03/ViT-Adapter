import os
import shutil
from pathlib import Path

def crea_dataset_specchio_v4(dir_base, dir_output="data_medical/WbcMSBench_Merged"):
    path_base = Path(dir_base)
    
    img_out_dir = Path(dir_output) / "images_all"
    lbl_out_dir = Path(dir_output) / "labels_all"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Contatori per il report finale
    stats = {"train": 0, "val": 0, "errori": 0}

    if not path_base.exists():
        print(f"Errore: La cartella {dir_base} non esiste!")
        return

    print(f"Analizzo la cartella: {path_base}\n")

    for cartella in path_base.iterdir():
        if cartella.is_dir():
            nome_cartella = cartella.name
            
            # PRENDIAMO SOLO TRAIN E VAL (ignoriamo i test per la K-Fold)
            if "images_C" in nome_cartella and ("train" in nome_cartella or "val" in nome_cartella):
                
                tipo_split = "train" if "train" in nome_cartella else "val"
                
                # Cerchiamo la cartella label corrispondente (es. train_label_C1)
                nome_cartella_label = nome_cartella.replace("images", "label")
                cartella_label = path_base / nome_cartella_label
                
                if not cartella_label.exists():
                    print(f"❌ Manca la cartella label per: {nome_cartella}")
                    continue

                for file_img in cartella.iterdir():
                    if file_img.is_file() and not file_img.name.startswith('.'):
                        
                        # Il nome originale della maschera nella cartella sorgente
                        # (es. da "train_images_C3_0019" cerchiamo "train_label_C3_0019.png")
                        nome_file_label_atteso = file_img.stem.replace("images", "label")
                        
                        file_lbl_png = cartella_label / f"{nome_file_label_atteso}.png"
                        file_lbl_fallback = cartella_label / f"{nome_file_label_atteso}{file_img.suffix}"
                        
                        if file_lbl_png.exists():
                            file_lbl = file_lbl_png
                        elif file_lbl_fallback.exists():
                            file_lbl = file_lbl_fallback
                        else:
                            print(f"⚠️ Label non trovata per: {file_img.name}")
                            stats["errori"] += 1
                            continue

                        # IL PASSAGGIO CHIAVE:
                        # Manteniamo il nome ESATTO dell'immagine (es. train_images_C3_0019.png)
                        nome_finale = file_img.name
                        
                        # Percorsi di destinazione
                        dest_img = img_out_dir / nome_finale
                        
                        # Rinominiamo la label per avere LO STESSO IDENTICO NOME DELL'IMMAGINE
                        # Usiamo la stessa estensione della maschera originale (es. .png)
                        nome_finale_label = f"{file_img.stem}{file_lbl.suffix}"
                        dest_lbl = lbl_out_dir / nome_finale_label
                        
                        # Copiamo i file
                        shutil.copy2(file_img, dest_img)
                        shutil.copy2(file_lbl, dest_lbl)
                        
                        stats[tipo_split] += 1

    print("\n--- REPORT FINALE ---")
    print(f"✓ Immagini di TRAIN processate: {stats['train']}")
    print(f"✓ Immagini di VAL processate:   {stats['val']}")
    print(f"✓ Totale file nel merged:       {stats['train'] + stats['val']}")
    if stats["errori"] > 0:
        print(f"⚠️ Errori (label mancanti):     {stats['errori']}")
    print(f"\nTutto pronto nella cartella: '{dir_output}'")


# --- AVVIO ---
if __name__ == "__main__":
    # INSERISCI QUI IL PERCORSO DELLA CARTELLA PRINCIPALE (quella che contiene train_images_C1, ecc.)
    PERCORSO_DATASET = "data_medical/WBC512" 
    
    crea_dataset_specchio_v4(dir_base=PERCORSO_DATASET)