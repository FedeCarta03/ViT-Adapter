import os
import shutil
from pathlib import Path

def crea_dataset_specchio_v3(dir_base, dir_output="data_medical/WbcMSBench_Flattened"):
    path_base = Path(dir_base)
    
    img_out_dir = Path(dir_output) / "images_all"
    lbl_out_dir = Path(dir_output) / "labels_all"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    
    file_processati = 0

    if not path_base.exists():
        print(f"Errore: La cartella {dir_base} non esiste!")
        return

    print(f"Analizzo la cartella: {path_base}\n")

    for cartella in path_base.iterdir():
        if cartella.is_dir():
            nome_cartella = cartella.name
            
            if "images_C" in nome_cartella:
                classe = nome_cartella.split('_')[-1]
                nome_cartella_label = nome_cartella.replace("images", "label")
                cartella_label = path_base / nome_cartella_label
                
                if not cartella_label.exists():
                    continue

                for file_img in cartella.iterdir():
                    if file_img.is_file() and not file_img.name.startswith('.'):
                        
                        # IL TRUCCO È QUI: Sostituiamo la parola "images" con "label" 
                        # anche all'interno del nome del file stesso!
                        nome_file_label_atteso = file_img.stem.replace("images", "label")
                        
                        # Cerchiamo il file .png
                        file_lbl_png = cartella_label / f"{nome_file_label_atteso}.png"
                        # Fallback (se avesse la stessa estensione dell'immagine)
                        file_lbl_fallback = cartella_label / f"{nome_file_label_atteso}{file_img.suffix}"
                        
                        if file_lbl_png.exists():
                            file_lbl = file_lbl_png
                        elif file_lbl_fallback.exists():
                            file_lbl = file_lbl_fallback
                        else:
                            print(f"⚠️ Non trovo la label per l'immagine: {file_img.name}")
                            print(f"   Ho cercato esattamente: {file_lbl_png.name} dentro {cartella_label.name}")
                            continue

                        # Creiamo un nome pulito (es. da "test_images_C1_0000" diventa "C1_0000")
                        nome_pulito = file_img.stem.replace("test_images_", "").replace("train_images_", "").replace("val_images_", "")
                        nuovo_nome_base = f"{classe}_{nome_pulito}"
                        
                        dest_img = img_out_dir / f"{nuovo_nome_base}{file_img.suffix}"
                        dest_lbl = lbl_out_dir / f"{nuovo_nome_base}{file_lbl.suffix}"
                        
                        shutil.copy2(file_img, dest_img)
                        shutil.copy2(file_lbl, dest_lbl)
                        
                        file_processati += 1

    print(f"\n✓ Operazione completata! {file_processati} coppie copiate in '{dir_output}'")

# --- AVVIO ---
if __name__ == "__main__":
    # INSERISCI QUI IL PERCORSO DELLA CARTELLA "WBC512"
    PERCORSO_DATASET = "data_medical/WBC512" 
    
    crea_dataset_specchio_v3(dir_base=PERCORSO_DATASET)