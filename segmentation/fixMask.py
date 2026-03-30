import os
import cv2
import numpy as np
from pathlib import Path

def formatta_maschere_mslesseg(cartella_input, cartella_output):
    """
    Converte le maschere in scala di grigi a 1 canale e mappa i valori a 0 e 1,
    cercando solo dentro le sottocartelle *_MASK.
    """
    in_dir = Path(cartella_input)
    out_dir = Path(cartella_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # LA MAGIA È QUI: cerca i .png SOLO dentro le cartelle che finiscono per "_MASK"
    file_trovati = list(in_dir.rglob('*_MASK/*.png'))
    
    if not file_trovati:
        print(f"Nessun file PNG trovato nelle cartelle maschera di {cartella_input}")
        return

    print(f"Inizio conversione di {len(file_trovati)} maschere da {cartella_input}...")

    for img_path in file_trovati:
        # 1. Carichiamo l'immagine in scala di grigi
        mask = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Errore nella lettura di {img_path}")
            continue

        # 2. Binarizzazione: i pixel > 0 diventano 1 (lesione), il resto 0 (sfondo)
        mask_corretta = np.where(mask > 0, 1, 0).astype(np.uint8)

        # 3. Salviamo la maschera mantenendo la struttura annidata originale
        percorso_relativo = img_path.relative_to(in_dir)
        path_salvataggio = out_dir / percorso_relativo
        path_salvataggio.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(path_salvataggio), mask_corretta)

    print(f"Completato! Maschere salvate in: {cartella_output}")

if __name__ == "__main__":
    # Il percorso base estratto dal tuo screenshot
    # Assicurati di lanciare lo script dalla cartella "segmentation"
    BASE_DIR = "data_medical/27919209/MSLesSeg_Dataset_PNG"
    
    # Cartella dove salverà i risultati mantenendo la stessa struttura, 
    # ma in una root separata per non sovrascrivere i file originali
    OUT_DIR = "data_medical/27919209/MSLesSeg_Dataset_PNG_Fixed"
    
    # Presumo tu abbia le cartelle 'test', 'train' (e forse 'val')
    splits = ['train', 'val', 'test']
    
    for split in splits:
        cartella_in = f"{BASE_DIR}/{split}"
        cartella_out = f"{OUT_DIR}/{split}"
        
        # Controlla se la cartella esiste (es. potrebbe mancare 'val')
        if Path(cartella_in).exists():
            formatta_maschere_mslesseg(cartella_in, cartella_out)
        else:
            print(f"Cartella {cartella_in} non trovata, salto...")