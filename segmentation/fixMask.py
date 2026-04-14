import os
import cv2
import numpy as np
from pathlib import Path

def applica_fix_a_tutto(cartella_input, cartella_output):
    """
    Entra in ogni sottocartella e applica la binarizzazione (0-1) 
    a OGNI immagine PNG che trova, mantenendo la struttura identica.
    """
    in_dir = Path(cartella_input)
    out_dir = Path(cartella_output)

    # Prende veramente TUTTI i PNG in ogni sottocartella possibile
    tutti_i_png = list(in_dir.rglob('*.png'))
    
    if not tutti_i_png:
        print(f"Non ho trovato nemmeno un PNG in {in_dir}. Controlla il percorso!")
        return

    print(f"Rilevati {len(tutti_i_png)} file. Inizio la conversione totale...")

    for img_path in tutti_i_png:
        # 1. Carica l'immagine (qualsiasi essa sia)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Salto file corrotto o illeggibile: {img_path}")
            continue

        # 2. Applica il fix: tutto quello che non è nero (0) diventa 1
        img_fixed = np.where(img > 0, 1, 0).astype(np.uint8)

        # 3. Ricostruisce il percorso identico nella cartella di destinazione
        rel_path = img_path.relative_to(in_dir)
        target_path = out_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Salva con lo stesso nome
        cv2.imwrite(str(target_path), img_fixed)

    print(f"\nOperazione completata. Struttura clonata e file fixati in: {out_dir}")

if __name__ == "__main__":
    # Assicurati che questi percorsi siano corretti per il tuo ambiente
    PATH_INPUT = "data_medical/27919209/MSLesSeg3C/masks1000"
    PATH_OUTPUT = "data_medical/27919209/MSLesSeg3C/masks_fix1000"

    applica_fix_a_tutto(PATH_INPUT, PATH_OUTPUT)