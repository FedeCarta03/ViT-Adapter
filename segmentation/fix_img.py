import cv2
import numpy as np
from pathlib import Path
import os

def correggi_palette_immagini():
    # --- CONFIGURAZIONE CARTELLE ---
    cartella_input = 'risultati_ensemble_wbc/test_images_C1'  # La cartella con le immagini rosso/verdi
    cartella_output = 'risultati_corretti/C1'     # Dove verranno salvate quelle nuove
    
    os.makedirs(cartella_output, exist_ok=True)
    
    # Trova tutte le immagini PNG o JPG
    immagini = list(Path(cartella_input).rglob('*.png'))
    if not immagini:
        immagini = list(Path(cartella_input).rglob('*.jpg'))
        
    if not immagini:
        print(f"Nessuna immagine trovata in {cartella_input}")
        return

    print(f"Trovate {len(immagini)} immagini. Inizio conversione...")

    for img_path in immagini:
        # 1. Leggi l'immagine (OpenCV legge i canali in ordine BGR)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
            
        # Converti in RGB per comodità logica
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Crea una tela vuota completamente nera per i nuovi colori
        # (Questo gestisce automaticamente lo sfondo nero [0, 0, 0])
        nuova_img = np.zeros_like(img_rgb)
        
        # 3. CREAZIONE MASCHERE LOGICHE
        # Trova i pixel dove il ROSSO domina (Nucleo) -> tolleranza generosa (>100)
        mask_red = (img_rgb[:, :, 0] > 100) & (img_rgb[:, :, 1] < 100) & (img_rgb[:, :, 2] < 100)
        
        # Trova i pixel dove il VERDE domina (Citoplasma)
        mask_green = (img_rgb[:, :, 1] > 100) & (img_rgb[:, :, 0] < 100) & (img_rgb[:, :, 2] < 100)
        
        # 4. APPLICAZIONE NUOVA PALETTE
        # Nucleo: da Rosso a Bianco
        nuova_img[mask_red] = [255, 255, 255]
        
        # Citoplasma: da Verde a Grigio
        nuova_img[mask_green] = [174, 174, 174]
        
        # 5. SALVATAGGIO
        # Ricostruisci il percorso mantenendo l'eventuale struttura di sottocartelle
        percorso_relativo = img_path.relative_to(Path(cartella_input))
        path_salvataggio = os.path.join(cartella_output, str(percorso_relativo))
        os.makedirs(os.path.dirname(path_salvataggio), exist_ok=True)
        
        # Salva convertendo di nuovo in BGR (formato richiesto da OpenCV per scrivere)
        cv2.imwrite(path_salvataggio, cv2.cvtColor(nuova_img, cv2.COLOR_RGB2BGR))

    print(f"\n✅ Conversione completata! Le immagini corrette sono pronte in: '{cartella_output}'")

if __name__ == '__main__':
    correggi_palette_immagini()