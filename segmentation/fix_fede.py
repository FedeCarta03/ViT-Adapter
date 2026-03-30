import os
import cv2

# INSERISCI QUI IL PERCORSO ALLE TUE IMMAGINI
img_folder = '/mslesseg_folds_mstype/fold_2' 

print(f"Controllo le immagini in: {img_folder}")
error_found = False

for root, _, files in os.walk(img_folder):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            file_path = os.path.join(root, file)
            
            # Controlla se il file è vuoto
            if os.path.getsize(file_path) == 0:
                print(f":x: TROVATO FILE VUOTO (0 byte): {file_path}")
                error_found = True
                continue
            
            # Prova a leggerlo con OpenCV
            img = cv2.imread(file_path)
            if img is None:
                print(f":x: IMPOSSIBILE DECODIFICARE: {file_path}")
                error_found = True

if not error_found:
    print(":white_check_mark: Tutte le immagini sembrano valide! Controlla i percorsi nel tuo file Config.")