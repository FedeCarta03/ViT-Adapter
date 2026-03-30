import os
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path

def processa_dataset(input_dir, output_dir):
    """
    Scorre la cartella di input, mantiene la struttura delle sottocartelle
    e converte i file .nii.gz in PNG nella cartella di output.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Verifica che la cartella di input esista
    if not input_path.exists():
        print(f"Errore: Il percorso di input {input_path} non esiste.")
        return

    # rglob cerca in modo ricorsivo tutti i file che finiscono con .nii.gz
    file_trovati = list(input_path.rglob('*.nii.gz'))
    print(f"Trovati {len(file_trovati)} file .nii.gz da processare.")

    for nifti_file in file_trovati:
        # Calcola il percorso relativo (es. 'P54/immagine.nii.gz')
        percorso_relativo = nifti_file.relative_to(input_path)
        
        # Nome del file senza estensione (toglie '.nii.gz')
        nome_base = nifti_file.name.replace('.nii.gz', '')
        
        # Crea il percorso di destinazione
        # Esempio: output_dir/P54/nome_base_del_file/
        cartella_destinazione = output_path / percorso_relativo.parent / nome_base
        
        # Crea le cartelle necessarie (exist_ok=True evita errori se esistono già)
        cartella_destinazione.mkdir(parents=True, exist_ok=True)
        
        print(f"\nElaborazione: {percorso_relativo}")
        
        # --- LOGICA DI CONVERSIONE ---
        try:
            img = nib.load(str(nifti_file))
            data = img.get_fdata()
            
            # Gestione file 4D o scarto file non 3D
            if len(data.shape) < 3:
                print(f"  -> Saltato: {nifti_file.name} non è un volume 3D.")
                continue
            elif len(data.shape) == 4:
                data = data[:, :, :, 0]

            num_slices = data.shape[2]
            
            for i in range(num_slices):
                slice_2d = data[:, :, i]

                # Normalizzazione [0 - 255]
                min_val = np.min(slice_2d)
                max_val = np.max(slice_2d)

                if max_val - min_val == 0:
                    slice_norm = np.zeros(slice_2d.shape, dtype=np.uint8)
                else:
                    slice_norm = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                # Opzionale: decommenta per ruotare se le immagini sono storte
                # slice_norm = np.rot90(slice_norm)

                # Salva l'immagine
                img_png = Image.fromarray(slice_norm)
                percorso_salvataggio = cartella_destinazione / f"slice_{i:04d}.png"
                img_png.save(percorso_salvataggio)
                
            print(f"  -> Salvate {num_slices} fette in {cartella_destinazione}")

        except Exception as e:
            print(f"  -> Errore durante l'elaborazione di {nifti_file.name}: {e}")

    print("\nConversione di tutto il dataset completata!")

# === IMPOSTA I TUOI PERCORSI QUI ===
if __name__ == "__main__":
    # Scegli il percorso in base a dove stai eseguendo lo script (Windows o WSL)
    
    # ESEMPIO SE ESEGUI DENTRO WSL (Ubuntu):
    DIRECTORY_INPUT = "MSLesSeg_Dataset/MSLesSeg_Dataset/test"
    DIRECTORY_OUTPUT = "MSLesSeg_Dataset_PNG/test"
    
    
    processa_dataset(DIRECTORY_INPUT, DIRECTORY_OUTPUT)