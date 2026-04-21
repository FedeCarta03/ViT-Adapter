import nibabel as nib
import numpy as np
import cv2
import os

def normalize_volume(volume):
    """Normalizza rimuovendo i picchi anomali."""
    p1, p99 = np.percentile(volume, (1, 99))
    vol_clipped = np.clip(volume, p1, p99)
    vol_norm = (vol_clipped - p1) / (p99 - p1 + 1e-8)
    return (vol_norm * 255.0).astype(np.uint8)

def process_mslesseg_flair_only(dataset_root, output_img_dir, output_mask_dir):
    """
    Naviga la struttura MSLesSeg cercando SOLO file Flair e Mask: 
    dataset_root/Paziente/Timepoint/[FLAIR, MASK].nii.gz
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 1. Trova tutti i pazienti (es. P1, P2, P3...)
    patients = [p for p in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, p))]
    
    total_saved_slices = 0

    for patient in patients:
        patient_path = os.path.join(dataset_root, patient)
        
        # 2. Trova tutti i timepoint per questo paziente (es. T1, T2, T3)
        timepoints = [tp for tp in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, tp))]
        
        for tp in timepoints:
            tp_path = os.path.join(patient_path, tp)
            
            # Percorsi attesi SOLO per FLAIR e MASK
            flair_path = os.path.join(tp_path, f"{patient}_{tp}_FLAIR.nii.gz")
            mask_path = os.path.join(tp_path, f"{patient}_{tp}_MASK.nii.gz") 
            
            # Salta se mancano i file
            if not all(os.path.exists(p) for p in [flair_path, mask_path]):
                print(f"File mancanti per {patient} - {tp} (Richiesti: FLAIR, MASK). Salto.")
                continue
                
            print(f"Elaborazione: Paziente {patient} | Timepoint {tp}...")
            
            # Caricamento
            vol_flair = normalize_volume(nib.load(flair_path).get_fdata())
            vol_mask = nib.load(mask_path).get_fdata() # Maschera NON normalizzata
            
            depth = vol_flair.shape[2]
            saved_for_this_tp = 0
            
            # 3. Estrazione fette
            for z in range(depth):
                mask_slice = vol_mask[:, :, z]
                lesion_pixels = np.sum(mask_slice > 0)
                
                # SOGLIA PER SCLEROSI MULTIPLA (mantenuta uguale al tuo codice)
                is_lesion = lesion_pixels > 1
                is_empty_but_kept = (lesion_pixels == 0) and (z % 100 == 0)
                
                if is_lesion or is_empty_but_kept:
                    
                    # Estrai la singola fetta FLAIR (scala di grigi 2D)
                    img_flair = vol_flair[:, :, z]
                    
                    # --- NOTA IMPORTANTE ---
                    # Se la tua rete neurale (es. U-Net basata su ResNet) si aspetta input a 3 canali (RGB),
                    # scommenta la riga seguente per duplicare il canale FLAIR su 3 dimensioni:
                    # img_flair = np.stack((img_flair, img_flair, img_flair), axis=-1)
                    
                    # Maschera a 8-bit
                    mask_img = (mask_slice > 0).astype(np.uint8) * 255 
                    
                    # NOME FILE UNIVOCO: P1_T1_slice_045.png
                    filename = f"{patient}_{tp}_slice_{z:03d}.png"
                    
                    # Salvataggio
                    cv2.imwrite(os.path.join(output_img_dir, filename), img_flair)
                    cv2.imwrite(os.path.join(output_mask_dir, filename), mask_img)
                    
                    saved_for_this_tp += 1
                    total_saved_slices += 1
                    
            print(f"  -> Salvate {saved_for_this_tp} fette utili.")

    print(f"\nPROCESSO COMPLETATO! Fette totali salvate: {total_saved_slices}")

# ==========================================
# ESECUZIONE
# ==========================================
if __name__ == "__main__":
    # La cartella che contiene "P1", "P2", "P3" ecc.
    root_del_dataset = "MSLesSeg_Dataset/train" 
    
    # Cartelle di output rinominate per distinguere questo dataset da quello 3C
    cartella_immagini = "MSLesSeg_FLAIR/images/"
    cartella_maschere = "MSLesSeg_FLAIR/masks/"
    
    process_mslesseg_flair_only(root_del_dataset, cartella_immagini, cartella_maschere)