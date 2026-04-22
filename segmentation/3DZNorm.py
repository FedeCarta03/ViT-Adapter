import nibabel as nib
import numpy as np
import os
import cv2
from glob import glob

def process_mslesseg_dataset(input_base_dir, output_base_dir, modality='FLAIR'):
    """
    Scansiona il dataset MSLesSeg (SOLO TRAIN), normalizza i volumi in 3D e salva le slice 2D.
    Mantiene sincronizzate le immagini (.npy) e le maschere (.png).
    """
    # Impostato solo su 'train' come richiesto
    splits = ['train']

    for split in splits:
        split_dir = os.path.join(input_base_dir, split)
        if not os.path.exists(split_dir):
            print(f"La cartella {split_dir} non esiste. Controlla il percorso.")
            continue

        print(f"--- Processando la cartella: {split.upper()} ---")

        # Creiamo le cartelle di destinazione per MMSegmentation
        out_img_dir = os.path.join(output_base_dir, split, 'images')
        out_mask_dir = os.path.join(output_base_dir, split, 'masks_fix')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        # Iteriamo sui pazienti (P1, P2, P3...)
        patients = [p for p in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, p))]
        
        for patient in patients:
            patient_dir = os.path.join(split_dir, patient)
            
            # Iteriamo sui timepoint (T1, T2, T3...)
            timepoints = [t for t in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, t))]
            
            for tp in timepoints:
                tp_dir = os.path.join(patient_dir, tp)
                
                # Costruiamo i nomi dei file basandoci sulla tua struttura
                img_path = os.path.join(tp_dir, f"{patient}_{tp}_{modality}.nii.gz")
                mask_path = os.path.join(tp_dir, f"{patient}_{tp}_MASK.nii.gz")

                if not os.path.exists(img_path) or not os.path.exists(mask_path):
                    print(f"  [ATTENZIONE] File mancanti in {tp_dir}, salto questa cartella.")
                    continue
                
                print(f"  Elaborando: {patient} - {tp}")

                # 1. Carica i volumi 3D
                img_vol = nib.load(img_path).get_fdata(dtype=np.float32)
                
                # FIX APPLICATO QUI: get_fdata() restituisce float, poi lo castiamo a uint8
                mask_vol = nib.load(mask_path).get_fdata().astype(np.uint8) 

                # 2. Maschera del cervello (tutto ciò che è > 0)
                brain_mask_3d = img_vol > 0
                
                if np.any(brain_mask_3d):
                    # 3. Calcola mean e std sull'intero volume 3D
                    vol_mean = np.mean(img_vol[brain_mask_3d])
                    vol_std = np.std(img_vol[brain_mask_3d])
                    
                    # 4. Z-Score Normalization solo sul cervello
                    img_vol = (img_vol - vol_mean) / (vol_std + 1e-8)
                    
                    # Opzionale: rimetti il background esattamente a 0 (può aiutare il Vision Transformer)
                    # img_vol[~brain_mask_3d] = 0.0

                # 5. Salva le slice in 2D
                # Assumiamo che l'asse Z sia il terzo asse (indice 2)
                for i in range(img_vol.shape[2]): 
                    img_slice = img_vol[:, :, i]
                    mask_slice = mask_vol[:, :, i]
                    
                    # Estraiamo la maschera del cervello per QUESTA singola slice
                    brain_slice_mask = brain_mask_3d[:, :, i]
                    
                    # SALVIAMO SOLO SE C'E' CERVELLO NELLA SLICE
                    if np.any(brain_slice_mask): 
                        # Nome base identico per immagine e maschera
                        base_name = f"{patient}_{tp}_slice_{i:03d}"
                        
                        # Salva l'immagine normalizzata come array numpy (.npy)
                        np.save(os.path.join(out_img_dir, f"{base_name}.npy"), img_slice)
                        
                        # Salva la maschera come immagine .png
                        cv2.imwrite(os.path.join(out_mask_dir, f"{base_name}.png"), mask_slice)

    print("\nPre-processing del TRAIN completato con successo!")

# ==========================================
# ESECUZIONE DELLO SCRIPT
# ==========================================
if __name__ == "__main__":
    # INSERISCI I TUOI PERCORSI REALI QUI SOTTO
    INPUT_DATASET_DIR = 'data_medical/27919209/MSLesSeg_Dataset/MSLesSeg_Dataset' # Esempio di percorso in base al tuo log
    OUTPUT_READY_DIR = 'data_medical/27919209/NormDataset'
    
    process_mslesseg_dataset(INPUT_DATASET_DIR, OUTPUT_READY_DIR, modality='FLAIR')