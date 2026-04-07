import mmcv_custom  
import mmdet_custom 
from mmdet.apis import init_detector, inference_detector
import mmcv

# ==========================================
# 1. IMPOSTA I PERCORSI (Modificali se serve)
# ==========================================
config_file = 'work_dirs/Config_fold2_1/Config_fold2.py' 
checkpoint_file = 'work_dirs/Config_fold2_1/latest.pth' 
img_path = 'dataset/MergeDataset/Immagini/set1_20250328_CCimage1.png' 
out_file = 'risultato_test1.jpg' 

# ==========================================
# 2. INIZIALIZZA IL MODELLO
# ==========================================
print("Caricamento del modello...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# ==========================================
# 3. ESEGUI L'INFERENZA
# ==========================================
print(f"Analisi dell'immagine: {img_path} ...")
result = inference_detector(model, img_path)

# ==========================================
# 4. SALVA IL RISULTATO
# ==========================================
# Definisci i colori per ogni classe usando tuple RGB (Rosso, Verde, Blu)
# Ordine delle classi: ('AM', 'HC', 'NU')
custom_palette = [
    (255, 0, 0),    # Colore per 'AM' (Rosso)
    (0, 150, 0),    # Colore per 'HC' (Verde scuro)
    (0, 0, 255)     # Colore per 'NU' (Blu)
]

# Puoi anche definire un colore specifico per il testo delle etichette
text_color = (255, 255, 255) # Bianco per maggiore contrasto

model.show_result(
    img_path, 
    result, 
    score_thr=0.5, 
    show=False, 
    out_file=out_file,
    bbox_color=custom_palette,  # Colore dei contorni/box
    mask_color=custom_palette,  # Colore del riempimento delle maschere
    text_color=text_color       # Colore del testo
)

print(f"✅ Fatto! Immagine salvata con successo in: {out_file}")