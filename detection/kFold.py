import json
import os
import copy
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

def load_and_merge_coco(json_paths):
    """Unisce più file COCO JSON gestendo i conflitti di ID."""
    merged_data = {'images': [], 'annotations': [], 'categories': []}
    
    # Assumiamo che le categorie siano uguali per tutti i set
    with open(json_paths[0], 'r') as f:
        first_json = json.load(f)
        merged_data['categories'] = first_json['categories']

    global_img_id = 1
    global_ann_id = 1
    
    # Mappature per tenere traccia dei vecchi e nuovi ID
    for idx, path in enumerate(json_paths):
        print(f"Caricamento {path}...")
        with open(path, 'r') as f:
            data = json.load(f)
            
        img_id_map = {} # {old_id: new_id}
        
        # 1. Aggiorna le immagini
        for img in data['images']:
            old_id = img['id']
            img_id_map[old_id] = global_img_id
            
            new_img = copy.deepcopy(img)
            new_img['id'] = global_img_id
            # Aggiungiamo un prefisso al nome file in caso di nomi uguali in cartelle diverse
            new_img['file_name'] = f"set{idx+1}_{img['file_name']}" 
            
            merged_data['images'].append(new_img)
            global_img_id += 1
            
        # 2. Aggiorna le annotazioni
        for ann in data['annotations']:
            new_ann = copy.deepcopy(ann)
            new_ann['id'] = global_ann_id
            new_ann['image_id'] = img_id_map[ann['image_id']]
            
            merged_data['annotations'].append(new_ann)
            global_ann_id += 1

    return merged_data

def create_stratified_folds(merged_data, n_splits=5, output_dir='kfold_dataset'):
    os.makedirs(output_dir, exist_ok=True)
    
    images = merged_data['images']
    annotations = merged_data['annotations']
    categories = merged_data['categories']
    
    # Crea una mappa {category_id: category_name}
    cat_map = {cat['id']: cat['name'] for cat in categories}
    
    # Trova quali classi sono presenti in ciascuna immagine
    img_to_classes = defaultdict(set)
    for ann in annotations:
        img_to_classes[ann['image_id']].add(cat_map[ann['category_id']])
        
    # Crea la "label di stratificazione" (es. "AM_HC", "HC_NU", "Solo_HC")
    X = [] # ID delle immagini
    y = [] # Label di stratificazione
    
    for img in images:
        img_id = img['id']
        X.append(img_id)
        
        present_classes = sorted(list(img_to_classes[img_id]))
        if not present_classes:
            strat_label = "EMPTY"
        else:
            strat_label = "_".join(present_classes)
        y.append(strat_label)

    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDistribuzione delle combinazioni di classi nel dataset unito:")
    unique, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"- {val}: {count} immagini")

    # Applica Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Se una classe ha troppe poche immagini per il numero di fold, StratifiedKFold darà un warning.
    # In quel caso, alcune immagini rare verranno distribuite al meglio possibile.
    
    print(f"\nGenerazione di {n_splits} Fold in corso...")
    fold = 1
    for train_idx, val_idx in skf.split(X, y):
        train_img_ids = set(X[train_idx])
        val_img_ids = set(X[val_idx])
        
        # Prepara la struttura COCO per Train e Val
        train_data = {'images': [], 'annotations': [], 'categories': categories}
        val_data = {'images': [], 'annotations': [], 'categories': categories}
        
        # Smista le immagini
        for img in images:
            if img['id'] in train_img_ids:
                train_data['images'].append(img)
            elif img['id'] in val_img_ids:
                val_data['images'].append(img)
                
        # Smista le annotazioni
        for ann in annotations:
            if ann['image_id'] in train_img_ids:
                train_data['annotations'].append(ann)
            elif ann['image_id'] in val_img_ids:
                val_data['annotations'].append(ann)
                
        # Salva i JSON
        train_path = os.path.join(output_dir, f'fold{fold}_train.json')
        val_path = os.path.join(output_dir, f'fold{fold}_val.json')
        
        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        with open(val_path, 'w') as f:
            json.dump(val_data, f)
            
        print(f"Fold {fold} salvato: Train ({len(train_data['images'])} img) | Val ({len(val_data['images'])} img)")
        fold += 1

# ==========================================
# ESECUZIONE DELLO SCRIPT
# ==========================================
if __name__ == '__main__':
    # 1. Inserisci qui i percorsi dei tuoi due JSON originali
    json_files = [
        'dataset/AIR_LEISH_dataset/Set1/_annotations.coco.json',
        'dataset/AIR_LEISH_dataset/Set2/_annotations.coco.json'
    ]
    
    # 2. Carica e unisci
    merged_dataset = load_and_merge_coco(json_files)
    
    # 3. Crea i fold stratificati
    create_stratified_folds(merged_dataset, n_splits=5, output_dir='dataset/MergeDataset/KFold_Annotations')