import os
from collections import Counter

def verifica_proporzioni_fold(cartella_output="stratified_folds"):
    # 1. Troviamo tutte le cartelle "fold_X" in ordine
    if not os.path.exists(cartella_output):
        print(f"Errore: Cartella '{cartella_output}' non trovata.")
        return
        
    folds = sorted([f for f in os.listdir(cartella_output) if f.startswith("fold_")])
    
    for fold in folds:
        print(f"\n{'-'*45}")
        print(f"🔍 ANALISI: {fold.upper()}")
        print(f"{'-'*45}")
        
        fold_path = os.path.join(cartella_output, fold)
        
        # Analizziamo sia il training che il validation set
        for file_txt in ["train.txt", "val.txt"]:
            percorso_txt = os.path.join(fold_path, file_txt)
            
            if not os.path.exists(percorso_txt):
                continue
            
            # Leggiamo tutte le righe (che sono i percorsi delle immagini)
            with open(percorso_txt, 'r') as f:
                percorsi = f.read().splitlines()
            
            # Estraiamo la classe da ogni percorso
            # Es: da "/path/.../train_images_C1/img.jpg" estraiamo "C1"
            classi = []
            for p in percorsi:
                cartella_padre = os.path.basename(os.path.dirname(p))
                classe = cartella_padre.split('_')[-1]
                classi.append(classe)
            
            # Contiamo quante volte compare ogni classe e calcoliamo le percentuali
            conteggio = Counter(classi)
            totale = sum(conteggio.values())
            
            print(f"[{file_txt}] - Totale immagini: {totale}")
            
            # Stampiamo in ordine alfabetico (C1, C2, C3, C4)
            for classe, count in sorted(conteggio.items()):
                percentuale = (count / totale) * 100
                print(f"  ➔ {classe}: {count} file ({percentuale:.1f}%)")
        print()

# --- AVVIO ---
if __name__ == "__main__":
    # Assicurati che il nome della cartella coincida con quella generata prima
    verifica_proporzioni_fold(cartella_output="stratified_folds")