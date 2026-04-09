import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. INSERISCI QUI LE TUE 5 MATRICI (Sostituisci i dati finti con i tuoi reali)
# Formato: lista di liste o array numpy 2x2
fold_1 = [[53144900,   294247], [  786773,  4756480]]
fold_2 = [[53281524,   367319], [  649372,  4684185]] # Esempio fittizio
fold_3 = [[53156836,   417178], [  948195,  4460191]] # Esempio fittizio
fold_4 = [[53991148,   436053], [  780029,  3775170]] # Esempio fittizio
fold_5 = [[52323895,   684729], [  649574,  5062058]] # Esempio fittizio

# Raggruppiamo le matrici in un singolo array numpy tridimensionale
matrices = np.array([fold_1, fold_2, fold_3, fold_4, fold_5])

# 2. NORMALIZZAZIONE (per riga: True Labels)
# Per ogni matrice, dividiamo ogni cella per la somma della sua riga
normalized_matrices = []
for matrix in matrices:
    # Calcola la somma per ogni riga (asse orizzontale)
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Normalizza e trasforma in percentuale (x 100)
    norm_matrix = (matrix / row_sums) * 100
    normalized_matrices.append(norm_matrix)

normalized_matrices = np.array(normalized_matrices)

# 3. CALCOLO MEDIA E DEVIAZIONE STANDARD
# Calcoliamo la media e la std per ogni singola cella attraverso i 5 fold
mean_matrix = np.mean(normalized_matrices, axis=0)
std_matrix = np.std(normalized_matrices, axis=0)

# 4. CREAZIONE DELLE ETICHETTE TESTUALI (Es: "99.4% ± 0.1%")
labels = np.empty_like(mean_matrix, dtype=object)
for i in range(mean_matrix.shape[0]):
    for j in range(mean_matrix.shape[1]):
        labels[i, j] = f"{mean_matrix[i, j]:.2f}%\n± {std_matrix[i, j]:.2f}%"

# --- STAMPA IN CONSOLE ---
print("=== MATRICE MEDIA NORMALIZZATA (%) ===")
print(mean_matrix)
print("\n=== DEVIAZIONE STANDARD (%) ===")
print(std_matrix)
print("\n=== FORMATO PER SLIDE ===")
print(labels)

# 5. GENERAZIONE DEL GRAFICO (Opzionale ma consigliato per le slide)
plt.figure(figsize=(8, 6))
# Usiamo la mean_matrix per i colori, ma mostriamo le labels con media e std
sns.heatmap(mean_matrix, annot=labels, fmt="", cmap="Blues", 
            cbar_kws={'label': 'Percentuale (%)'}, 
            annot_kws={"size": 14, "weight": "bold"}) # Font grande per le slide

plt.title("Confusion Matrix Media (5-Fold CV)", fontsize=16, pad=20)
plt.ylabel("True Label", fontsize=14)
plt.xlabel("Predicted Label", fontsize=14)

# Imposta i nomi delle classi (modifica 'Classe 0' e 'Classe 1' con i tuoi)
plt.xticks(ticks=[0.5, 1.5], labels=['Background', 'Lesion'], fontsize=12)
plt.yticks(ticks=[0.5, 1.5], labels=['Background', 'Lesion'], fontsize=12, rotation=0)

# Salva l'immagine in alta risoluzione
plt.tight_layout()
plt.savefig("confusion_matrix_kfold.png", dpi=300)
plt.show()