import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# I tuoi risultati reali dei 5-Fold
results = {
    'F1-Macro': [0.9485, 0.9463, 0.9273, 0.9251, 0.9355],
    'F1-Weighted': [0.9827, 0.9825, 0.9763, 0.9790, 0.9773],
    'MCC': [0.8974, 0.8931, 0.8560, 0.8509, 0.8710],
    'Accuracy': [0.9829, 0.9828, 0.9769, 0.9794, 0.9773]
}

# Trasformiamo i dati per Seaborn
df = pd.DataFrame(results).melt(var_name='Metrica', value_name='Score')

# Creazione della figura
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid", font_scale=1.1)

# 1. Disegniamo i violini (trasparenti per non coprire i punti)
sns.violinplot(
    x='Metrica', 
    y='Score', 
    data=df, 
    inner=None,          # Rimuoviamo i trattini interni
    palette="pastel", 
    cut=0,               # Non far sbordare il violino oltre i dati reali
    alpha=0.6            # Leggera trasparenza
)

# 2. Disegniamo i singoli punti (i 5 fold) sovrapposti
sns.stripplot(
    x='Metrica', 
    y='Score', 
    data=df, 
    color="black", 
    alpha=0.8,           # Punti leggermente trasparenti
    jitter=True,         # Sposta leggermente i punti sull'asse x se si sovrappongono
    size=7               # Grandezza dei puntini
)

plt.title('Distribuzione e Stabilità delle metriche (5-Fold CV)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Punteggio', fontsize=13)
plt.xlabel('', fontsize=13)

# LA MAGIA È QUI: Zoom sull'asse Y per enfatizzare la varianza!
# Troviamo il valore minimo assoluto tra tutti i tuoi dati e togliamo un 2% per margine
min_val = min([min(v) for v in results.values()])
plt.ylim(min_val - 0.02, 1.0) 

# Salvataggio in alta risoluzione (facoltativo, perfetto per paper)
plt.tight_layout()
plt.savefig('kfold_distribution_zoomed.png', dpi=300)

plt.show()