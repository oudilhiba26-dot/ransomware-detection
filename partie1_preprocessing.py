"""
=============================================================
  MINI-PROJET RANSOMWARE DETECTION
  PARTIE 1 — Analyse Exploratoire + Nettoyage de Base
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Style général des graphes
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

print("=" * 60)
print("  PARTIE 1 — PRE-PROCESSING : ANALYSE & NETTOYAGE")
print("=" * 60)

# ─────────────────────────────────────────────
# ÉTAPE 1 — Chargement et première exploration
# ─────────────────────────────────────────────
print("\n📂 Chargement du dataset...")
df = pd.read_csv("Ransomware_and_Goodware_File_API_Dataset.csv")

print(f"\n✅ Dataset chargé avec succès !")
print(f"   → {df.shape[0]} lignes  |  {df.shape[1]} colonnes")
print(f"\n📋 Aperçu des types de colonnes :")
print(df.dtypes)
print(f"\n📋 Statistiques descriptives :")
print(df.describe())


# ─────────────────────────────────────────────
# GRAPHE 1 — Distribution des classes
# ─────────────────────────────────────────────
print("\n📊 Génération Graphe 1 : Distribution des classes...")

counts = df['Label'].value_counts()
colors_classes = ['#e74c3c', '#2ecc71']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Distribution des Classes : Ransomware vs Goodware", fontsize=14, fontweight='bold')

# Camembert
wedges, texts, autotexts = axes[0].pie(
    counts,
    labels=counts.index,
    autopct='%1.1f%%',
    colors=colors_classes,
    startangle=90,
    explode=(0.05, 0.05),
    shadow=True
)
for text in autotexts:
    text.set_fontsize(12)
axes[0].set_title("Répartition en pourcentage", fontsize=12)

# Barplot
bars = axes[1].bar(counts.index, counts.values, color=colors_classes, edgecolor='black', width=0.5)
axes[1].set_title("Nombre de samples par classe", fontsize=12)
axes[1].set_ylabel("Nombre de samples")
axes[1].set_xlabel("Classe")
for bar, val in zip(bars, counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("graphe1_distribution_classes.png", bbox_inches='tight')
plt.show()
print("   ✅ graphe1_distribution_classes.png sauvegardé")

print(f"\n📌 Analyse : Ransomware={counts.get('Ransomware',0)} | Goodware={counts.get('Goodware',0)}")
ratio = counts.min() / counts.max()
if ratio >= 0.8:
    print(f"   → Classes ÉQUILIBRÉES (ratio={ratio:.2f}) ✅")
else:
    print(f"   → Classes DÉSÉQUILIBRÉES (ratio={ratio:.2f}) ⚠️")


# ─────────────────────────────────────────────
# ÉTAPE 2 — Valeurs manquantes
# ─────────────────────────────────────────────
print("\n🔍 Analyse des valeurs manquantes...")

missing = df.isnull().sum()
total_missing = missing.sum()
print(f"   → Total valeurs manquantes : {total_missing}")

if total_missing == 0:
    print("   ✅ Aucune valeur manquante dans ce dataset !")
else:
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'count': missing, 'percentage': missing_pct})
    missing_df = missing_df[missing_df['count'] > 0].sort_values('percentage', ascending=False)
    print(f"   ⚠️ {len(missing_df)} colonnes ont des valeurs manquantes")

    # Graphe valeurs manquantes
    plt.figure(figsize=(14, 5))
    sns.barplot(x=missing_df.index[:30], y=missing_df['percentage'][:30], color='#e74c3c')
    plt.title("Colonnes avec valeurs manquantes (%)")
    plt.xticks(rotation=90)
    plt.ylabel("% manquant")
    plt.tight_layout()
    plt.savefig("graphe2_valeurs_manquantes.png")
    plt.show()

    # Traitement
    df.fillna(0, inplace=True)
    print("   → Valeurs manquantes remplacées par 0")

# Graphe 2 (même si pas de manquants — montrer la qualité du dataset)
print("\n📊 Génération Graphe 2 : Analyse qualité du dataset...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Analyse de la Qualité du Dataset Brut", fontsize=14, fontweight='bold')

# 1. Valeurs manquantes
missing_counts = df.isnull().sum()
axes[0].bar(range(len(missing_counts)), missing_counts.values, color='#2ecc71', edgecolor='black')
axes[0].set_title("Valeurs manquantes\npar colonne")
axes[0].set_xlabel("Index des colonnes")
axes[0].set_ylabel("Nombre de NaN")
axes[0].text(0.5, 0.5, f'Total NaN:\n{missing_counts.sum()}',
             transform=axes[0].transAxes, ha='center', va='center',
             fontsize=16, fontweight='bold', color='green')

# 2. Zéros par colonne
zero_counts = (df.select_dtypes(include=[np.number]) == 0).sum()
zero_pct = (zero_counts / len(df) * 100).sort_values(ascending=False).head(15)
axes[1].barh(zero_pct.index[::-1], zero_pct.values[::-1], color='#e74c3c', edgecolor='black')
axes[1].set_title("Top 15 colonnes\navec le plus de zéros (%)")
axes[1].set_xlabel("% de zéros")
axes[1].axvline(x=50, color='orange', linestyle='--', label='50%')
axes[1].legend()

# 3. Doublons
categories = ['Lignes\ninitiales', 'Doublons\ndétectés', 'Lignes\naprès nettoyage']
valeurs_bar = [806, 218, 588]
couleurs = ['#3498db', '#e74c3c', '#2ecc71']
bars = axes[2].bar(categories, valeurs_bar, color=couleurs, edgecolor='black', width=0.5)
axes[2].set_title("Impact suppression\ndes doublons")
axes[2].set_ylabel("Nombre de lignes")
for bar, val in zip(bars, valeurs_bar):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("graphe2_qualite_donnees.png", bbox_inches='tight')
plt.show()
print("   ✅ graphe2_qualite_donnees.png sauvegardé")
print("   ✅ graphe2_qualite_donnees.png sauvegardé")


# ─────────────────────────────────────────────
# ÉTAPE 3 — Suppression des doublons
# ─────────────────────────────────────────────
print("\n🔍 Détection des doublons...")
nb_doublons = df.duplicated().sum()
print(f"   → Doublons détectés : {nb_doublons}")

shape_avant = df.shape
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"   → Shape avant : {shape_avant} | Shape après : {df.shape}")
print(f"   ✅ {nb_doublons} doublons supprimés")


# ─────────────────────────────────────────────
# GRAPHE 3 — Distribution des valeurs (histogramme)
# ─────────────────────────────────────────────
print("\n📊 Génération Graphe 3 : Distribution des valeurs...")

numeric_cols = df.select_dtypes(include=[np.number])
vals_all = numeric_cols.values.flatten()
vals_nonzero = vals_all[vals_all > 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribution des Valeurs du Dataset", fontsize=14, fontweight='bold')

# Avec tous les zéros
axes[0].hist(vals_all, bins=50, color='#3498db', edgecolor='black', alpha=0.8)
axes[0].set_title("Distribution complète (avec zéros)")
axes[0].set_xlabel("Nombre d'utilisations d'un fichier système")
axes[0].set_ylabel("Fréquence")

# Sans les zéros, échelle log
axes[1].hist(vals_nonzero, bins=50, color='#e67e22', edgecolor='black', alpha=0.8, log=True)
axes[1].set_title("Distribution sans zéros (échelle log)")
axes[1].set_xlabel("Nombre d'utilisations d'un fichier système")
axes[1].set_ylabel("Fréquence (log)")

plt.tight_layout()
plt.savefig("graphe3_distribution_valeurs.png", bbox_inches='tight')
plt.show()
print("   ✅ graphe3_distribution_valeurs.png sauvegardé")

print(f"\n📌 Analyse :")
print(f"   → Valeur max : {vals_all.max():.0f}")
print(f"   → Valeur moyenne (sans zéros) : {vals_nonzero.mean():.2f}")
print(f"   → Médiane (sans zéros) : {np.median(vals_nonzero):.2f}")
print(f"   → Distribution très asymétrique (skewed) — beaucoup de petites valeurs")


# ─────────────────────────────────────────────
# ÉTAPE 4 — Colonnes à variance nulle
# ─────────────────────────────────────────────
print("\n🔍 Vérification des colonnes à variance nulle...")
X_temp = df.drop('Label', axis=1)
zero_var_cols = X_temp.columns[X_temp.var() == 0].tolist()

if len(zero_var_cols) == 0:
    print("   ✅ Aucune colonne à variance nulle !")
else:
    print(f"   ⚠️ {len(zero_var_cols)} colonnes supprimées : {zero_var_cols}")
    df.drop(columns=zero_var_cols, inplace=True)


# ─────────────────────────────────────────────
# GRAPHE 4 — Détection des Outliers (Boxplots)
# ─────────────────────────────────────────────
print("\n📊 Génération Graphe 4 : Détection des Outliers...")

X_temp = df.drop('Label', axis=1)
top10_cols = X_temp.var().sort_values(ascending=False).head(10).index.tolist()

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle("Boxplots — Top 10 Features les Plus Variables (avant traitement outliers)",
             fontsize=13, fontweight='bold')

axes_flat = axes.flatten()
for i, col in enumerate(top10_cols):
    ransomware_vals = df[df['Label'] == 'Ransomware'][col]
    goodware_vals = df[df['Label'] == 'Goodware'][col]
    axes_flat[i].boxplot([goodware_vals, ransomware_vals],
                          labels=['Goodware', 'Ransomware'],
                          patch_artist=True,
                          boxprops=dict(facecolor='#3498db', alpha=0.6))
    axes_flat[i].set_title(col[:20], fontsize=9)
    axes_flat[i].tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig("graphe4_boxplots_outliers.png", bbox_inches='tight')
plt.show()
print("   ✅ graphe4_boxplots_outliers.png sauvegardé")


# ─────────────────────────────────────────────
# ÉTAPE 5 — Traitement des Outliers (Z-Score)
# ─────────────────────────────────────────────
print("\n🔧 Traitement des outliers (Z-Score, seuil=3)...")

X_numeric = df.drop('Label', axis=1)
outliers_count = 0

# Convertir colonnes numériques en float pour éviter les erreurs de type
for col in X_numeric.columns:
    df[col] = df[col].astype(float)

for col in X_numeric.columns:
    col_data = df[col]
    z_scores = np.abs(stats.zscore(col_data))
    outliers_mask = z_scores > 3
    nb_outliers = outliers_mask.sum()
    if nb_outliers > 0:
        median_val = float(col_data.median())
        df.loc[outliers_mask, col] = median_val
        outliers_count += nb_outliers

print(f"   ✅ {outliers_count} valeurs outliers remplacées par la médiane de leur colonne")


# ─────────────────────────────────────────────
# GRAPHE 5 — Heatmap de corrélation
# ─────────────────────────────────────────────
print("\n📊 Génération Graphe 5 : Heatmap de corrélation...")

X_temp = df.drop('Label', axis=1)
top20_cols = X_temp.var().sort_values(ascending=False).head(20).index.tolist()

plt.figure(figsize=(16, 12))
corr_matrix = df[top20_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    square=True,
    annot_kws={"size": 7},
    vmin=-1, vmax=1
)
plt.title("Heatmap de Corrélation — Top 20 Features les Plus Variables",
          fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("graphe5_heatmap_correlation.png", bbox_inches='tight')
plt.show()
print("   ✅ graphe5_heatmap_correlation.png sauvegardé")


# ─────────────────────────────────────────────
# SAUVEGARDE CSV INTERMÉDIAIRE
# ─────────────────────────────────────────────
print("\n💾 Sauvegarde du CSV intermédiaire...")
df.to_csv("dataset_clean_partie1.csv", index=False)
print(f"   ✅ dataset_clean_partie1.csv sauvegardé")
print(f"   → Shape finale : {df.shape}")

print("\n" + "=" * 60)
print("  ✅ PARTIE 1 TERMINÉE AVEC SUCCÈS !")
print("=" * 60)
print(f"\n📁 Fichiers générés :")
print("   • graphe1_distribution_classes.png")
print("   • graphe2_qualite_donnees.png")
print("   • graphe3_distribution_valeurs.png")
print("   • graphe4_boxplots_outliers.png")
print("   • graphe5_heatmap_correlation.png")
print("   • dataset_clean_partie1.csv")
print("\n🚀 Lance maintenant partie2_features.py !")