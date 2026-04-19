"""
=============================================================
  MINI-PROJET RANSOMWARE DETECTION
  PARTIE 3 — Normalisation + Selection + PCA
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
sns.set_style("whitegrid")

print("=" * 60)
print("  PARTIE 3 - NORMALISATION + SELECTION + PCA")
print("=" * 60)


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
print("\n[LOAD] Chargement des datasets...")

df      = pd.read_csv("dataSansDoublons.csv")
df_log  = pd.read_csv("dataLogTransforme.csv")
df_bin  = pd.read_csv("dataBinaire.csv")

print(f"   [OK] dataSansDoublons  : {df.shape}")
print(f"   [OK] dataLogTransforme : {df_log.shape}")
print(f"   [OK] dataBinaire       : {df_bin.shape}")

# Encodage label
for d in [df, df_log, df_bin]:
    d['label'] = d['Label'].map({'Ransomware': 1, 'Goodware': 0})

X       = df.drop(['Label', 'label'], axis=1)
y       = df['label']

X_log   = df_log.drop(['Label', 'label'], axis=1)
y_log   = df_log['label']

X_bin   = df_bin.drop(['Label', 'label'], axis=1)
y_bin   = df_bin['label']

print(f"\n   Features disponibles : {X.shape[1]}")
print(f"   Samples              : {X.shape[0]}")


# ─────────────────────────────────────────────
# ETAPE 1 — Normalisation StandardScaler
# ─────────────────────────────────────────────
print("\n[ETAPE] ETAPE 1 : Normalisation StandardScaler...")

scaler = StandardScaler()
X_scaled     = pd.DataFrame(scaler.fit_transform(X),     columns=X.columns)
X_log_scaled = pd.DataFrame(scaler.fit_transform(X_log), columns=X_log.columns)

print(f"   [OK] StandardScaler applique sur donnees brutes et log")
print(f"   Verification brut : moyenne={X_scaled.mean().mean():.4f} | std={X_scaled.std().mean():.4f}")
print(f"   Verification log  : moyenne={X_log_scaled.mean().mean():.4f} | std={X_log_scaled.std().mean():.4f}")


# ─────────────────────────────────────────────
# GRAPHE 1 — Avant/Apres normalisation
# ─────────────────────────────────────────────
print("\n[GRAPHE] Graphe 1 : Effet de la normalisation...")

col_ex = 'activite_fichier'
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Effet de la Normalisation StandardScaler", fontsize=14, fontweight='bold')

# Brut avant
axes[0][0].hist(X[col_ex], bins=40, color='#e74c3c', edgecolor='black', alpha=0.8)
axes[0][0].set_title(f"AVANT normalisation\n(donnees brutes) — '{col_ex}'")
axes[0][0].set_xlabel("Valeur")
axes[0][0].set_ylabel("Frequence")

# Brut apres
axes[0][1].hist(X_scaled[col_ex], bins=40, color='#2ecc71', edgecolor='black', alpha=0.8)
axes[0][1].set_title(f"APRES normalisation\n(StandardScaler) — '{col_ex}'")
axes[0][1].set_xlabel("Valeur normalisee")
axes[0][1].set_ylabel("Frequence")

# Log avant
axes[1][0].hist(X_log[col_ex], bins=40, color='#e67e22', edgecolor='black', alpha=0.8)
axes[1][0].set_title(f"AVANT normalisation\n(donnees log) — '{col_ex}'")
axes[1][0].set_xlabel("Valeur log")
axes[1][0].set_ylabel("Frequence")

# Log apres
axes[1][1].hist(X_log_scaled[col_ex], bins=40, color='#3498db', edgecolor='black', alpha=0.8)
axes[1][1].set_title(f"APRES normalisation\n(log + StandardScaler) — '{col_ex}'")
axes[1][1].set_xlabel("Valeur normalisee")
axes[1][1].set_ylabel("Frequence")

plt.tight_layout()
plt.savefig("p3_graphe1_normalisation.png", bbox_inches='tight')
plt.show()
print("   [OK] p3_graphe1_normalisation.png sauvegarde")


# ─────────────────────────────────────────────
# ETAPE 2 — Selection de features (SelectKBest)
# ─────────────────────────────────────────────
print("\n[ETAPE] ETAPE 2 : Selection de features (ANOVA F-test)...")

# Sur donnees log normalisees
selector_all = SelectKBest(score_func=f_classif, k='all')
selector_all.fit(X_log_scaled, y_log)

scores_all = pd.Series(selector_all.scores_, index=X_log.columns)
scores_sorted = scores_all.sort_values(ascending=False)

# Selectionner top 30
k_best = 30
selector = SelectKBest(score_func=f_classif, k=k_best)
X_selected = selector.fit_transform(X_log_scaled, y_log)
selected_cols = X_log.columns[selector.get_support()].tolist()

print(f"   [OK] {k_best} features selectionnees sur {X_log.shape[1]} totales")
print(f"\n   Top 10 features les plus discriminantes :")
for i, (feat, score) in enumerate(scores_sorted.head(10).items(), 1):
    print(f"   {i:2d}. {feat:<40} F-Score={score:.2f}")


# ─────────────────────────────────────────────
# GRAPHE 2 — Top features discriminantes
# ─────────────────────────────────────────────
print("\n[GRAPHE] Graphe 2 : Features les plus discriminantes...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Top 20 Features les Plus Discriminantes\n(ANOVA F-Score)", fontsize=14, fontweight='bold')

# Barplot horizontal top 20
top20 = scores_sorted.head(20)
colors_bar = ['#e74c3c' if i < 5 else '#e67e22' if i < 10 else '#3498db'
              for i in range(len(top20))]

axes[0].barh(top20.index[::-1], top20.values[::-1], color=colors_bar[::-1], edgecolor='black')
axes[0].set_title("F-Score par feature (top 20)")
axes[0].set_xlabel("F-Score ANOVA")
axes[0].axvline(x=scores_sorted.iloc[k_best-1], color='red', linestyle='--',
                label=f'Seuil selection (top {k_best})')
axes[0].legend()

# Distribution de tous les scores
axes[1].hist(scores_all.values, bins=25, color='#9b59b6', edgecolor='black', alpha=0.8)
axes[1].set_title("Distribution des F-Scores\n(toutes features)")
axes[1].set_xlabel("F-Score")
axes[1].set_ylabel("Nombre de features")
axes[1].axvline(x=scores_sorted.iloc[k_best-1], color='red', linestyle='--',
                label=f'Seuil top {k_best}')
axes[1].legend()

plt.tight_layout()
plt.savefig("p3_graphe2_features_discriminantes.png", bbox_inches='tight')
plt.show()
print("   [OK] p3_graphe2_features_discriminantes.png sauvegarde")


# ─────────────────────────────────────────────
# GRAPHE 3 — Distribution top 6 features par classe
# ─────────────────────────────────────────────
print("\n[GRAPHE] Graphe 3 : Distribution top 6 features par classe...")

top6_cols = scores_sorted.head(6).index.tolist()
df_plot = df_log.copy()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Distribution des Top 6 Features Discriminantes\npar Type de Logiciel",
             fontsize=14, fontweight='bold')
axes_flat = axes.flatten()

for i, col in enumerate(top6_cols):
    for label, color in [('Ransomware', '#e74c3c'), ('Goodware', '#2ecc71')]:
        vals = df_plot[df_plot['Label'] == label][col]
        axes_flat[i].hist(vals, bins=25, alpha=0.6, color=color,
                          label=label, edgecolor='black', linewidth=0.3)
    axes_flat[i].set_title(f"{col}", fontsize=9, fontweight='bold')
    axes_flat[i].set_xlabel("Valeur (log)")
    axes_flat[i].set_ylabel("Frequence")
    axes_flat[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig("p3_graphe3_distribution_top_features.png", bbox_inches='tight')
plt.show()
print("   [OK] p3_graphe3_distribution_top_features.png sauvegarde")


# ─────────────────────────────────────────────
# ETAPE 3 — Importance via Random Forest
# ─────────────────────────────────────────────
print("\n[ETAPE] ETAPE 3 : Importance des features par Random Forest...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_log_scaled, y_log)

rf_importance = pd.Series(rf.feature_importances_, index=X_log.columns)
rf_sorted = rf_importance.sort_values(ascending=False)

print(f"   [OK] Random Forest entraine pour extraire l'importance")
print(f"\n   Top 10 features importantes (Random Forest) :")
for i, (feat, imp) in enumerate(rf_sorted.head(10).items(), 1):
    print(f"   {i:2d}. {feat:<40} Importance={imp:.4f}")


# ─────────────────────────────────────────────
# GRAPHE 4 — Importance Random Forest
# ─────────────────────────────────────────────
print("\n[GRAPHE] Graphe 4 : Importance Random Forest...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Importance des Features — Random Forest vs ANOVA F-Score",
             fontsize=14, fontweight='bold')

# Random Forest top 20
top20_rf = rf_sorted.head(20)
axes[0].barh(top20_rf.index[::-1], top20_rf.values[::-1],
             color='#27ae60', edgecolor='black', alpha=0.85)
axes[0].set_title("Top 20 — Importance Random Forest")
axes[0].set_xlabel("Importance")

# Comparaison RF vs ANOVA sur top 15
top15_feat = rf_sorted.head(15).index.tolist()
rf_vals    = rf_sorted[top15_feat].values
anova_vals = scores_all[top15_feat].values
anova_norm = anova_vals / anova_vals.max() * rf_vals.max()

x_pos = np.arange(len(top15_feat))
width = 0.35
axes[1].bar(x_pos - width/2, rf_vals,    width, label='Random Forest', color='#27ae60', alpha=0.85)
axes[1].bar(x_pos + width/2, anova_norm, width, label='ANOVA (normalise)', color='#e67e22', alpha=0.85)
axes[1].set_title("Comparaison RF vs ANOVA\n(top 15 features RF)")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(top15_feat, rotation=45, ha='right', fontsize=7)
axes[1].set_ylabel("Score")
axes[1].legend()

plt.tight_layout()
plt.savefig("p3_graphe4_importance_rf.png", bbox_inches='tight')
plt.show()
print("   [OK] p3_graphe4_importance_rf.png sauvegarde")


# ─────────────────────────────────────────────
# ETAPE 4 — PCA
# ─────────────────────────────────────────────
print("\n[ETAPE] ETAPE 4 : Analyse en Composantes Principales (PCA)...")

# PCA complete
pca_full = PCA(random_state=42)
pca_full.fit(X_log_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

n_comp_95 = np.argmax(cumvar >= 95) + 1
n_comp_99 = np.argmax(cumvar >= 99) + 1
print(f"   [OK] {n_comp_95} composantes pour 95% de variance")
print(f"   [OK] {n_comp_99} composantes pour 99% de variance")

# PCA 2D pour visualisation
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_log_scaled)

# PCA optimale (95%)
pca_opt = PCA(n_components=n_comp_95, random_state=42)
X_pca_opt = pca_opt.fit_transform(X_log_scaled)


# ─────────────────────────────────────────────
# GRAPHE 5 — Variance expliquee PCA
# ─────────────────────────────────────────────
print("\n[GRAPHE] Graphe 5 : Variance expliquee par PCA...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Analyse en Composantes Principales (ACP/PCA)",
             fontsize=14, fontweight='bold')

# Courbe variance cumulee
axes[0].plot(range(1, len(cumvar)+1), cumvar,
             color='#2c3e50', linewidth=2, marker='o', markersize=3)
axes[0].fill_between(range(1, len(cumvar)+1), cumvar, alpha=0.1, color='#3498db')
axes[0].axhline(y=95, color='#e74c3c', linestyle='--', linewidth=1.5,
                label=f'95% variance -> {n_comp_95} composantes')
axes[0].axhline(y=99, color='#f39c12', linestyle='--', linewidth=1.5,
                label=f'99% variance -> {n_comp_99} composantes')
axes[0].axvline(x=n_comp_95, color='#e74c3c', linestyle=':', alpha=0.7)
axes[0].set_xlabel("Nombre de composantes")
axes[0].set_ylabel("Variance expliquee cumulee (%)")
axes[0].set_title("Variance expliquee cumulee")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, len(cumvar))
axes[0].set_ylim(0, 101)

# Variance par composante (top 20)
var_par_comp = pca_full.explained_variance_ratio_[:20] * 100
axes[1].bar(range(1, 21), var_par_comp, color='#9b59b6', edgecolor='black', alpha=0.85)
axes[1].set_title("Variance expliquee\npar composante (top 20)")
axes[1].set_xlabel("Composante principale")
axes[1].set_ylabel("% de variance expliquee")
axes[1].set_xticks(range(1, 21))

plt.tight_layout()
plt.savefig("p3_graphe5_pca_variance.png", bbox_inches='tight')
plt.show()
print("   [OK] p3_graphe5_pca_variance.png sauvegarde")


# ─────────────────────────────────────────────
# GRAPHE 6 — PCA 2D visualisation classes
# ─────────────────────────────────────────────
print("\n[GRAPHE] Graphe 6 : Projection PCA 2D...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Projection ACP 2D — Ransomware vs Goodware",
             fontsize=14, fontweight='bold')

color_map  = {0: '#2ecc71', 1: '#e74c3c'}
label_map  = {0: 'Goodware', 1: 'Ransomware'}
marker_map = {0: 'o', 1: '^'}

# Scatter simple
for lbl in [0, 1]:
    mask = y_log == lbl
    axes[0].scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=color_map[lbl], label=label_map[lbl],
        alpha=0.6, marker=marker_map[lbl],
        edgecolors='k', linewidth=0.3, s=60
    )
axes[0].set_xlabel(f"CP1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
axes[0].set_ylabel(f"CP2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
axes[0].set_title("ACP : 2 premieres composantes principales")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Avec centroides
for lbl in [0, 1]:
    mask = y_log == lbl
    axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=color_map[lbl], label=label_map[lbl],
                    alpha=0.4, s=40, marker=marker_map[lbl])
    cx = X_2d[mask, 0].mean()
    cy = X_2d[mask, 1].mean()
    axes[1].scatter(cx, cy, c=color_map[lbl], s=400,
                    marker='*', edgecolors='black', linewidth=1.5, zorder=5)
    axes[1].annotate(f'Centre\n{label_map[lbl]}', (cx, cy),
                     textcoords="offset points", xytext=(10, 10), fontsize=9,
                     fontweight='bold')

axes[1].set_xlabel(f"CP1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
axes[1].set_ylabel(f"CP2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
axes[1].set_title("ACP avec centroides des classes")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Annotation variance totale
var_totale = sum(pca_2d.explained_variance_ratio_) * 100
fig.text(0.5, 0.01, f'Variance totale expliquee par 2 composantes : {var_totale:.1f}%',
         ha='center', fontsize=11, style='italic', color='#2c3e50')

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("p3_graphe6_pca_2d.png", bbox_inches='tight')
plt.show()
print("   [OK] p3_graphe6_pca_2d.png sauvegarde")


# ─────────────────────────────────────────────
# SAUVEGARDE CSV FINAL POUR ML
# ─────────────────────────────────────────────
print("\n[SAVE] Sauvegarde du dataset final pour ML...")

# Dataset log + normalise + features selectionnees
df_final = pd.DataFrame(X_selected, columns=selected_cols)
df_final['label'] = y_log.values
df_final.to_csv("dataset_final_ML.csv", index=False)
print(f"   [OK] dataset_final_ML.csv sauvegarde")
print(f"   -> {df_final.shape[0]} lignes | {df_final.shape[1]} colonnes")
print(f"   -> Features selectionnees : {len(selected_cols)}")


# ─────────────────────────────────────────────
# RESUME FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PARTIE 3 TERMINEE AVEC SUCCES !")
print("=" * 60)

print("\nResume complet du Pre-processing :")
print(f"   Dataset brut          : 806 lignes x 65 colonnes")
print(f"   Apres nettoyage       : 588 lignes x 65 colonnes")
print(f"   Apres feature eng.    : 588 lignes x 78 colonnes (+13 features)")
print(f"   Apres selection       : 588 lignes x {len(selected_cols)} colonnes")
print(f"   Composantes PCA 95%%  : {n_comp_95} composantes")

print("\n[FICHIER] Fichiers CSV produits :")
print("   1. dataset_clean_partie1.csv  <- nettoyage de base")
print("   2. dataSansDoublons.csv       <- + feature engineering")
print("   3. dataLogTransforme.csv      <- + transformation log")
print("   4. dataBinaire.csv            <- + encodage binaire")
print("   5. dataset_final_ML.csv       <- + normalisation + selection")

print("\n[GRAPHE] Graphes produits (Partie 3) :")
print("   p3_graphe1_normalisation.png")
print("   p3_graphe2_features_discriminantes.png")
print("   p3_graphe3_distribution_top_features.png")
print("   p3_graphe4_importance_rf.png")
print("   p3_graphe5_pca_variance.png")
print("   p3_graphe6_pca_2d.png")

print("\n*** Pre-processing COMPLET ! Donner dataset_final_ML.csv a la Personne 3 ***")