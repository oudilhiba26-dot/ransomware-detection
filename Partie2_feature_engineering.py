"""
=============================================================
  MINI-PROJET RANSOMWARE DETECTION
  PARTIE 2 — Feature Engineering + Transformations
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
sns.set_style("whitegrid")

print("=" * 60)
print("  PARTIE 2 — FEATURE ENGINEERING & TRANSFORMATIONS")
print("=" * 60)

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
print("\n📂 Chargement du dataset nettoyé (Partie 1)...")
df = pd.read_csv("dataset_clean_partie1.csv")
print(f"   ✅ Shape : {df.shape}")
print(f"   → Colonnes : {df.columns.tolist()[:5]} ...")

# Séparer label et features
X = df.drop('Label', axis=1)
y = df['Label']


# ─────────────────────────────────────────────
# ÉTAPE 1 — Catégorisation des appels API
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 1 : Catégorisation des appels API en groupes...")

toutes_colonnes = X.columns.tolist()

# Groupe 1 : Activité Fichier
cols_fichier = [c for c in toutes_colonnes if any(k in c for k in [
    'CopyFile', 'CreateDirectory', 'DeleteFile', 'FindFirstFile',
    'GetFileAttributes', 'GetFileInformation', 'GetFileSize',
    'GetFileType', 'GetFileVersionInfo', 'GetShortPathName',
    'MoveFile', 'NtCreateFile', 'NtOpenFile', 'NtQueryAttributesFile',
    'NtQueryDirectoryFile', 'NtQueryInformationFile', 'NtReadFile',
    'NtSetInformationFile', 'NtWriteFile', 'RemoveDirectory',
    'SearchPathW', 'SetEndOfFile', 'SetFileAttributes',
    'SetFileInformationByHandle', 'SetFilePointer', 'SetFileTime',
    'SUMMARY_FILE', 'SUMMARY_DIRECTORY', 'NtQueryFullAttributesFile',
    'FindFirstFileExA', 'FindFirstFileExW'
])]

# Groupe 2 : Activité Réseau
cols_reseau = [c for c in toutes_colonnes if any(k in c for k in [
    'InternetReadFile', 'URLDownloadToFileW'
])]

# Groupe 3 : Activité Système
cols_systeme = [c for c in toutes_colonnes if any(k in c for k in [
    'GetSystemDirectory', 'GetSystemTime', 'GetSystemWindows',
    'GetTempPath', 'DeviceIoControl', 'NtDeviceIoControlFile'
])]

# Groupe 4 : Activité Volume/Stockage
cols_volume = [c for c in toutes_colonnes if any(k in c for k in [
    'GetVolume', 'GetVolumeNameForVolumeMountPointW',
    'GetVolumePathNamesForVolumeNameW', 'GetVolumePathNameW'
])]

print(f"   → Fichier    : {len(cols_fichier)} colonnes")
print(f"   → Réseau     : {len(cols_reseau)} colonnes")
print(f"   → Système    : {len(cols_systeme)} colonnes")
print(f"   → Volume     : {len(cols_volume)} colonnes")

# Créer les métriques agrégées
df['activite_fichier']  = df[cols_fichier].sum(axis=1)  if cols_fichier  else 0
df['activite_reseau']   = df[cols_reseau].sum(axis=1)   if cols_reseau   else 0
df['activite_systeme']  = df[cols_systeme].sum(axis=1)  if cols_systeme  else 0
df['activite_volume']   = df[cols_volume].sum(axis=1)   if cols_volume   else 0

print("   ✅ 4 métriques d'activité créées")


# ─────────────────────────────────────────────
# GRAPHE 1 — Distribution des métriques par classe
# ─────────────────────────────────────────────
print("\n📊 Graphe 1 : Distribution des métriques d'activité...")

metriques = ['activite_fichier', 'activite_reseau', 'activite_systeme', 'activite_volume']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distribution des Métriques d'Activité\npar Type de Logiciel", fontsize=14, fontweight='bold')

axes_flat = axes.flatten()
colors = {'Ransomware': '#e74c3c', 'Goodware': '#2ecc71'}

for i, metrique in enumerate(metriques):
    for label, color in colors.items():
        vals = df[df['Label'] == label][metrique]
        axes_flat[i].hist(vals, bins=30, alpha=0.6, color=color,
                          label=label, edgecolor='black', linewidth=0.3)
    axes_flat[i].set_title(f"{metrique}", fontsize=11, fontweight='bold')
    axes_flat[i].set_xlabel("Valeur")
    axes_flat[i].set_ylabel("Fréquence")
    axes_flat[i].legend()
    axes_flat[i].set_yscale('log')

plt.tight_layout()
plt.savefig("p2_graphe1_metriques_activite.png", bbox_inches='tight')
plt.show()
print("   ✅ p2_graphe1_metriques_activite.png sauvegardé")


# ─────────────────────────────────────────────
# ÉTAPE 2 — Indicateurs comportementaux booléens
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 2 : Création des indicateurs comportementaux...")

# Indicateur 1 : Modifie beaucoup de fichiers
cols_ecriture = [c for c in toutes_colonnes if any(k in c for k in [
    'NtWriteFile', 'SetFileAttributesW', 'MoveFileWithProgressW',
    'DeleteFileW', 'CopyFileW', 'CopyFileA', 'SetEndOfFile'
])]
if cols_ecriture:
    df['modifie_fichiers'] = (df[cols_ecriture].sum(axis=1) > 5).astype(int)
else:
    df['modifie_fichiers'] = 0

# Indicateur 2 : Présence réseau
df['presence_reseau'] = (df['activite_reseau'] > 0).astype(int)

# Indicateur 3 : Chiffrement de fichiers
cols_chiffrement = [c for c in toutes_colonnes if any(k in c for k in [
    'NtWriteFile', 'NtReadFile', 'SetFilePointer',
    'SetFileTime', 'SetEndOfFile'
])]
if cols_chiffrement:
    df['chiffre_fichiers'] = (
        (df[cols_chiffrement].sum(axis=1) > 5) &
        (df['activite_fichier'] > 10)
    ).astype(int)
else:
    df['chiffre_fichiers'] = 0

# Indicateur 4 : Manipule le système
df['manipule_systeme'] = (df['activite_systeme'] > 0).astype(int)

print("   ✅ 4 indicateurs comportementaux créés :")
print(f"      → modifie_fichiers  : {df['modifie_fichiers'].sum()} logiciels concernés")
print(f"      → presence_reseau   : {df['presence_reseau'].sum()} logiciels concernés")
print(f"      → chiffre_fichiers  : {df['chiffre_fichiers'].sum()} logiciels concernés")
print(f"      → manipule_systeme  : {df['manipule_systeme'].sum()} logiciels concernés")


# ─────────────────────────────────────────────
# GRAPHE 2 — Indicateurs par classe
# ─────────────────────────────────────────────
print("\n📊 Graphe 2 : Indicateurs comportementaux par classe...")

indicateurs = ['modifie_fichiers', 'presence_reseau', 'chiffre_fichiers', 'manipule_systeme']

ransomware_df = df[df['Label'] == 'Ransomware']
goodware_df   = df[df['Label'] == 'Goodware']

pct_ransomware = [ransomware_df[ind].mean() * 100 for ind in indicateurs]
pct_goodware   = [goodware_df[ind].mean() * 100   for ind in indicateurs]

x_pos = np.arange(len(indicateurs))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Indicateurs Comportementaux : Ransomware vs Goodware", fontsize=14, fontweight='bold')

# Barplot groupé
bars1 = axes[0].bar(x_pos - width/2, pct_ransomware, width, label='Ransomware',
                     color='#e74c3c', edgecolor='black', alpha=0.85)
bars2 = axes[0].bar(x_pos + width/2, pct_goodware, width, label='Goodware',
                     color='#2ecc71', edgecolor='black', alpha=0.85)
axes[0].set_title("% de logiciels présentant chaque indicateur")
axes[0].set_ylabel("% de logiciels")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(indicateurs, rotation=20, ha='right')
axes[0].legend()
axes[0].set_ylim(0, 120)
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=8, fontweight='bold')
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=8, fontweight='bold')

# Heatmap des indicateurs
ind_data = df[indicateurs + ['Label']].copy()
ind_data['Label_num'] = (ind_data['Label'] == 'Ransomware').astype(int)
corr_ind = ind_data[indicateurs + ['Label_num']].corr()
sns.heatmap(corr_ind, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=axes[1], linewidths=0.5, square=True)
axes[1].set_title("Corrélation indicateurs vs Label")

plt.tight_layout()
plt.savefig("p2_graphe2_indicateurs_comportementaux.png", bbox_inches='tight')
plt.show()
print("   ✅ p2_graphe2_indicateurs_comportementaux.png sauvegardé")


# ─────────────────────────────────────────────
# ÉTAPE 3 — Score de suspicion
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 3 : Calcul du score de suspicion...")

df['score_suspicion'] = df[['modifie_fichiers', 'presence_reseau',
                             'chiffre_fichiers', 'manipule_systeme']].sum(axis=1)

print("   ✅ Score de suspicion créé (0 à 4)")
print(f"\n   Distribution du score :")
print(df.groupby(['Label', 'score_suspicion']).size().unstack(fill_value=0))


# ─────────────────────────────────────────────
# GRAPHE 3 — Score de suspicion
# ─────────────────────────────────────────────
print("\n📊 Graphe 3 : Score de suspicion...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Score de Suspicion : Ransomware vs Goodware", fontsize=14, fontweight='bold')

scores = [0, 1, 2, 3, 4]
for label, color in [('Ransomware', '#e74c3c'), ('Goodware', '#2ecc71')]:
    counts = [len(df[(df['Label'] == label) & (df['score_suspicion'] == s)]) for s in scores]
    axes[0].plot(scores, counts, marker='o', color=color, label=label, linewidth=2, markersize=8)
    axes[0].fill_between(scores, counts, alpha=0.2, color=color)

axes[0].set_title("Distribution du score de suspicion")
axes[0].set_xlabel("Score de suspicion (0=bénin, 4=très suspect)")
axes[0].set_ylabel("Nombre de logiciels")
axes[0].set_xticks(scores)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Boxplot score par classe
data_r = df[df['Label'] == 'Ransomware']['score_suspicion']
data_g = df[df['Label'] == 'Goodware']['score_suspicion']
bp = axes[1].boxplot([data_g, data_r], labels=['Goodware', 'Ransomware'],
                      patch_artist=True, notch=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
axes[1].set_title("Boxplot du score de suspicion")
axes[1].set_ylabel("Score")

plt.tight_layout()
plt.savefig("p2_graphe3_score_suspicion.png", bbox_inches='tight')
plt.show()
print("   ✅ p2_graphe3_score_suspicion.png sauvegardé")


# ─────────────────────────────────────────────
# ÉTAPE 4 — Ratios d'activité
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 4 : Calcul des ratios d'activité...")

total_api = df[toutes_colonnes].sum(axis=1)

df['ratio_fichier_total']    = df['activite_fichier']  / (total_api + 1)
df['ratio_reseau_total']     = df['activite_reseau']   / (total_api + 1)
df['ratio_systeme_total']    = df['activite_systeme']  / (total_api + 1)
df['ratio_fichier_systeme']  = df['activite_fichier']  / (df['activite_systeme'] + 1)

print("   ✅ 4 ratios créés :")
print("      → ratio_fichier_total   : % d'activité fichier sur le total")
print("      → ratio_reseau_total    : % d'activité réseau sur le total")
print("      → ratio_systeme_total   : % d'activité système sur le total")
print("      → ratio_fichier_systeme : rapport fichier/système")


# ─────────────────────────────────────────────
# GRAPHE 4 — Ratios par classe
# ─────────────────────────────────────────────
print("\n📊 Graphe 4 : Ratios d'activité...")

ratios = ['ratio_fichier_total', 'ratio_reseau_total',
          'ratio_systeme_total', 'ratio_fichier_systeme']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Ratios d'Activité : Ransomware vs Goodware", fontsize=14, fontweight='bold')
axes_flat = axes.flatten()

for i, ratio in enumerate(ratios):
    for label, color in [('Ransomware', '#e74c3c'), ('Goodware', '#2ecc71')]:
        vals = df[df['Label'] == label][ratio]
        axes_flat[i].hist(vals, bins=30, alpha=0.6, color=color,
                          label=label, edgecolor='black', linewidth=0.3)
    axes_flat[i].set_title(ratio, fontsize=10, fontweight='bold')
    axes_flat[i].set_xlabel("Valeur du ratio")
    axes_flat[i].set_ylabel("Fréquence")
    axes_flat[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig("p2_graphe4_ratios_activite.png", bbox_inches='tight')
plt.show()
print("   ✅ p2_graphe4_ratios_activite.png sauvegardé")


# ─────────────────────────────────────────────
# SAUVEGARDE CSV 1 — dataSansDoublons.csv
# ─────────────────────────────────────────────
print("\n💾 Sauvegarde CSV 1 : dataSansDoublons.csv...")
df.to_csv("dataSansDoublons.csv", index=False)
print(f"   ✅ dataSansDoublons.csv — Shape : {df.shape}")


# ─────────────────────────────────────────────
# ÉTAPE 5 — Transformation logarithmique
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 5 : Transformation logarithmique log(x+1)...")

df_log = df.copy()
colonnes_a_transformer = [c for c in df_log.columns
                           if c not in ['Label', 'modifie_fichiers', 'presence_reseau',
                                        'chiffre_fichiers', 'manipule_systeme', 'score_suspicion']]

nb_transformees = 0
for col in colonnes_a_transformer:
    if df_log[col].dtype in ['int64', 'float64']:
        if not df_log[col].isin([0, 1]).all():
            df_log[col] = np.log1p(df_log[col])
            nb_transformees += 1

print(f"   ✅ {nb_transformees} colonnes transformées en log(x+1)")


# ─────────────────────────────────────────────
# GRAPHE 5 — Avant / Après transformation log
# ─────────────────────────────────────────────
print("\n📊 Graphe 5 : Avant/Après transformation logarithmique...")

exemple_col = 'activite_fichier'
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Transformation Logarithmique — Exemple : '{exemple_col}'",
             fontsize=13, fontweight='bold')

axes[0].hist(df[exemple_col], bins=40, color='#e74c3c', edgecolor='black', alpha=0.8)
axes[0].set_title("AVANT — Distribution originale")
axes[0].set_xlabel("Valeur brute")
axes[0].set_ylabel("Fréquence")

axes[1].hist(df_log[exemple_col], bins=40, color='#2ecc71', edgecolor='black', alpha=0.8)
axes[1].set_title("APRÈS — Distribution log(x+1)")
axes[1].set_xlabel("log(valeur + 1)")
axes[1].set_ylabel("Fréquence")

plt.tight_layout()
plt.savefig("p2_graphe5_log_transformation.png", bbox_inches='tight')
plt.show()
print("   ✅ p2_graphe5_log_transformation.png sauvegardé")


# SAUVEGARDE CSV 2 — dataLogTransforme.csv
print("\n💾 Sauvegarde CSV 2 : dataLogTransforme.csv...")
df_log.to_csv("dataLogTransforme.csv", index=False)
print(f"   ✅ dataLogTransforme.csv — Shape : {df_log.shape}")


# ─────────────────────────────────────────────
# ÉTAPE 6 — Encodage binaire
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 6 : Encodage binaire (présence/absence)...")

df_binaire = df.copy()
colonnes_binaires = [c for c in df_binaire.columns if c != 'Label']

for col in colonnes_binaires:
    if df_binaire[col].dtype == 'bool':
        df_binaire[col] = df_binaire[col].astype(int)
    elif pd.api.types.is_numeric_dtype(df_binaire[col]):
        df_binaire[col] = (df_binaire[col] > 0).astype(int)

print(f"   ✅ Toutes les colonnes numériques converties en 0/1")
print(f"   → Exemple activite_fichier :")
print(f"      Avant : {df['activite_fichier'].head(5).tolist()}")
print(f"      Après : {df_binaire['activite_fichier'].head(5).tolist()}")


# ─────────────────────────────────────────────
# GRAPHE 6 — Comparaison des 3 versions
# ─────────────────────────────────────────────
print("\n📊 Graphe 6 : Comparaison des 3 versions du dataset...")

col_exemple = 'activite_fichier'
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Comparaison des 3 Versions du Dataset — '{col_exemple}'",
             fontsize=13, fontweight='bold')

datasets = [
    (df, "Version Brute\n(dataSansDoublons)", '#3498db'),
    (df_log, "Version Log\n(dataLogTransforme)", '#e67e22'),
    (df_binaire, "Version Binaire\n(dataBinaire)", '#9b59b6')
]

for i, (data, titre, color) in enumerate(datasets):
    axes[i].hist(data[col_exemple], bins=30, color=color, edgecolor='black', alpha=0.85)
    axes[i].set_title(titre, fontweight='bold')
    axes[i].set_xlabel("Valeur")
    axes[i].set_ylabel("Fréquence")
    mean_val = data[col_exemple].mean()
    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Moyenne={mean_val:.2f}')
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig("p2_graphe6_comparaison_versions.png", bbox_inches='tight')
plt.show()
print("   ✅ p2_graphe6_comparaison_versions.png sauvegardé")


# SAUVEGARDE CSV 3 — dataBinaire.csv
print("\n💾 Sauvegarde CSV 3 : dataBinaire.csv...")
df_binaire.to_csv("dataBinaire.csv", index=False)
print(f"   ✅ dataBinaire.csv — Shape : {df_binaire.shape}")


# ─────────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ PARTIE 2 TERMINÉE AVEC SUCCÈS !")
print("=" * 60)

print(f"\n📊 Nouvelles features créées :")
print(f"   → 4 métriques d'activité   : activite_fichier, activite_reseau, ...")
print(f"   → 4 indicateurs booléens   : modifie_fichiers, presence_reseau, ...")
print(f"   → 1 score de suspicion     : score_suspicion (0 à 4)")
print(f"   → 4 ratios d'activité      : ratio_fichier_total, ...")
print(f"   → Total features ajoutées  : 13 nouvelles colonnes")

print(f"\n📁 3 CSV générés :")
print(f"   • dataSansDoublons.csv    ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
print(f"   • dataLogTransforme.csv   ({df_log.shape[0]} lignes, {df_log.shape[1]} colonnes)")
print(f"   • dataBinaire.csv         ({df_binaire.shape[0]} lignes, {df_binaire.shape[1]} colonnes)")

print(f"\n📊 6 Graphes générés :")
print("   • p2_graphe1_metriques_activite.png")
print("   • p2_graphe2_indicateurs_comportementaux.png")
print("   • p2_graphe3_score_suspicion.png")
print("   • p2_graphe4_ratios_activite.png")
print("   • p2_graphe5_log_transformation.png")
print("   • p2_graphe6_comparaison_versions.png")

print("\n*** Donner les 3 CSV a la Personne 3 pour le ML ! ***")