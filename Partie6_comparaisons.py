# ═══════════════════════════════════════════════════════════
# COMPARAISON DE TOUS LES MODÈLES (incluant LightGBM)
# ═══════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from  sklearn.neural_network import MLPClassifier   
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data directly
df = pd.read_csv("dataset_final_ML.csv")
X = df.drop('label', axis=1).values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a namespace object to hold the data (for compatibility with existing code)
class P5:
    pass
p5 = P5()
p5.X_train = X_train
p5.X_test = X_test
p5.y_train = y_train
p5.y_test = y_test

lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    learning_rate='adaptive',
    alpha=0.001
)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000)

# ─── Fit all models ──────────────────────────────────────────────────────────
svm_proba = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
svm_proba.fit(p5.X_train, p5.y_train)

rf.fit(p5.X_train, p5.y_train)
lr.fit(p5.X_train, p5.y_train)
xgb_model.fit(p5.X_train, p5.y_train)
mlp.fit(p5.X_train, p5.y_train)
lgbm.fit(p5.X_train, p5.y_train)

plt.rcParams['figure.dpi'] = 120
sns.set_style("whitegrid")

# ─── Probas pour ROC (SVM nécessite probability=True) ───────────────────────
y_pred_svm   = svm_proba.predict(p5.X_test)
y_proba_svm  = svm_proba.predict_proba(p5.X_test)[:, 1]

y_pred_rf    = rf.predict(p5.X_test)
y_proba_rf   = rf.predict_proba(p5.X_test)[:, 1]

y_pred_lr    = lr.predict(p5.X_test)
y_proba_lr   = lr.predict_proba(p5.X_test)[:, 1]

y_pred_xgb   = xgb_model.predict(p5.X_test)
y_proba_xgb  = xgb_model.predict_proba(p5.X_test)[:, 1]

y_pred_mlp   = mlp.predict(p5.X_test)
y_proba_mlp  = mlp.predict_proba(p5.X_test)[:, 1]

y_pred_lgbm  = lgbm.predict(p5.X_test)
y_proba_lgbm = lgbm.predict_proba(p5.X_test)[:, 1]   # ← LightGBM

# ─── Dictionnaire des modèles ────────────────────────────────────────────────
models = {
    'Random Forest':      {'y_pred': y_pred_rf,   'y_proba': y_proba_rf,   'model': rf},
    'SVM':                {'y_pred': y_pred_svm,  'y_proba': y_proba_svm,  'model': svm_proba},
    'Logistic Regression':{'y_pred': y_pred_lr,   'y_proba': y_proba_lr,   'model': lr},
    'XGBoost':            {'y_pred': y_pred_xgb,  'y_proba': y_proba_xgb,  'model': xgb_model},
    'MLP':                {'y_pred': y_pred_mlp,  'y_proba': y_proba_mlp,  'model': mlp},
    'LightGBM':           {'y_pred': y_pred_lgbm, 'y_proba': y_proba_lgbm, 'model': lgbm},
}

colors_models = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']

# ─── Tableau comparatif ──────────────────────────────────────────────────────
results = {}
for name, m in models.items():
    results[name] = {
        'Accuracy':  accuracy_score(p5.y_test, m['y_pred']),
        'Precision': precision_score(p5.y_test, m['y_pred']),
        'Recall':    recall_score(p5.y_test, m['y_pred']),
        'F1-Score':  f1_score(p5.y_test, m['y_pred']),
        'AUC-ROC':   roc_auc_score(p5.y_test, m['y_proba']),
    }

comparison = pd.DataFrame(results).T.sort_values('F1-Score', ascending=False)

print("\n" + "=" * 65)
print("  TABLEAU COMPARATIF DE TOUS LES MODÈLES")
print("=" * 65)
print(comparison.to_string())
comparison.to_csv("resultats_comparaison_modeles.csv")
print("\n   resultats_comparaison_modeles.csv sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 1 — Comparaison des métriques (barplot)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Comparaison des 6 Modèles — Ransomware Detection",
             fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2][idx % 2]
    values = comparison[metric].sort_values(ascending=True)
    bar_colors = [colors_models[list(comparison.index).index(n)] for n in values.index]
    bars = ax.barh(values.index, values.values,
                   color=bar_colors, edgecolor='black', height=0.5)
    ax.set_title(f"{metric}", fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.08)
    for bar, val in zip(bars, values.values):
        ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig("p6_graphe1_comparaison_metriques.png", bbox_inches='tight')
plt.show()
print("   p6_graphe1_comparaison_metriques.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 2 — Courbes ROC (tous les modèles)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Courbes ROC — Tous les Modèles", fontsize=14, fontweight='bold')

for i, (name, m) in enumerate(models.items()):
    fpr, tpr, _ = roc_curve(p5.y_test, m['y_proba'])
    auc_val = roc_auc_score(p5.y_test, m['y_proba'])
    lw = 3 if name in ('XGBoost', 'LightGBM') else 2
    ax.plot(fpr, tpr, color=colors_models[i], linewidth=lw,
            label=f"{name} (AUC={auc_val:.4f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUC=0.5)')
ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("p6_graphe2_roc_curves.png", bbox_inches='tight')
plt.show()
print("   p6_graphe2_roc_curves.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 3 — Matrices de confusion (6 modèles)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Matrices de Confusion — 6 Modèles", fontsize=14, fontweight='bold')

for idx, (name, m) in enumerate(models.items()):
    ax = axes[idx // 3][idx % 3]
    cm = confusion_matrix(p5.y_test, m['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Goodware', 'Ransomware'],
                yticklabels=['Goodware', 'Ransomware'],
                annot_kws={'size': 14})
    f1 = f1_score(p5.y_test, m['y_pred'])
    ax.set_title(f"{name}\nF1={f1:.4f}", fontsize=11, fontweight='bold')
    ax.set_ylabel("Réel")
    ax.set_xlabel("Prédit")

plt.tight_layout()
plt.savefig("p6_graphe3_confusion_matrices.png", bbox_inches='tight')
plt.show()
print("   p6_graphe3_confusion_matrices.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 4 — Cross-Validation F1 Boxplot (10-Fold)
# ════════════════════════════════════════════════════════════════════════════
print("\n Cross-Validation 10-Fold en cours...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

X_full = np.vstack([p5.X_train, p5.X_test])
y_full = np.concatenate([p5.y_train, p5.y_test])

cv_results = {}
for name, m in models.items():
    scores = cross_val_score(m['model'], X_full, y_full, cv=cv, scoring='f1')
    cv_results[name] = scores
    print(f"   {name}: CV F1 = {scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(13, 6))
data_bp   = [cv_results[name] for name in cv_results]
labels_bp = list(cv_results.keys())

bp = ax.boxplot(data_bp, labels=labels_bp, patch_artist=True, vert=True)
for patch, color in zip(bp['boxes'], colors_models):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.set_title("Cross-Validation F1-Score (10-Fold Stratifié)", fontsize=14, fontweight='bold')
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_xticklabels(labels_bp, rotation=15, ha='right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

means = [s.mean() for s in data_bp]
ax.scatter(range(1, len(means) + 1), means, color='red', s=80,
           zorder=5, marker='D', label='Moyenne')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("p6_graphe4_cv_boxplot.png", bbox_inches='tight')
plt.show()
print("   p6_graphe4_cv_boxplot.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 5 — Feature Importance (RF + XGBoost)
# ════════════════════════════════════════════════════════════════════════════
feature_names = df.drop("label", axis=1).columns

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Feature Importance — Random Forest vs XGBoost",
             fontsize=14, fontweight='bold')

imp_rf = pd.Series(rf.feature_importances_, index=feature_names)
top15_rf = imp_rf.sort_values(ascending=False).head(15)
top15_rf.sort_values().plot(kind='barh', ax=axes[0], color='#2ecc71', edgecolor='black')
axes[0].set_title("Random Forest", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Importance")

imp_xgb = pd.Series(xgb.feature_importances_, index=feature_names)
top15_xgb = imp_xgb.sort_values(ascending=False).head(15)
top15_xgb.sort_values().plot(kind='barh', ax=axes[1], color='#f39c12', edgecolor='black')
axes[1].set_title("XGBoost", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("p6_graphe5_feature_importance.png", bbox_inches='tight')
plt.show()
print("   p6_graphe5_feature_importance.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 6 — Courbes Precision-Recall
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Courbes Precision-Recall — Tous les Modèles",
             fontsize=14, fontweight='bold')

for i, (name, m) in enumerate(models.items()):
    prec, rec, _ = precision_recall_curve(p5.y_test, m['y_proba'])
    ap = average_precision_score(p5.y_test, m['y_proba'])
    lw = 3 if name in ('XGBoost', 'LightGBM') else 2
    ax.plot(rec, prec, color=colors_models[i], linewidth=lw,
            label=f"{name} (AP={ap:.4f})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("p6_graphe6_precision_recall.png", bbox_inches='tight')
plt.show()
print("   p6_graphe6_precision_recall.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 7 — XGBoost vs LightGBM : Comparaison des métriques (radar)
# ════════════════════════════════════════════════════════════════════════════
metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
xgb_vals  = [results['XGBoost'][m]  for m in metrics_radar]
lgbm_vals = [results['LightGBM'][m] for m in metrics_radar]

angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
xgb_vals  += xgb_vals[:1]
lgbm_vals += lgbm_vals[:1]
angles    += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_title("XGBoost vs LightGBM — Radar des Métriques",
             fontsize=14, fontweight='bold', pad=20)

ax.plot(angles, xgb_vals,  color='#f39c12', linewidth=2.5, label='XGBoost')
ax.fill(angles, xgb_vals,  color='#f39c12', alpha=0.25)

ax.plot(angles, lgbm_vals, color='#1abc9c', linewidth=2.5, label='LightGBM')
ax.fill(angles, lgbm_vals, color='#1abc9c', alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_radar, fontsize=12)
ax.set_ylim(0, 1)
ax.yaxis.set_tick_params(labelsize=8)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

plt.tight_layout()
plt.savefig("p6_graphe7_radar_xgb_lgbm.png", bbox_inches='tight')
plt.show()
print("   p6_graphe7_radar_xgb_lgbm.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 8 — XGBoost vs LightGBM : Feature Importance côte à côte
# ════════════════════════════════════════════════════════════════════════════
imp_xgb  = pd.Series(xgb.feature_importances_,  index=feature_names)
imp_lgbm = pd.Series(lgbm.feature_importances_, index=feature_names)

# Top 15 features union des deux modèles
top_features = list(
    pd.concat([imp_xgb, imp_lgbm], axis=1)
    .max(axis=1)
    .sort_values(ascending=False)
    .head(15)
    .index
)

imp_xgb_top  = imp_xgb[top_features].sort_values()
imp_lgbm_top = imp_lgbm[top_features].sort_values()

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
fig.suptitle("Feature Importance — XGBoost vs LightGBM (Top 15)",
             fontsize=14, fontweight='bold')

imp_xgb_top.plot(kind='barh', ax=axes[0], color='#f39c12', edgecolor='black')
axes[0].set_title("XGBoost", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Importance")
for i, v in enumerate(imp_xgb_top.values):
    axes[0].text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

imp_lgbm_top.plot(kind='barh', ax=axes[1], color='#1abc9c', edgecolor='black')
axes[1].set_title("LightGBM", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Importance")
for i, v in enumerate(imp_lgbm_top.values):
    axes[1].text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("p6_graphe8_feature_importance_xgb_lgbm.png", bbox_inches='tight')
plt.show()
print("   p6_graphe8_feature_importance_xgb_lgbm.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 9 — XGBoost vs LightGBM : Courbes ROC superposées (zoom)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_title("Courbes ROC — XGBoost vs LightGBM (comparaison détaillée)",
             fontsize=13, fontweight='bold')

for name, color in [('XGBoost', '#f39c12'), ('LightGBM', '#1abc9c')]:
    fpr, tpr, _ = roc_curve(p5.y_test, models[name]['y_proba'])
    auc_val = roc_auc_score(p5.y_test, models[name]['y_proba'])
    ax.plot(fpr, tpr, color=color, linewidth=3,
            label=f"{name} (AUC={auc_val:.4f})")
    # Zone sous la courbe
    ax.fill_between(fpr, tpr, alpha=0.08, color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("p6_graphe9_roc_xgb_lgbm.png", bbox_inches='tight')
plt.show()
print("   p6_graphe9_roc_xgb_lgbm.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 10 — XGBoost vs LightGBM : Barplot métriques côte à côte
# ════════════════════════════════════════════════════════════════════════════
metrics_bar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
xgb_scores  = [results['XGBoost'][m]  for m in metrics_bar]
lgbm_scores = [results['LightGBM'][m] for m in metrics_bar]

x = np.arange(len(metrics_bar))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars_xgb  = ax.bar(x - width/2, xgb_scores,  width, label='XGBoost',
                   color='#f39c12', edgecolor='black')
bars_lgbm = ax.bar(x + width/2, lgbm_scores, width, label='LightGBM',
                   color='#1abc9c', edgecolor='black')

ax.set_title("XGBoost vs LightGBM — Toutes les Métriques",
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_bar, fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

for bar in bars_xgb:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.4f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
for bar in bars_lgbm:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.4f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("p6_graphe10_barplot_xgb_lgbm.png", bbox_inches='tight')
plt.show()
print("   p6_graphe10_barplot_xgb_lgbm.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# GRAPHE 11 — XGBoost vs LightGBM : Matrices de confusion côte à côte
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Matrices de Confusion — XGBoost vs LightGBM",
             fontsize=14, fontweight='bold')

for ax, name, cmap in zip(axes,
                           ['XGBoost', 'LightGBM'],
                           ['Oranges', 'Greens']):
    cm = confusion_matrix(p5.y_test, models[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Goodware', 'Ransomware'],
                yticklabels=['Goodware', 'Ransomware'],
                annot_kws={'size': 16})
    f1  = f1_score(p5.y_test, models[name]['y_pred'])
    acc = accuracy_score(p5.y_test, models[name]['y_pred'])
    ax.set_title(f"{name}\nF1={f1:.4f}  |  Acc={acc:.4f}",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Réel")
    ax.set_xlabel("Prédit")

plt.tight_layout()
plt.savefig("p6_graphe11_confusion_xgb_lgbm.png", bbox_inches='tight')
plt.show()
print("   p6_graphe11_confusion_xgb_lgbm.png sauvegardé")


# ════════════════════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  RAPPORT FINAL")
print("=" * 65)

best = comparison.index[0]
print(f"\n MEILLEUR MODÈLE : {best}")
print(f"   Accuracy  : {comparison.loc[best, 'Accuracy']:.4f}")
print(f"   Precision : {comparison.loc[best, 'Precision']:.4f}")
print(f"   Recall    : {comparison.loc[best, 'Recall']:.4f}")
print(f"   F1-Score  : {comparison.loc[best, 'F1-Score']:.4f}")
print(f"   AUC-ROC   : {comparison.loc[best, 'AUC-ROC']:.4f}")

print(f"\n Classification Report ({best}) :")
print(classification_report(p5.y_test, models[best]['y_pred'],
                             target_names=['Goodware', 'Ransomware']))

print("\n─── XGBoost vs LightGBM ──────────────────────────────────")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
    xgb_v  = results['XGBoost'][metric]
    lgbm_v = results['LightGBM'][metric]
    winner = "LightGBM ✓" if lgbm_v > xgb_v else ("XGBoost ✓" if xgb_v > lgbm_v else "Égalité")
    print(f"   {metric:<12} XGBoost={xgb_v:.4f}  LightGBM={lgbm_v:.4f}  → {winner}")

print("\n Classement final :")
for i, (name, row) in enumerate(comparison.iterrows(), 1):
    print(f"   {i}. {name:<25} F1={row['F1-Score']:.4f}  AUC={row['AUC-ROC']:.4f}")

print("\n" + "=" * 65)
print("  GRAPHIQUES GÉNÉRÉS :")
print("=" * 65)
for g in [
    "p6_graphe1_comparaison_metriques.png",
    "p6_graphe2_roc_curves.png",
    "p6_graphe3_confusion_matrices.png",
    "p6_graphe4_cv_boxplot.png",
    "p6_graphe5_feature_importance.png",
    "p6_graphe6_precision_recall.png",
    "p6_graphe7_radar_xgb_lgbm.png",
    "p6_graphe8_feature_importance_xgb_lgbm.png",
    "p6_graphe9_roc_xgb_lgbm.png",
    "p6_graphe10_barplot_xgb_lgbm.png",
    "p6_graphe11_confusion_xgb_lgbm.png",
]:
    print(f"   {g}")

print("\n PARTIE 6 TERMINÉE !")
print("=" * 65)