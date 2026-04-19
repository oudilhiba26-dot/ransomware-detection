# import pandas as pd
import numpy as np
import pandas as pd
# Split
from sklearn.model_selection import train_test_split

# Modèles
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Évaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("dataset_final_ML.csv")
print(df.head())
X = df.drop("label", axis=1)  # remplacer "target"
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP (Neural Network)

from sklearn.neural_network import MLPClassifier

print("\n MLP (NEURAL NETWORK) ")
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
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)

print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))



# LightGBM

import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n LightGBM ")

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

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

y_pred_lgbm = lgbm.predict(X_test)

print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgbm))
print(classification_report(y_test, y_pred_lgbm))

# --- Matrice de confusion ---
cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lgbm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - LightGBM")
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("confusion_matrix_lgbm.png", dpi=120)
plt.show()

# --- Importance des features ---
lgb.plot_importance(lgbm, max_num_features=15, figsize=(8, 6), title="Feature Importance - LightGBM")
plt.tight_layout()
plt.savefig("feature_importance_lgbm.png", dpi=120)
plt.show()