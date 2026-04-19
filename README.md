# Ransomware Detection - DATA Project

## 📋 Description

Ce projet implémente un pipeline complet d'apprentissage automatique et de deep learning pour la **détection et classification de logiciels malveillants (ransomware) versus logiciels bénins (goodware)**. 

Le projet utilise un dataset basé sur les **API de fichiers** pour extraire des caractéristiques comportementales et les utiliser dans différents modèles de classification.

---

## 🎯 Objectifs

- ✅ Prétraitement et nettoyage des données
- ✅ Ingénierie des caractéristiques (Feature Engineering)
- ✅ Normalisation des données et réduction dimensionnelle (PCA)
- ✅ Entraînement de modèles de Machine Learning
- ✅ Développement de modèles de Deep Learning
- ✅ Comparaison des performances entre les approches

---

## 📂 Structure du Projet

```
project_py/
├── partie1_preprocessing.py              # Analyse exploratoire + nettoyage
├── Partie2_feature_engineering.py        # Ingénierie des caractéristiques
├── Partie3_normalisation_pca.py          # Normalisation + Sélection + PCA
├── Partie4_machine_learning.py           # Modèles de ML classiques
├── partie5_deep_learning.py              # Réseaux de neurones
├── Partie6_comparaisons.py               # Comparaison des résultats
├── Ransomware_and_Goodware_File_API_Dataset.csv  # Dataset d'entrée
├── requirements.txt                      # Dépendances du projet
└── README.md                             # Ce fichier
```

---

## 📊 Dataset

**Nom :** Ransomware_and_Goodware_File_API_Dataset.csv

**Description :**
- Contient des enregistrements de fichiers avec leurs appels API
- Chaque ligne représente un fichier avec ses caractéristiques comportementales
- Étiquette binaire : **Ransomware** (1) ou **Goodware** (0)

**Caractéristiques :**
- Basées sur les appels API de fichiers
- Plusieurs centaines de colonnes numériques et catégoriques

---

## 🔄 Pipeline d'Exécution

### **Partie 1 : Prétraitement**
- Chargement du dataset
- Exploration statistique
- Analyse des valeurs manquantes
- Nettoyage et transformation des données
- Visualisation de la distribution des classes

**Exécution :**
```bash
python partie1_preprocessing.py
```

### **Partie 2 : Feature Engineering**
- Sélection des caractéristiques pertinentes
- Création de nouvelles variables
- Transformations mathématiques
- Réduction du bruit

**Exécution :**
```bash
python Partie2_feature_engineering.py
```

### **Partie 3 : Normalisation & PCA**
- Standardisation des données (StandardScaler, MinMaxScaler)
- Sélection des meilleures caractéristiques (SelectKBest, RFE)
- Réduction dimensionnelle par PCA
- Analyse de la variance expliquée

**Exécution :**
```bash
python Partie3_normalisation_pca.py
```

### **Partie 4 : Machine Learning**
- Entraînement de modèles classiques :
  - Régression Logistique
  - Random Forest
  - SVM
  - Gradient Boosting
  - Etc.
- Évaluation avec Cross-Validation
- Rapport de classification

**Exécution :**
```bash
python Partie4_machine_learning.py
```

### **Partie 5 : Deep Learning**
- Construction de réseaux de neurones
- Entraînement et validation
- Ajustement des hyperparamètres
- Analyse de la convergence

**Exécution :**
```bash
python partie5_deep_learning.py
```

### **Partie 6 : Comparaisons**
- Synthèse des performances
- Graphiques comparatifs
- Métriques : Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Recommandations sur le meilleur modèle

**Exécution :**
```bash
python Partie6_comparaisons.py
```

---

## 🛠️ Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Git (optionnel)

### Installation des Dépendances

**Méthode recommandée (avec requirements.txt) :**

```bash
pip install -r requirements.txt
```

**Ou installation manuelle :**

```bash
pip install pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 seaborn>=0.11.0 scipy>=1.7.0 scikit-learn>=1.0.0 tensorflow>=2.8.0 keras>=2.8.0
```

**Packages principales :**

| Package | Version | Description |
|---------|---------|------------|
| `pandas` | ≥1.3.0 | Manipulation et analyse de données |
| `numpy` | ≥1.21.0 | Calculs numériques et tableaux |
| `matplotlib` | ≥3.4.0 | Visualisations graphiques |
| `seaborn` | ≥0.11.0 | Visualisations statistiques |
| `scipy` | ≥1.7.0 | Calculs et statistiques avancés |
| `scikit-learn` | ≥1.0.0 | Algorithmes de Machine Learning |
| `tensorflow` | ≥2.8.0 | Framework Deep Learning |
| `keras` | ≥2.8.0 | API haut niveau pour TensorFlow |

---

## 🚀 Utilisation

### Exécuter le pipeline complet

```bash
# Étape par étape
python partie1_preprocessing.py
python Partie2_feature_engineering.py
python Partie3_normalisation_pca.py
python Partie4_machine_learning.py
python partie5_deep_learning.py
python Partie6_comparaisons.py
```

### Exécuter une partie spécifique

```bash
# Seulement le preprocessing
python partie1_preprocessing.py

# Seulement les modèles ML
python Partie4_machine_learning.py
```

---

## 📈 Résultats Attendus

Chaque script génère :
- **Graphiques** : Distributions, corrélations, courbes d'apprentissage, ROC-AUC
- **Fichiers CSV** : Datasets nettoyés et transformés (dataset_clean_partie1.csv, etc.)
- **Rapports** : Métriques de performance, matrices de confusion
- **Console** : Affichage des étapes et statistiques principales

---

## 📊 Métriques de Performance

Les modèles sont évalués selon :

| Métrique | Description |
|----------|------------|
| **Accuracy** | Proportion de prédictions correctes |
| **Precision** | Proportion de positifs correctement identifiés |
| **Recall** | Proportion de positifs détectés |
| **F1-Score** | Moyenne harmonique de Precision et Recall |
| **ROC-AUC** | Aire sous la courbe ROC |

---

## 🔍 Observations Clés

- Distribution des classes dans le dataset
- Importance relative des caractéristiques
- Variance expliquée par PCA
- Comparaison ML vs Deep Learning
- Temps d'entraînement et de prédiction

---

## 👥 Auteurs

Projet réalisé par : Hiba OUDIL - Ikram SAAIDI - Kaoutar BAALA

---

## 📝 Notes

- Les datasets intermédiaires sont générés automatiquement lors de chaque exécution
- Les visualisations sont sauvegardées dans le dossier du projet
- Adapter les chemins des fichiers si nécessaire
- Consulter les commentaires dans les scripts pour plus de détails techniques

---

