# TP2 : Clustering avec l'algorithme K-Means 📊

## 📝 Sujet du TP

Ce TP explore l’algorithme **K-Means** pour effectuer du clustering (classification non supervisée) sur des jeux de données variés.  
On utilise principalement **scikit-learn**, tout en proposant une **implémentation manuelle** de K-Means en Python.

> 📄 [Télécharger le sujet complet (PDF)](./TP2_Kmeans_algorithm.pdf)

---

## 📁 Contenu du dépôt

| Fichier                           | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `exercice1.ipynb`                 | Clustering de données synthétiques avec `make_blobs()`                      |
| `exercice2.ipynb`                 | Clustering sur le dataset Iris (sans les étiquettes)                        |
| `exercice3.ipynb`                 | Clustering sur le dataset Wine (avec PCA et normalisation)                 |
| `exercice4.ipynb`                 | Application de K-Means aux datasets Titanic et Fromage                     |
| `ex5_implement_kmean.py`          | Implémentation manuelle de l’algorithme K-Means sans `sklearn`             |
| `test_exercice5.ipynb`            | Notebook de test pour valider le fonctionnement de `ex5_implement_kmean.py`|

---

## 🧠 Concepts clés

- Algorithme K-Means (assignation, mise à jour, convergence)
- Initialisation (`k-means++`, `random`)
- Évaluation : ARI, NMI, Silhouette Score
- Réduction de dimension : PCA
- Normalisation des données
- Implémentation from scratch en Python

---

## 🛠️ Bibliothèques utilisées

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 🎯 Objectifs

- Appliquer le clustering non supervisé sur divers jeux de données
- Visualiser et analyser les résultats des clusters
- Comparer l’impact de la normalisation et de la réduction de dimension
- Implémenter et tester l’algorithme K-Means soi-même

---

📄 Sujet complet : [TP2_Kmeans_algorithm.pdf](./TP2_Kmeans_algorithm.pdf)
