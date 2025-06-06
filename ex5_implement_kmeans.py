import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """  Implémentation de l'algorithme K-means from scratch  """
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):

        """ Initialisation de l'algorithme kmeans  """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        """ Initialisation des centroides de manière aléatoire parmi les points de données   """
        if self.random_state :
            np.random.seed(self.random_state)

        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def assign_clusters(self, X, centroids):
        """ Assgne chaque point au cluster dont le centroide est le plus proche """
        distances = np.zeros((X.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            distances[:,k] = np.sum((X - centroids[k])**2, axis=1)

        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        """ Met à jour les centroides en calculant la moyenne des points dans chaque cluster """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            # Sélectionne les points appartenant au cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Calcule la moyenne des points dans le cluster
                centroids[k] = np.mean(cluster_points, axis=0)
            
        return centroids
        
    def compute_inertia(self, X, labels, centroids):
        """ Calcule l'inertie (somme des carrés des distances de chaque point à son centroide)  """
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k])**2)

        return inertia

    def fit(self, X):
        """  Ajuste le modèle K-means aux données """
        X = np.array(X)

        self.centroids = self.initialize_centroids(X)

        inertia_history = []

        for interation in range(self.max_iters):
            self.labels = self.assign_clusters(X, self.centroids)
            
            new_centroids = self.update_centroids(X, self.labels)

            inertia = self.compute_inertia(X, self.labels, self.centroids)
            inertia_history.append(inertia)

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X):
        """ Prédit le cluster le plus proche pour chaque échantillon dans X """

        X = np.array(X)
        return self.assign_clusters(X, self.centroids)