import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

class UKMeans:
    def __init__(self, gamma=1.0, beta=1.0, epsilon=1e-4, max_iter=100):
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.max_iter = max_iter

    def fit(self, X):
        n, d = X.shape
        c = n  # Initial clusters equal to the number of points
        self.centers = X.copy()
        alpha = np.ones(c) / c
        t = 0

        while t < self.max_iter:
            distances = np.linalg.norm(X[:, None] - self.centers, axis=2)
            weighted_distances = distances - self.gamma * np.log(alpha + 1e-10)
            z = np.argmin(weighted_distances, axis=1)

            new_centers = np.array([X[z == k].mean(axis=0) if np.sum(z == k) > 0 else self.centers[k] for k in range(c)])
            valid_clusters = np.array([np.sum(z == k) > 0 for k in range(c)])
            self.centers = new_centers[valid_clusters]
            alpha = alpha[valid_clusters]
            c = len(self.centers)

            alpha = np.array([np.sum(z == k) / n for k in range(c)])
            alpha /= np.sum(alpha)

            if np.linalg.norm(self.centers - new_centers[valid_clusters]) < self.epsilon:
                break

            t += 1

        self.cluster_labels_ = z
        self.cluster_centers_ = self.centers

    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

# Generate synthetic dataset
X, _ = make_blobs(n_samples=400, centers=6, cluster_std=0.5, random_state=42)

# Apply U-k-means
ukmeans = UKMeans(gamma=0.5, beta=0.5, epsilon=1e-3)
ukmeans.fit(X)
uk_labels = ukmeans.cluster_labels_

# Apply traditional k-means
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.labels_

# Plot results
plt.figure(figsize=(12, 6))

# U-k-means result
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=uk_labels, cmap='viridis', s=30)
plt.scatter(ukmeans.cluster_centers_[:, 0], ukmeans.cluster_centers_[:, 1], color='red', marker='x')
plt.title("U-k-means Clustering")

# k-means result
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x')
plt.title("Traditional k-means Clustering")

plt.show()
