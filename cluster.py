import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def train_cluster_model(X, preprocessor, max_clusters=10):
    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)

    # Determine optimal number of clusters
    range_n_clusters = range(2, max_clusters + 1)
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_processed)
        silhouette_avg = silhouette_score(X_processed, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, silhouette score is {silhouette_avg:.3f}")

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(range_n_clusters, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal Number of Clusters')
    plt.grid(True)
    plt.show()

    # Select the number of clusters with the highest silhouette score
    optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_clusters}")

    # Create final pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('cluster', KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10))
    ])

    # Fit the pipeline
    pipeline.fit(X)

    return pipeline
