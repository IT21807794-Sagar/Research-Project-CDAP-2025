import pandas as pd
import joblib
import Preprocess as Pre
import cluster as Clu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Load the dataset
file_path = 'Dataset/Dataset.csv'
df = pd.read_csv(file_path)

print(f"Dataset loaded with {len(df)} students")
print("\nFirst 5 records:")
print(df.head())

# 3. Create complete pipeline with preprocessing and clustering
# Train the model
pipeline = Clu.train_cluster_model(Pre.X, Pre.preprocessor)

# 4. Save the entire pipeline (preprocessing + model)
model_path = 'student_clustering_pipeline.joblib'
joblib.dump(pipeline, model_path)
print(f"\nTrained pipeline saved to {model_path}")

# 5. Add cluster labels to original data
df['cluster'] = pipeline.predict(Pre.X)


# 6. Calculate clustering evaluation metrics
def evaluate_clustering(pipeline, X):
    # Get preprocessed data
    X_processed = pipeline.named_steps['preprocessor'].transform(X)

    # Get cluster labels
    labels = pipeline.predict(X)

    # Calculate metrics
    silhouette = silhouette_score(X_processed, labels)


    print("\nClustering Evaluation Metrics:")
    print(f"Silhouette Score: {silhouette:.3f} ")



# Evaluate clustering
evaluate_clustering(pipeline, Pre.X)


# 7. Demonstrate loading and using the saved model
def load_and_predict(model_path, new_data):
    # Load the pipeline
    loaded_pipeline = joblib.load(model_path)

    # Make predictions
    predictions = loaded_pipeline.predict(new_data)

    return predictions


# Test loading and prediction (using the same data as example)
print("\nTesting model loading and prediction...")
sample_data = Pre.X.sample(5, random_state=42)
predictions = load_and_predict(model_path, sample_data)
print("\nSample predictions:")
print(pd.DataFrame({
    'student_id': df.loc[sample_data.index, 'student_id'],
    'cluster_prediction': predictions
}))


# 8. Visualize and analyze clusters (same as before)
def analyze_clusters(df):
    # Visualize clusters using PCA
    X_processed = pipeline.named_steps['preprocessor'].transform(Pre.X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis', s=80)
    plt.title('Student Clusters (PCA Visualization)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()

    # Analyze cluster characteristics
    print("\nCluster characteristics (numerical features):")
    print(df.groupby('cluster').mean(numeric_only=True))

    # Generate detailed cluster profiles
    print("\nDetailed Cluster Profiles:")
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        print(f"\n--- Cluster {cluster} ({len(cluster_data)} students) ---")

        print("\nKey Characteristics:")
        print(f"Average age: {cluster_data['age'].mean():.1f} years")
        interests = ['art_interest', 'music_interest', 'sports_interest',
                     'science_interest', 'storytelling_interest']
        top_interest = cluster_data[interests].mean().idxmax().replace('_interest', '')
        print(f"- Highest interest: {top_interest} ({cluster_data[top_interest + '_interest'].mean():.1f}/5)")

        print("\nLearning Preferences:")
        print(f"Preferred learning style: {cluster_data['learning_style'].mode()[0]}")
        print(f"Common activity preference: {cluster_data['preferred_activity'].mode()[0]}")
        print(f"Typical energy level: {cluster_data['energy_level'].mode()[0]}")


analyze_clusters(df)