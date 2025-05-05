# --- STEP 1: Imports ---
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- STEP 2: Load Embeddings (just to get Song, Artist, Population info) ---
def load_embeddings(path, population_label):
    with open(path, 'r') as f:
        data = json.load(f)
    rows = []
    for entry in data:
        artist = entry["artist"]
        song = entry["song"]
        embedding_matrix = np.array(entry["embedding"])
        agg_embedding = np.mean(embedding_matrix, axis=0)
        rows.append({
            "Song": song,
            "Artist": artist,
            "Population": population_label,
            "Embedding": agg_embedding
        })
    return rows

before = load_embeddings("song_embeddings/before_2012_effnet_embeddings.json", "Before 2012")
after = load_embeddings("song_embeddings/after_2018_effnet_embeddings.json", "After 2018")
embedding_data = before + after
embedding_df = pd.DataFrame(embedding_data)

# --- STEP 3: Load original metadata CSV ---
metadata_df = pd.read_csv("metadata/full_metadata.csv")

# --- STEP 4: Merge metadata with embedding metadata ---
merged_df = pd.merge(metadata_df, embedding_df, left_on=["Song Name", "Band"], right_on=["Song", "Artist"])

# --- STEP 5: Encode categorical metadata for PCA ---
features = merged_df[["Instrumentation", "Genre", "Acoustic vs Electronic", "Gender Voice", "Bpm"]].copy()

# One-hot encode categorical columns
categorical_cols = ["Genre", "Acoustic vs Electronic", "Gender Voice"]
features_encoded = pd.get_dummies(features, columns=categorical_cols)

# Standardize numeric values (including "Instrumentation" and "Bpm")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_encoded)

# --- STEP 6: Apply PCA ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# --- STEP 7: Add PCA results to DataFrame ---
merged_df["pca1"] = pca_result[:, 0]
merged_df["pca2"] = pca_result[:, 1]

# --- STEP 8: Plot ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=merged_df,
    x="pca1",
    y="pca2",
    hue="Band",
    style="Population",
    s=80
)
plt.title("PCA on Metadata (Excluding Audio Embeddings)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# --- STEP 9: Save plot ---
output_dir = "visualization_results/pca_metadata"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "pca_metadata_plot.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ PCA metadata visualization saved to: {output_path}")
# --- STEP 10: Save merged DataFrame to CSV ---