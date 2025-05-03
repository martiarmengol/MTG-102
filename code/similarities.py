import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import json

# --- Load embeddings from JSON and build DataFrame ---
def load_embeddings(path, population_label):
    with open(path, 'r') as f:
        data = json.load(f)
    rows = []
    for entry in data:
        embedding_matrix = np.array(entry["embedding"])
        agg_embedding = np.mean(embedding_matrix, axis=0)
        rows.append({
            "Song": entry["song"],
            "Artist": entry["artist"],
            "Population": population_label,
            "Embedding": agg_embedding
        })
    return rows

before = load_embeddings("song_embeddings/before_2012_effnet_embeddings.json", "Before 2012")
after = load_embeddings("song_embeddings/after_2018_effnet_embeddings.json", "After 2018")
all_songs = before + after

# Create a DataFrame from flattened data
flat_data = []
for song in all_songs:
    row = {
        "Song": song["Song"],
        "Artist": song["Artist"],
        "Population": song["Population"]
    }
    for i, val in enumerate(song["Embedding"]):
        row[f"e{i}"] = val
    flat_data.append(row)

df = pd.DataFrame(flat_data)

# --- Define similarity computation ---
def compute_similarity_metrics(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return {
        "cosine_similarity": cosine_similarity(vec1, vec2)[0][0],
        "euclidean_distance": euclidean(vec1.flatten(), vec2.flatten())
    }

# --- Select two songs by artist + title ---
def get_embedding(df, artist, song):
    row = df[(df["Artist"] == artist) & (df["Song"] == song)]
    if row.empty:
        raise ValueError(f"Song '{song}' by '{artist}' not found.")
    return row.filter(regex="^e\\d+$").values.flatten()

# EXAMPLE: replace these with two songs from your dataset
song1_artist = "Manel"
song1_title = "Boomerang"

song2_artist = "the_tyets"
song2_title = "Olívia"

vec1 = get_embedding(df, song1_artist, song1_title)
vec2 = get_embedding(df, song2_artist, song2_title)

# --- Compute and print similarity ---
results = compute_similarity_metrics(vec1, vec2)
print(f"🔍 Comparing '{song1_title}' by {song1_artist} and '{song2_title}' by {song2_artist}")
print("Cosine Similarity:", results["cosine_similarity"])
print("Euclidean Distance:", results["euclidean_distance"])
