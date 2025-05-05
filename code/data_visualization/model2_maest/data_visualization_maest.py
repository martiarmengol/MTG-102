# --- STEP 1: Imports ---
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px
import subprocess

# --- STEP 2: Load MAEST Embeddings from PKL files ---
def load_embeddings(path, population_label):
    print(f"Loading MAEST embeddings from: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    rows = []
    for item in data:
        # Extract elements from the tuple
        embedding_data = item[0]  # This is a list of nested lists
        artist = item[1]
        song = item[2]
        
        print(f"Processing {song} by {artist}")
        
        # Convert the nested list structure to a flattened embedding
        # Based on professor's instructions: need to flatten the T, 6, 1, 685, 765 structure
        # The key is to multiply dimensions 685 x 765 as mentioned
        
        # Process first element of the embedding data to get a representative feature
        if embedding_data and isinstance(embedding_data, list):
            # Flatten the nested list structure to get a usable representation
            # This is an approximation since we don't have full details on the exact structure
            flattened_features = []
            
            # Take a sample of elements to avoid memory issues
            sample_size = min(len(embedding_data), 100)  # Sample size to prevent memory issues
            for i in range(0, sample_size):
                # Add embedding features to our flattened representation
                if i < len(embedding_data) and embedding_data[i]:
                    flattened_features.extend(embedding_data[i])
            
            # Convert to numpy array and ensure it has reasonable dimensions
            embedding_vector = np.array(flattened_features[:1000])  # Take first 1000 dimensions
            
            rows.append({
                "Song": song,
                "Artist": artist,
                "Population": population_label,
                "Embedding": embedding_vector
            })
        
    print(f"Loaded {len(rows)} songs from {population_label}")
    return rows

before = load_embeddings("song_embeddings/before_2012_maest_embeddings.pkl", "Before 2012")
after = load_embeddings("song_embeddings/after_2018_maest_embeddings.pkl", "After 2018")
all_songs = before + after


# --- STEP 3: Flatten embeddings ---
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

# --- STEP 4: Apply t-SNE ---
embedding_cols = [col for col in df.columns if col.startswith("e")]
X = df[embedding_cols].values
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)
df["x"] = X_2d[:, 0]
df["y"] = X_2d[:, 1]

# --- STEP 5: Static plot (unchanged) ---
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x="x", y="y", hue="Population", style="Artist")
plt.title("t-SNE of Essentia Audio Embeddings (Colored by Population)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_dir = "visualization_maest_results/embedding_visualization"
os.makedirs(output_dir, exist_ok=True)
png_path = os.path.join(output_dir, "tsne_by_population.png")
plt.savefig(png_path, dpi=300)
plt.close()

# --- STEP 6: Interactive Plot with Custom Artist Shapes ---
artist_symbol_map = {
    "Antonia_Font": "circle",
    "Els_Catarres": "x",
    "Macedonia": "square",
    "Manel": "cross",
    "Marina_Rossell": "diamond",
    "Txarango": "triangle-up",
    "31_fam": "triangle-down",
    "julieta": "triangle-left",
    "la_ludwig_band": "triangle-right",
    "mushkaa": "star",
    "oques_grasses": "hexagon",
    "the_tyets": "pentagon"
}

df["Label"] = df["Artist"] + " - " + df["Song"]

fig = px.scatter(
    df,
    x="x", y="y",
    color="Population",
    symbol="Artist",  # <- assign shape by artist
    symbol_map=artist_symbol_map,  # <- assign fixed shapes
    hover_name="Label",
    title="Interactive t-SNE of Essentia Audio Embeddings",
    labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
    width=1000,
    height=750
)

# 4. Save and open
html_path = os.path.join(output_dir, "tsne_by_population_plotly.html")
fig.write_html(html_path)
print(f"✅ Interactive Plot saved to: {html_path}")

html_full_path = os.path.abspath(html_path)
if html_full_path.startswith("/mnt/"):
    drive_letter = html_full_path[5]
    windows_path = html_full_path.replace(f"/mnt/{drive_letter}/", f"{drive_letter.upper()}:\\").replace("/", "\\")
    subprocess.run(["powershell.exe", "Start-Process", windows_path])
else:
    print("⚠️ Could not convert path to Windows. Please open the HTML file manually.")
