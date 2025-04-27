import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

# ------------------------
# Cargar datos
# ------------------------

# Carga embeddings
# embeddings = np.load('embeddings.npy')  # (120, 512)
# labels = pd.read_csv('metadata.csv')  # contiene las columnas de features

# ----------- Simulación de los embeddings y las features (Datos de ejemplo) -----------
X_embeddings = np.random.rand(120, 512)  # 120 canciones, 512 dimensiones
df_features = pd.DataFrame({
    'instrumental': np.random.choice([1, 2, 3, 4, 5], 120),
    'genre': np.random.choice(['pop', 'rock', 'folk', 'rap'], 120),
    'acoustic_electronic': np.random.choice(['acoustic', 'electronic'], 120),
    'gender_voice': np.random.choice(['male', 'female', 'mixed'], 120),
    'bpm': np.random.randint(60, 180, 120)
})

# Lista de features a colorear
features = ['instrumental', 'genre', 'acoustic_electronic', 'gender_voice', 'bpm']

# Crear carpeta de screenshots si no existe
output_folder = "screenshots"
os.makedirs(output_folder, exist_ok=True)

# Dimensionality Reduction (Usando t-SNE, podemos cambiar a PCA o UMAP)
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X_embeddings)

# Crear una figura grande con subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 filas, 3 columnas
axes = axes.flatten()  # hace más fácil acceder a cada subplot

# ================================================
# Loop para hacer un plot para cada feature
# ================================================

for idx, feat in enumerate(features):
    ax = axes[idx]

    # Plot en el subplot correspondiente
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=df_features[feat], palette="tab10", ax=ax)
    ax.set_title(f"t-SNE colored by {feat}")
    ax.legend(title=feat, loc='best', fontsize='small')

    # Guardar cada plot individualmente
    single_fig, single_ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=df_features[feat], palette="tab10", ax=single_ax)
    single_ax.set_title(f"t-SNE colored by {feat}")
    single_ax.legend(title=feat, loc='best', fontsize='small')
    single_fig.savefig(os.path.join(output_folder, f"tsne_colored_by_{feat}.png"))
    plt.close(single_fig)  # Cerrar figura individual después de guardar para no consumir memoria

# Borrar el subplot vacío si hay más subplots de la cuenta
if len(features) < len(axes):
    fig.delaxes(axes[-1])

# Ajustar diseño y mostrar todo en una ventana
plt.tight_layout()
plt.show()
