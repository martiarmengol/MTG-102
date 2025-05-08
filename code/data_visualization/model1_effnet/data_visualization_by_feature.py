# --- STEP 1: Imports ---
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import re
import unicodedata
import difflib
from plotly.io import write_image

# --- STEP 2: Load Embeddings ---
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
all_songs = before + after

# --- STEP 3: Load metadata from CSVs ---
before_meta = pd.read_csv("database_csv/db_before_2012.csv")
after_meta = pd.read_csv("database_csv/db_after_2018.csv")

# Rename 'Band' column to 'Artist' to match embeddings
before_meta = before_meta.rename(columns={'Band': 'Artist'})
after_meta = after_meta.rename(columns={'Band': 'Artist'})

# Rename 'Song Name' column to 'Song' to match embeddings
before_meta = before_meta.rename(columns={'Song Name': 'Song'})
after_meta = after_meta.rename(columns={'Song Name': 'Song'})

# --- STEP 4: Normalizar nombres para que coincidan con los embeddings ---
def remove_accents(input_str):
    """Eliminar acentos y normalizar caracteres unicode"""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalize_name(name):
    """Convierte nombres a formato similar al de los embeddings: minúsculas y guiones bajos"""
    # Convertir a string y eliminar espacios al inicio/fin
    name_str = str(name).strip()
    
    # Eliminar acentos
    name_str = remove_accents(name_str)
    
    # Reemplazar puntuación y espacios con guiones bajos
    normalized = re.sub(r'[,\.\'\"\(\)\[\]\{\}\-\/\\\s]+', '_', name_str)
    
    # Reducir guiones bajos múltiples a uno solo
    normalized = re.sub(r'_+', '_', normalized)
    
    # Eliminar guiones bajos al inicio y fin
    normalized = normalized.strip('_')
    
    # Convertir a minúsculas
    normalized = normalized.lower()
    
    # Eliminar otros caracteres especiales
    normalized = re.sub(r'[^\w_]', '', normalized)
    
    return normalized

# Crear versiones normalizadas de Artist y Song en los metadatos
before_meta['Artist_Norm'] = before_meta['Artist'].apply(normalize_name)
before_meta['Song_Norm'] = before_meta['Song'].apply(normalize_name)
after_meta['Artist_Norm'] = after_meta['Artist'].apply(normalize_name)
after_meta['Song_Norm'] = after_meta['Song'].apply(normalize_name)

# --- STEP 5: Flatten embeddings ---
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

# --- STEP 6: Merge metadata with embeddings ---
# Combine metadata
before_meta['Population'] = 'Before 2012'
after_meta['Population'] = 'After 2018'
all_meta = pd.concat([before_meta, after_meta], ignore_index=True)

# Select only the features we need
meta_features = all_meta[['Artist', 'Song', 'Artist_Norm', 'Song_Norm', 'Population', 'Instrumentation', 'Genre', 'Acoustic vs Electronic', 'Gender Voice', 'Bpm']]

# Función para encontrar la mejor coincidencia
def find_best_match(artist, song, meta_features, threshold=0.85):
    """Encuentra la mejor coincidencia basada en similitud de strings"""
    # Primero intenta coincidencia exacta
    exact_match = meta_features[(meta_features['Artist_Norm'] == artist) & (meta_features['Song_Norm'] == song)]
    if not exact_match.empty:
        return exact_match.iloc[0]
    
    # Luego intenta coincidencia fuzzy
    best_score = 0
    best_match = None
    
    # Filtra primero por artista para mejorar rendimiento
    artist_matches = meta_features[meta_features['Artist_Norm'].str.contains(artist[:4], regex=False)]
    
    if not artist_matches.empty:
        for _, row in artist_matches.iterrows():
            # Calcular similitud para artista y canción
            artist_score = difflib.SequenceMatcher(None, artist, row['Artist_Norm']).ratio()
            song_score = difflib.SequenceMatcher(None, song, row['Song_Norm']).ratio()
            
            # Promedio ponderado (damos más peso a la coincidencia de canciones)
            combined_score = (artist_score * 0.4) + (song_score * 0.6)
            
            if combined_score > best_score and combined_score >= threshold:
                best_score = combined_score
                best_match = row
    
    return best_match

# Para cada embedding, encuentra la fila correspondiente en los metadatos
result_rows = []
match_count = 0
fuzzy_match_count = 0

for _, row in df.iterrows():
    artist = row['Artist']
    song = row['Song']
    
    # Intentar normalizar el nombre del artista y canción del embedding
    artist_norm = normalize_name(artist)
    song_norm = normalize_name(song)
    
    # Buscar coincidencia exacta
    matching_meta = meta_features[
        (meta_features['Artist_Norm'] == artist_norm) & 
        (meta_features['Song_Norm'] == song_norm)
    ]
    
    if not matching_meta.empty:
        match_count += 1
        meta_row = matching_meta.iloc[0]
        new_row = row.copy()
        new_row['Instrumentation'] = meta_row['Instrumentation']
        new_row['Genre'] = meta_row['Genre']
        new_row['Acoustic vs Electronic'] = meta_row['Acoustic vs Electronic']
        new_row['Gender Voice'] = meta_row['Gender Voice']
        new_row['Bpm'] = meta_row['Bpm']
        new_row['Artist_Original'] = meta_row['Artist']
        new_row['Song_Original'] = meta_row['Song']
        new_row['Match_Type'] = 'Exact'
        result_rows.append(new_row)
    else:
        # Buscar coincidencia aproximada
        best_match = find_best_match(artist_norm, song_norm, meta_features)
        
        if best_match is not None:
            fuzzy_match_count += 1
            new_row = row.copy()
            new_row['Instrumentation'] = best_match['Instrumentation']
            new_row['Genre'] = best_match['Genre']
            new_row['Acoustic vs Electronic'] = best_match['Acoustic vs Electronic']
            new_row['Gender Voice'] = best_match['Gender Voice']
            new_row['Bpm'] = best_match['Bpm']
            new_row['Artist_Original'] = best_match['Artist']
            new_row['Song_Original'] = best_match['Song']
            new_row['Match_Type'] = 'Fuzzy'
            result_rows.append(new_row)
            print(f"Coincidencia aproximada: {artist} - {song} → {best_match['Artist']} - {best_match['Song']}")
        else:
            print(f"No se encontró coincidencia para: {artist} - {song}")
            # Intentar identificar artistas similares
            similar_artists = meta_features[meta_features['Artist_Norm'].str.contains(artist_norm[:3], regex=False)]
            if not similar_artists.empty:
                unique_artists = similar_artists['Artist'].unique()
                print(f"  Artistas similares encontrados: {', '.join(unique_artists)}")

# Crear nuevo DataFrame con las filas que tienen metadatos
df_with_meta = pd.DataFrame(result_rows)

# Si no hay filas con metadatos, usar el DataFrame original
if len(result_rows) == 0:
    print("⚠️ No se encontraron coincidencias entre embeddings y metadatos. Usando DataFrame sin metadatos.")
    df_with_meta = df.copy()
else:
    print(f"✅ Se encontraron {match_count} coincidencias exactas y {fuzzy_match_count} coincidencias aproximadas entre embeddings y metadatos (total: {len(result_rows)}).")

# --- STEP 7: Preprocesar las características para mejor visualización ---
# Convertir Instrumentation de valores numéricos a categorías
def categorize_instrumentation(value):
    value = int(value) if pd.notna(value) and value != '' else 0
    if value == 2:
        return "Instruments (2)"
    elif value == 3:
        return "Instruments (3)"
    elif value == 4:
        return "Instruments (4)"
    elif value == 5:
        return "Instruments (5)"
    else:
        return "Unknown"

# Aplicar la transformación
if 'Instrumentation' in df_with_meta.columns:
    df_with_meta['Instrumentation_Category'] = df_with_meta['Instrumentation'].apply(categorize_instrumentation)

# Definir colores para Population
population_colors = {
    "Before 2012": "#1f77b4",  # Azul
    "After 2018": "#ff7f0e"    # Naranja
}

# Simplificar nombres de género para evitar leyendas muy largas
if 'Genre' in df_with_meta.columns:
    # Función para extraer el género principal
    def simplify_genre(genre):
        if pd.isna(genre) or genre == '':
            return 'Unknown'
        # Tomar el primer género si hay varios separados por /
        main_genre = genre.split('/')[0].strip()
        return main_genre
    
    df_with_meta['Genre_Main'] = df_with_meta['Genre'].apply(simplify_genre)
    
    # Obtener todos los géneros únicos para usarlos en la leyenda
    genre_colors = {
        "Pop": "#1f77b4",         # Azul
        "Folk": "#ff7f0e",        # Naranja
        "Urbà": "#2ca02c",        # Verde
        "Reggaeton": "#d62728",   # Rojo
        "Trap": "#9467bd",        # Púrpura
        "Unknown": "#7f7f7f"      # Gris
    }
    
    # Añadir colores para cualquier otro género que aparezca
    unique_genres = df_with_meta['Genre_Main'].unique()
    other_colors = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]  # Colores adicionales
    i = 0
    for genre in unique_genres:
        if genre not in genre_colors and pd.notna(genre):
            genre_colors[genre] = other_colors[i % len(other_colors)]
            i += 1

# --- STEP 8: Apply t-SNE ---
embedding_cols = [col for col in df_with_meta.columns if col.startswith("e")]
X = df_with_meta[embedding_cols].values
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)
df_with_meta["x"] = X_2d[:, 0]
df_with_meta["y"] = X_2d[:, 1]

# --- STEP 9: Static plot (unchanged) ---
plt.figure(figsize=(10, 7))

# Crear un mapa de colores personalizado para matplotlib
population_palette = {pop: color for pop, color in population_colors.items()}

# Usar el mapa de colores personalizado en el gráfico
sns.scatterplot(
    data=df_with_meta, 
    x="x", y="y", 
    hue="Population", 
    style="Artist", 
    palette=population_palette
)

plt.title("t-SNE of Essentia Audio Embeddings (Colored by Population)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_dir = "visualization_results/embedding_visualization_effnet_by_feature"
os.makedirs(output_dir, exist_ok=True)
png_path = os.path.join(output_dir, "tsne_by_population.png")
plt.savefig(png_path, dpi=300)
plt.close()

# --- STEP 10: Create Interactive Plots with Different Colorings ---
artist_symbol_map = {
    "antonia_font": "circle",
    "els_catarres": "x",
    "macedonia": "square",
    "manel": "cross",
    "marina_rossell": "diamond",
    "txarango": "triangle-up",
    "31_fam": "triangle-down",
    "julieta": "triangle-left",
    "la_ludwig_band": "triangle-right",
    "mushka": "star",
    "mushkaa": "star",  # Añadimos ambas variantes
    "oques_grasses": "hexagon",
    "the_tyets": "pentagon"
}

df_with_meta["Label"] = df_with_meta.apply(
    lambda row: f"{row.get('Artist_Original', row['Artist'])} - {row.get('Song_Original', row['Song'])}", 
    axis=1
)

# Function to create and save interactive plot
def create_interactive_plot(color_by, title_suffix=None, show_legend=True, custom_labels=None):
    if title_suffix is None:
        title_suffix = color_by
        
    # Determinar qué columna usar para cada característica
    column_to_use = color_by
    if color_by == "Instrumentation" and "Instrumentation_Category" in df_with_meta.columns:
        column_to_use = "Instrumentation_Category"
    elif color_by == "Genre" and "Genre_Main" in df_with_meta.columns:
        column_to_use = "Genre_Main"
    
    # Configurar paleta de colores personalizada para características categóricas
    if color_by == "Population":
        color_discrete_map = population_colors
    elif color_by == "Gender Voice":
        color_discrete_map = {
            "Male": "#2271B2",
            "Female": "#D55E00",
            "Male & Female": "#009E73"
        }
    elif color_by == "Acoustic vs Electronic":
        color_discrete_map = {
            "Acoustic": "#009E73",
            "Electronic": "#D55E00",
            "Acoustic & Electronic": "#CC79A7"
        }
    elif color_by == "Instrumentation" or color_by == "Instrumentation_Category":
        color_discrete_map = {
            "Instruments (2)": "#E69F00",
            "Instruments (3)": "#56B4E9",
            "Instruments (4)": "#009E73",
            "Instruments (5)": "#F0E442",
            "Unknown": "#999999"
        }
    elif color_by == "Genre":
        # Usar los colores definidos para géneros
        color_discrete_map = genre_colors if 'genre_colors' in globals() else None
    else:
        color_discrete_map = None
    
    # Para BPM, que es numérico, usar una paleta continua
    if color_by == "Bpm":
        color_continuous_scale = "Viridis"
    else:
        color_continuous_scale = None
    
    # Crear la figura con configuración explícita para la leyenda
    fig = px.scatter(
        df_with_meta,
        x="x", y="y",
        color=column_to_use,
        symbol="Artist",  # Mantener símbolos para todos los gráficos
        symbol_map=artist_symbol_map,
        hover_name="Label",
        color_discrete_map=color_discrete_map,
        color_continuous_scale=color_continuous_scale,
        title=f"Interactive t-SNE of Essentia Audio Embeddings (Colored by {title_suffix})",
        labels={
            "x": "t-SNE Dimension 1", 
            "y": "t-SNE Dimension 2", 
            column_to_use: custom_labels or title_suffix
        },
        width=1200,  # Aumentar el ancho para dejar espacio para la leyenda
        height=750
    )
    
    # Añadir información sobre el tipo de coincidencia (exacta o aproximada)
    if "Match_Type" in df_with_meta.columns:
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>Match: %{customdata[0]}<extra></extra>',
            customdata=df_with_meta[['Match_Type']]
        )
    
    # Ajustar la configuración de la leyenda según el tipo de visualización
    if color_by == "Bpm":
        # Para BPM, configurar solo la barra de color y ocultar la leyenda de símbolos
        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title=dict(text="BPM", font=dict(size=16)),
                    thickness=20,
                    len=0.7,
                    x=1.02,
                    y=0.5
                )
            ),
            margin=dict(r=150)  # Aumentar el margen derecho para la barra de color
        )
        
        # Ocultar solo la leyenda de símbolos de artistas, pero mantener las formas en el gráfico
        for trace in fig.data:
            if hasattr(trace, 'showlegend'):
                trace.showlegend = False
                
    else:
        # Para otras visualizaciones, configurar la leyenda normal
        fig.update_layout(
            showlegend=True,  # Forzar la visibilidad de la leyenda
            legend=dict(
                title=dict(text=custom_labels or title_suffix, font=dict(size=16)),
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=2,
                itemsizing='constant',
                itemwidth=30,
                # Posicionar la leyenda fuera del gráfico a la derecha
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,  # Posición a la derecha del gráfico
                orientation="v"
            ),
            # Ajustar los márgenes para asegurarse de que hay espacio para la leyenda
            margin=dict(r=150)  # Aumentar el margen derecho
        )
        
        # Si estamos visualizando una característica categórica, suprimimos todos 
        # los símbolos de artistas de la leyenda
        for trace in fig.data:
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'symbol'):
                trace.showlegend = False
        
        # Creamos traces específicos solo para la leyenda con cada categoría
        if color_by == "Population":
            # Para Population, mostrar explícitamente Before 2012 y After 2018
            for population, color in population_colors.items():
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],  # Sin datos, solo para leyenda
                        mode='markers',
                        marker=dict(size=10, color=color),
                        name=str(population),
                        showlegend=True,
                        legendgroup=str(population)
                    )
                )
        elif color_by == "Genre" and 'genre_colors' in globals():
            # Para géneros, usar los colores personalizados
            for genre, color in genre_colors.items():
                if genre in df_with_meta[column_to_use].values:
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None],  # Sin datos, solo para leyenda
                            mode='markers',
                            marker=dict(size=10, color=color),
                            name=str(genre),
                            showlegend=True,
                            legendgroup=str(genre)
                        )
                    )
        elif color_discrete_map:
            # Para otras categorías, crear un elemento de leyenda para cada una
            categories = sorted([cat for cat in df_with_meta[column_to_use].unique() if pd.notna(cat)])
            for category in categories:
                if str(category) in color_discrete_map:
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None],  # Sin datos, solo para leyenda
                            mode='markers',
                            marker=dict(size=10, color=color_discrete_map[str(category)]),
                            name=str(category),
                            showlegend=True,
                            legendgroup=str(category)
                        )
                    )
    
    # Guardar el HTML interactivo
    html_path = os.path.join(output_dir, f"tsne_by_{color_by.lower().replace(' ', '_')}.html")
    fig.write_html(html_path)
    print(f"✅ Interactive Plot colored by {color_by} saved to: {html_path}")
    
    # Configuración para PNG: aumentar el tamaño para mejor calidad y asegurar espacio para leyendas
    fig_png = fig.update_layout(
        width=1600,         # Ancho mayor para la imagen
        height=1000,        # Alto mayor para la imagen
        margin=dict(r=200), # Margen derecho ampliado para asegurar que la leyenda sea visible
        font=dict(size=16), # Fuente más grande para mejor legibilidad en el PNG
    )
    
    # Generate PNG capture con alta resolución
    png_path = os.path.join(output_dir, f"tsne_by_{color_by.lower().replace(' ', '_')}.png")
    write_image(fig_png, png_path, scale=2)  # Escala 2x para mayor resolución
    print(f"✅ PNG capture saved to: {png_path}")
    
    return html_path

# Create plots for each feature
features = [
    {"name": "Population", "show_legend": True, "label": "Population"},
    {"name": "Instrumentation", "show_legend": True, "label": "Instrumentation Level"},
    {"name": "Genre", "show_legend": True, "label": "Music Genre"},
    {"name": "Acoustic vs Electronic", "show_legend": True, "label": "Production Type"},
    {"name": "Gender Voice", "show_legend": True, "label": "Gender Voice"},
    {"name": "Bpm", "show_legend": True, "label": "Beats per minute (BPM)"}
]

html_paths = []

for feature in features:
    if feature["name"] in df_with_meta.columns or (
        feature["name"] == "Instrumentation" and "Instrumentation_Category" in df_with_meta.columns
    ) or (
        feature["name"] == "Genre" and "Genre_Main" in df_with_meta.columns
    ):
        html_paths.append(create_interactive_plot(
            feature["name"],
            show_legend=feature["show_legend"],
            custom_labels=feature["label"]
        ))
    else:
        print(f"⚠️ Feature '{feature['name']}' no está disponible en los datos")

# Open the population plot (as in the original script)
if html_paths:
    html_full_path = os.path.abspath(html_paths[0])  # Population plot
    if html_full_path.startswith("/mnt/"):
        drive_letter = html_full_path[5]
        windows_path = html_full_path.replace(f"/mnt/{drive_letter}/", f"{drive_letter.upper()}:\\").replace("/", "\\")
        subprocess.run(["powershell.exe", "Start-Process", windows_path])
    else:
        print("⚠️ Could not convert path to Windows. Please open the HTML file manually.")

    # Print summary of all plots created
    print("\n📊 Resumen de visualizaciones creadas:")
    for feature, path in zip([f["name"] for f in features if f["name"] in df_with_meta.columns], html_paths):
        print(f"- {feature}: {os.path.basename(path)}")
else:
    print("⚠️ No se crearon visualizaciones")
