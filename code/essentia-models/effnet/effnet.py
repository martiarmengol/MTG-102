import os
import glob
import pickle
import argparse
import datetime
from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs
)


def compute_effnet_embeddings_for_folder(
    folder: str, output_folder: str = None
) -> None:
    """
    Compute embeddings for every .mp3 in `folder` (including subdirectories) and write to pickle.
    Output filename is current date (YYYY-MM-DD) with format `<date>_effnet.pkl`.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = "/Users/javierechavarri/Desktop/MTG-102/code/essentia_models/discogs_multi_embeddings-effnet-bs64-1.pb"
    if output_folder is None:
        output_folder = base_dir
    os.makedirs(output_folder, exist_ok=True)

    # Generate output filename based on current date
    date_str = datetime.date.today().isoformat()
    output_filename = f"{date_str}_effnet.pkl"
    output_path = os.path.join(output_folder, output_filename)

    # Initialize the embedding model
    model = TensorflowPredictEffnetDiscogs(
        graphFilename=model_path, output="PartitionedCall:1"
    )

    entries = []
    # Recursively process each MP3 file in sorted order
    for file_path in sorted(glob.glob(os.path.join(folder, "**", "*.mp3"), recursive=True)):
        filename = os.path.basename(file_path)
        # Split into artist and song (remove extension)
        name_no_ext = os.path.splitext(filename)[0]
        if "-" in name_no_ext and not name_no_ext.startswith("-"):
            artist, song = name_no_ext.split("-", 1)
        else:
            artist = "unknown"
            song = name_no_ext.lstrip("-")

        # Load audio and compute embedding
        audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
        embedding = model(audio)
        # Convert embedding to a list of floats
        try:
            embedding_list = embedding.tolist()
        except AttributeError:
            embedding_list = list(embedding)

        entries.append({"artist": artist, "song": song, "embedding": embedding_list})

    # Write all entries to pickle
    with open(output_path, "wb") as f:
        pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Processed {len(entries)} files, saved embeddings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute EffNet embeddings for all mp3s in a folder"
    )
    parser.add_argument("folder", help="Path to folder containing .mp3 files")
    parser.add_argument(
        "-o", "--output_folder",
        help="Directory to save pickle output",
        default=None
    )
    args = parser.parse_args()
    compute_effnet_embeddings_for_folder(args.folder, args.output_folder)
