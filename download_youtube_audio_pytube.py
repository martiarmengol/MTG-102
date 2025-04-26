import pandas as pd
from pytube import YouTube
from pydub import AudioSegment
import os

# Load CSV
csv_path = "Database - Before 2012.csv"
df = pd.read_csv(csv_path)

# Create output directory
output_dir = "downloaded_songs"
os.makedirs(output_dir, exist_ok=True)

# Iterate over each row (assuming the column with YouTube links is named 'youtube_link')
for idx, row in df.iterrows():
    try:
        url = row['YT Link']
        song_title = row.get('title', f'song_{idx}').replace("/", "_").replace("\\", "_")
        
        # Download audio
        url = url.split('&')[0]
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        temp_file = audio_stream.download(filename=f"{song_title}.mp4")

        # Convert to WAV
        audio = AudioSegment.from_file(temp_file)
        output_path = os.path.join(output_dir, f"{song_title}.wav")
        audio.export(output_path, format="wav")

        # Remove temp file
        os.remove(temp_file)

        print(f"Downloaded and converted: {song_title}")

    except Exception as e:
        print(f"Failed to process row {idx} ({url}): {e}")
