#!/usr/bin/env python3
"""
Script to download audio from YouTube links listed in a CSV file.
Usage:
    python download_from_csv.py <csv_file> <output_dir>
CSV must have at least 9 columns, with YouTube URL in column 9.
Filenames will be constructed as col1-col2-col3.mp3
"""

import os
import sys
import csv
import argparse
import re
from urllib.parse import urlparse, parse_qs
import yt_dlp

def extract_video_id(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().replace('www.', '')
    if 'youtu.be' in netloc:
        return parsed.path.lstrip('/')
    if 'youtube.com' in netloc:
        qs = parse_qs(parsed.query)
        return qs.get('v', [None])[0]
    return None

def sanitize_filename(name):
    # Remove illegal filesystem characters and collapse whitespace to underscores
    sanitized = re.sub(r'[\\/:"*?<>|]+', '', name)
    sanitized = sanitized.strip()
    sanitized = '_'.join(sanitized.split())
    return sanitized

def download_audio(video_id, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, f'{filename}.%(ext)s'),
        'quiet': True,
        'ignoreerrors': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        return True
    except Exception as e:
        print(f"Error downloading {video_id}: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Download audio from YouTube URLs in a CSV.'
    )
    parser.add_argument(
        'csv_file',
        help='Path to input CSV file'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to save downloaded audio'
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv_file):
        print(f"Error: CSV file not found at '{args.csv_file}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            print('Error: CSV file is empty.', file=sys.stderr)
            sys.exit(1)
        for idx, row in enumerate(reader, start=2):
            if len(row) < 9:
                print(f"Skipping row {idx}: expected ≥9 columns, found {len(row)}", file=sys.stderr)
                continue
            col1, col2, col3, url = row[0], row[1], row[2], row[8]
            video_id = extract_video_id(url)
            if not video_id:
                print(f"Skipping row {idx}: cannot extract video ID from URL '{url}'", file=sys.stderr)
                continue
            raw_name = f"{col1}-{col2}-{col3}"
            filename = sanitize_filename(raw_name)
            print(f"Row {idx}: downloading video {video_id} as '{filename}.mp3'")
            success = download_audio(video_id, filename, args.output_dir)
            if not success:
                print(f"Failed to download video {video_id}", file=sys.stderr)

if __name__ == '__main__':
    main()
