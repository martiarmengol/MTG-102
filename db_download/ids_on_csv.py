#!/usr/bin/env python3
"""
Script to update the 'ID' column in a CSV file by extracting the YouTube video ID from the 'YT Link' column.
Usage:
    python ids_on_csv.py <path_to_csv>
"""
import csv
import sys
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().replace("www.", "")
    if "youtu.be" in netloc:
        return parsed.path.lstrip("/")
    if "youtube.com" in netloc:
        qs = parse_qs(parsed.query)
        return qs.get("v", [None])[0]
    return None

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_file>", file=sys.stderr)
        sys.exit(1)

    csv_file = sys.argv[1]

    # Read all rows
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Error: CSV header not found.", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)

    # Update the 'ID' column
    for row in rows:
        url = row.get("YT Link", "")
        vid_id = extract_video_id(url)
        if vid_id:
            row["ID"] = vid_id
        else:
            print(f"Warning: could not extract video ID from URL '{url}'", file=sys.stderr)

    # Write back to the same file
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Processed {len(rows)} rows; overwritten '{csv_file}' with updated IDs.")

if __name__ == "__main__":
    main()
