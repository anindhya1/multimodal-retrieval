"""
Reorganizes each trailer folder into image/, video/, audio/ subfolders
and updates local_paths in trailers_metadata.json accordingly.
"""

import json
import shutil
from pathlib import Path

META_FILE = Path(__file__).parent / "trailer_data" / "trailers_metadata.json"

FILE_MAP = {
    "video": ["clip.mp4"],
    "audio": ["audio.wav", "vocals.wav", "no_vocals.wav", "audio_trim.wav"],
    "image": ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg", "title_frame.jpg"],
}

# Reverse lookup: filename -> subfolder
DEST = {f: sub for sub, files in FILE_MAP.items() for f in files}

with open(META_FILE) as f:
    metadata = json.load(f)

for entry in metadata:
    if "error" in entry or not entry.get("local_paths"):
        continue

    clip_dir = Path(entry.get("local_dir", ""))
    if not clip_dir.exists():
        continue

    # Create subfolders
    for sub in FILE_MAP:
        (clip_dir / sub).mkdir(exist_ok=True)

    # Move files
    for file in clip_dir.iterdir():
        if file.is_file() and file.name in DEST:
            dest = clip_dir / DEST[file.name] / file.name
            shutil.move(str(file), str(dest))

    # Update local_paths in metadata
    path_key_map = {
        "clip":      ("video", "clip.mp4"),
        "audio":     ("audio", "audio.wav"),
        "vocals":    ("audio", "vocals.wav"),
        "no_vocals": ("audio", "no_vocals.wav"),
        "frame_0":   ("image", "frame_0.jpg"),
        "frame_1":   ("image", "frame_1.jpg"),
        "frame_2":   ("image", "frame_2.jpg"),
    }
    for key, (sub, fname) in path_key_map.items():
        new_path = clip_dir / sub / fname
        if new_path.exists():
            entry["local_paths"][key] = str(new_path)

    # Update title_frame and audio_trim paths
    for fname, meta_key in [("title_frame.jpg", "title_frame"), ("audio_trim.wav", "audio_trim")]:
        sub  = DEST[fname]
        path = clip_dir / sub / fname
        if path.exists():
            entry[meta_key] = str(path)

    print(f"  Reorganized: {clip_dir.name}")

with open(META_FILE, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone. Metadata updated: {META_FILE}")
