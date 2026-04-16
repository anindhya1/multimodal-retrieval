"""
Whisper Transcription
Runs OpenAI Whisper on vocals.wav for each trailer and stores
the full transcription text + segments in trailers_metadata.json.
"""

import json
import os
import ssl
import certifi
from pathlib import Path

os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

META_FILE = Path(__file__).parent / "trailer_data" / "trailers_metadata.json"

def log(msg: str):
    print(msg, flush=True)


def main():
    with open(META_FILE) as f:
        metadata = json.load(f)

    # Load Whisper model once — "base" is fast on CPU and accurate enough
    log("Loading Whisper model (base)...")
    import whisper
    model = whisper.load_model("base")
    log("  Model ready.\n")

    processed = 0

    for i, entry in enumerate(metadata, 1):
        title = entry.get("title", "?")

        if "error" in entry:
            log(f"[{i}/120] {title} — skipped (no trailer)")
            continue

        # Skip if already transcribed
        if entry.get("whisper_transcript"):
            log(f"[{i}/120] {title} — already transcribed, skipping")
            continue

        vocals_path = entry.get("local_paths", {}).get("vocals") or ""
        vocals_path = Path(vocals_path)
        if not vocals_path.exists():
            log(f"[{i}/120] {title} — vocals.wav not found, skipping")
            continue

        log(f"[{i}/120] {title}")
        log(f"    Transcribing {vocals_path.name}...")

        try:
            result = model.transcribe(
                str(vocals_path),
                language="en",
                fp16=False,        # CPU — fp16 not supported
                verbose=False,
            )

            transcript = result["text"].strip()
            segments   = [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result["segments"]
            ]

            entry["whisper_transcript"] = transcript
            entry["whisper_segments"]   = segments
            log(f"    \"{transcript[:120]}{'...' if len(transcript) > 120 else ''}\"")

        except Exception as e:
            log(f"    FAILED: {e}")
            entry["whisper_transcript"] = ""
            entry["whisper_segments"]   = []

        processed += 1

        # Save progress after each trailer
        with open(META_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

    log(f"\n{'─'*55}")
    log(f"Done.  {processed} trailers transcribed.")
    log(f"Metadata updated: {META_FILE}")
    log(f"{'─'*55}")


if __name__ == "__main__":
    main()
