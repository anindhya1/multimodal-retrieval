"""
Title Frame Extractor
For each trailer, extracts a frame every 1 second from the last 20 seconds,
runs OCR on all frames, picks the one with the most text as the title card,
and saves it as title_frame.jpg alongside the OCR text in metadata.
"""

import json
import os
import re
import subprocess
import shutil
import ssl
import certifi
from pathlib import Path

os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# ── Config ────────────────────────────────────────────────────────────────────
META_FILE   = Path(__file__).parent / "trailer_data" / "trailers_metadata.json"
CLIPS_DIR   = Path(__file__).parent / "trailer_data" / "clips"
WINDOW_SECS = 20   # scan last N seconds of each trailer
FRAME_STEP  = 1    # extract one frame every N seconds in the window
# ─────────────────────────────────────────────────────────────────────────────


def log(msg: str):
    print(msg, flush=True)


def get_duration(clip_path: Path) -> float:
    result = subprocess.run(
        ["ffmpeg", "-i", str(clip_path)],
        capture_output=True, text=True
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def extract_frames(clip_path: Path, tmp_dir: Path, duration: float) -> list[Path]:
    """Extract one frame per second from the last WINDOW_SECS seconds."""
    tmp_dir.mkdir(exist_ok=True)
    start = max(0.0, duration - WINDOW_SECS)
    frames = []

    t = start
    i = 0
    while t < duration:
        out = tmp_dir / f"scan_{i:03d}.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t:.3f}",
            "-i", str(clip_path),
            "-frames:v", "1", "-q:v", "2", str(out),
        ], capture_output=True)
        if out.exists():
            frames.append(out)
        t += FRAME_STEP
        i += 1

    return frames


def run_ocr(reader, frames: list[Path]) -> tuple[Path | None, str]:
    """
    Run OCR on each frame. Return the frame with the most detected text
    and its combined OCR string.
    """
    best_frame = None
    best_text  = ""
    best_score = 0

    for frame in frames:
        try:
            results = reader.readtext(str(frame), detail=1)
        except Exception as e:
            log(f"    OCR error on {frame.name}: {e}")
            continue

        # Score = sum of (confidence * text length) for all detections
        score = sum(conf * len(text) for (_, text, conf) in results)
        text  = " ".join(t for (_, t, _) in results).strip()

        if score > best_score:
            best_score = score
            best_frame = frame
            best_text  = text

    return best_frame, best_text


def main():
    # Load metadata
    with open(META_FILE) as f:
        metadata = json.load(f)

    # Init EasyOCR once (downloads model on first run)
    log("Loading EasyOCR model (English)...")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    log("  Model ready.\n")

    updated = 0
    for i, entry in enumerate(metadata, 1):
        title = entry.get("title", "?")

        if "error" in entry:
            log(f"[{i}/120] {title} — skipped (no trailer)")
            continue

        clip_path = Path(entry.get("local_paths", {}).get("clip", ""))
        if not clip_path.exists():
            log(f"[{i}/120] {title} — clip not found, skipping")
            continue

        # Skip if already done
        if entry.get("title_ocr") and (clip_path.parent / "title_frame.jpg").exists():
            log(f"[{i}/120] {title} — already done, skipping")
            continue

        log(f"[{i}/120] {title}")

        duration = get_duration(clip_path)
        if duration < 1.0:
            log(f"    Could not determine duration, skipping")
            continue
        log(f"    Duration: {duration:.1f}s  |  Scanning last {WINDOW_SECS}s")

        tmp_dir = clip_path.parent / "title_scan_tmp"
        frames  = extract_frames(clip_path, tmp_dir, duration)
        log(f"    Extracted {len(frames)} frames for OCR")

        if not frames:
            log(f"    No frames extracted, skipping")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        best_frame, best_text = run_ocr(reader, frames)

        if best_frame:
            dst = clip_path.parent / "title_frame.jpg"
            shutil.copy(best_frame, dst)
            entry["title_frame"]  = str(dst)
            entry["title_ocr"]    = best_text
            log(f"    Best frame: {best_frame.name}")
            log(f"    OCR text  : \"{best_text[:120]}\"")
        else:
            log(f"    No text detected in any frame")
            entry["title_ocr"] = ""

        shutil.rmtree(tmp_dir, ignore_errors=True)
        updated += 1

        # Save progress after each trailer
        with open(META_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

    log(f"\n{'─'*55}")
    log(f"Done.  {updated} trailers processed.")
    log(f"Metadata updated: {META_FILE}")
    log(f"{'─'*55}")


if __name__ == "__main__":
    main()
