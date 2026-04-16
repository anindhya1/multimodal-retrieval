"""
MMTrail Dataset Downloader
Downloads 120 random samples (metadata + video clips) from MMTrail-2M.

Per sample output:
  clips/{clip_id}/
    clip.mp4        — video clip
    audio.wav       — mixed audio
    vocals.wav      — isolated speech/vocals (demucs)
    no_vocals.wav   — isolated instrumental (demucs)
    frame_0.jpg     — keyframe 1  (aligns with frame_caption[0])
    frame_1.jpg     — keyframe 2  (aligns with frame_caption[1])
    frame_2.jpg     — keyframe 3  (aligns with frame_caption[2])
"""

import json
import os
import random
import subprocess
import shutil
import ssl
import certifi
from pathlib import Path

# Fix macOS SSL cert issue for all urllib-based calls (including demucs model download)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# ── Config ────────────────────────────────────────────────────────────────────
JSON_URL   = "https://raw.githubusercontent.com/litwellchi/MMTrail/main/MMTrail2M_sample1w.json"
OUTPUT_DIR = Path(__file__).parent / "mmtrail_data"
META_FILE  = OUTPUT_DIR / "mmtrail_120_samples.json"
CLIPS_DIR  = OUTPUT_DIR / "clips"
N_SAMPLES  = 120
SEED       = 42
DEMUCS_MODEL = "htdemucs"   # good quality / reasonable CPU speed
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str):
    print(msg, flush=True)


def run(cmd: list, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def clip_timestamps(entry: dict) -> tuple[float, float]:
    fps = entry.get("video_fps", 25.0)
    start_frame, end_frame = entry["clip_start_end_idx"]
    return start_frame / fps, end_frame / fps


# ── Step 1: Metadata ─────────────────────────────────────────────────────────

def fetch_and_sample() -> list:
    import requests
    log("Fetching metadata JSON from GitHub...")
    r = requests.get(JSON_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    log(f"  Loaded {len(data):,} samples.")

    rng     = random.Random(SEED)
    samples = rng.sample(data, N_SAMPLES)
    log(f"  Sampled {N_SAMPLES} entries (seed={SEED}).")
    return samples


# ── Step 2: Video clip ────────────────────────────────────────────────────────

def download_clip(entry: dict, clip_dir: Path, idx: int, total: int) -> bool:
    vid_id   = entry["video_id"]
    clip_id  = entry["clip_id"]
    out_path = clip_dir / "clip.mp4"

    if out_path.exists() and out_path.stat().st_size > 10_000:
        log(f"  [{idx}/{total}] clip already exists — skipping download")
        return True

    start_s, end_s = clip_timestamps(entry)
    yt_url = f"https://www.youtube.com/watch?v={vid_id}"
    log(f"  [{idx}/{total}] Downloading  {clip_id}  t={start_s:.1f}s–{end_s:.1f}s")

    tmp = clip_dir / "clip_tmp.mp4"
    result = run([
        "yt-dlp",
        "--quiet", "--no-warnings",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--download-sections", f"*{start_s:.3f}-{end_s:.3f}",
        "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4",
        "--output", str(tmp),
        yt_url,
    ], timeout=180)

    if result.returncode != 0 or not tmp.exists():
        log(f"    FAILED (yt-dlp): {result.stderr.strip()[:150]}")
        return False

    tmp.rename(out_path)
    log(f"    OK  {out_path.stat().st_size / 1e6:.1f} MB")
    return True


# ── Step 3: Extract audio ────────────────────────────────────────────────────

def extract_audio(clip_dir: Path) -> bool:
    src = clip_dir / "clip.mp4"
    dst = clip_dir / "audio.wav"

    if dst.exists():
        return True

    result = run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vn",                      # drop video stream
        "-ar", "44100",             # sample rate
        "-ac", "2",                 # stereo
        "-f", "wav", str(dst),
    ])

    if result.returncode != 0:
        log(f"    FAILED (audio extract): {result.stderr.strip()[:150]}")
        return False

    log(f"    audio.wav extracted")
    return True


# ── Step 4: Source separation (demucs, CPU) ──────────────────────────────────

def separate_audio(clip_dir: Path) -> bool:
    vocals_dst    = clip_dir / "vocals.wav"
    no_vocals_dst = clip_dir / "no_vocals.wav"

    if vocals_dst.exists() and no_vocals_dst.exists():
        return True

    src = clip_dir / "audio.wav"
    log(f"    Running demucs (CPU) — this may take a minute...")

    # demucs outputs to a temp folder; we'll move the files after
    tmp_out = clip_dir / "demucs_tmp"
    result = run([
        "python3", "-m", "demucs",
        "--two-stems", "vocals",        # split into vocals + no_vocals only
        "--device", "cpu",
        "-n", DEMUCS_MODEL,
        "--out", str(tmp_out),
        str(src),
    ], timeout=600)   # 10-min ceiling per clip on CPU

    if result.returncode != 0:
        log(f"    FAILED (demucs): {result.stderr.strip()[:200]}")
        if tmp_out.exists():
            shutil.rmtree(tmp_out, ignore_errors=True)
        return False

    # demucs writes to: tmp_out/{model}/{input_stem}/vocals.wav + no_vocals.wav
    try:
        stems_dir = next(tmp_out.glob(f"{DEMUCS_MODEL}/*/"))
        (stems_dir / "vocals.wav").rename(vocals_dst)
        (stems_dir / "no_vocals.wav").rename(no_vocals_dst)
        shutil.rmtree(tmp_out, ignore_errors=True)
        log(f"    vocals.wav + no_vocals.wav extracted")
        return True
    except (StopIteration, FileNotFoundError) as e:
        log(f"    FAILED moving demucs outputs: {e}")
        shutil.rmtree(tmp_out, ignore_errors=True)
        return False


# ── Step 5: Keyframes ────────────────────────────────────────────────────────

def extract_keyframes(entry: dict, clip_dir: Path) -> bool:
    src               = clip_dir / "clip.mp4"
    start_s, end_s    = clip_timestamps(entry)
    duration          = end_s - start_s

    # evenly-spaced timestamps at 1/6, 3/6, 5/6 of the clip → 3 frames
    offsets = [duration * f for f in (1/6, 3/6, 5/6)]
    all_ok  = True

    for i, offset in enumerate(offsets):
        dst = clip_dir / f"frame_{i}.jpg"
        if dst.exists():
            continue

        result = run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{offset:.3f}",
            "-i", str(src),
            "-frames:v", "1",
            "-q:v", "2",            # high JPEG quality
            str(dst),
        ])

        if result.returncode != 0:
            log(f"    FAILED (frame {i}): {result.stderr.strip()[:100]}")
            all_ok = False

    if all_ok:
        log(f"    frame_0/1/2.jpg extracted")
    return all_ok


# ── Main ─────────────────────────────────────────────────────────────────────

def process_entry(entry: dict, clip_dir: Path, i: int, total: int, stats: dict):
    """Run all steps for a single entry. Returns True if video succeeded."""
    clip_dir.mkdir(exist_ok=True)

    ok_video = download_clip(entry, clip_dir, i, total)
    if not ok_video:
        stats["failed"].append(entry["clip_id"])
        entry["local_paths"] = {"error": "download failed"}
        return False
    stats["video"] += 1

    ok_audio = extract_audio(clip_dir)
    if ok_audio:
        stats["audio"] += 1

    ok_sep = False
    if ok_audio:
        ok_sep = separate_audio(clip_dir)
        if ok_sep:
            stats["separation"] += 1

    ok_frames = extract_keyframes(entry, clip_dir)
    if ok_frames:
        stats["frames"] += 1

    entry["local_paths"] = {
        "clip":      str(clip_dir / "clip.mp4")      if ok_video  else None,
        "audio":     str(clip_dir / "audio.wav")     if ok_audio  else None,
        "vocals":    str(clip_dir / "vocals.wav")    if ok_sep    else None,
        "no_vocals": str(clip_dir / "no_vocals.wav") if ok_sep    else None,
        "frame_0":   str(clip_dir / "frame_0.jpg")   if ok_frames else None,
        "frame_1":   str(clip_dir / "frame_1.jpg")   if ok_frames else None,
        "frame_2":   str(clip_dir / "frame_2.jpg")   if ok_frames else None,
    }
    return True


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CLIPS_DIR.mkdir(exist_ok=True)

    # 1. Fetch full pool + carve out initial 120 sample
    import requests
    log("Fetching full metadata pool from GitHub...")
    r = requests.get(JSON_URL, timeout=60)
    r.raise_for_status()
    all_data = r.json()
    log(f"  Loaded {len(all_data):,} samples.")

    rng          = random.Random(SEED)
    sampled_ids  = set()
    samples      = []

    # Reload existing metadata if present (resume run)
    if META_FILE.exists():
        with open(META_FILE) as f:
            samples = json.load(f)
        sampled_ids = {e["clip_id"] for e in samples}
        log(f"  Resuming — {len(samples)} entries already in metadata.")
    else:
        initial = rng.sample(all_data, N_SAMPLES)
        samples = initial
        sampled_ids = {e["clip_id"] for e in samples}
        with open(META_FILE, "w") as f:
            json.dump(samples, f, indent=2)
        log(f"  Sampled {N_SAMPLES} entries (seed={SEED}).")

    # Build replacement pool (entries not already sampled)
    replacement_pool = [e for e in all_data if e["clip_id"] not in sampled_ids]
    rng.shuffle(replacement_pool)
    replacement_iter = iter(replacement_pool)

    log(f"  Metadata saved -> {META_FILE}\n")

    # 2. Process each sample; swap out failures with replacements
    stats = {"video": 0, "audio": 0, "separation": 0, "frames": 0, "failed": []}
    i = 0

    while i < len(samples):
        entry    = samples[i]
        clip_id  = entry["clip_id"]
        clip_dir = CLIPS_DIR / clip_id

        # Skip if already fully processed
        lp = entry.get("local_paths", {})
        if lp.get("clip") and Path(lp["clip"]).exists() and lp.get("vocals") and Path(lp["vocals"]).exists():
            log(f"\n[{i+1}/{N_SAMPLES}] {clip_id} — already complete, skipping")
            stats["video"] += 1; stats["audio"] += 1
            stats["separation"] += 1; stats["frames"] += 1
            i += 1
            continue

        log(f"\n[{i+1}/{N_SAMPLES}] {clip_id}")
        ok = process_entry(entry, clip_dir, i + 1, N_SAMPLES, stats)

        if not ok:
            # Swap in a replacement
            try:
                replacement = next(replacement_iter)
                log(f"  -> Replacing with {replacement['clip_id']}")
                samples[i] = replacement
                sampled_ids.add(replacement["clip_id"])
                # Don't increment i — retry this slot with the replacement
                continue
            except StopIteration:
                log(f"  -> No more replacements available")
                i += 1
        else:
            i += 1

        # Save progress after each clip
        with open(META_FILE, "w") as f:
            json.dump(samples, f, indent=2)

    # Save final metadata with local paths
    with open(META_FILE, "w") as f:
        json.dump(samples, f, indent=2)

    # Summary
    log(f"\n{'─'*55}")
    log(f"Complete.")
    log(f"  Videos downloaded   : {stats['video']}/{N_SAMPLES}")
    log(f"  Audio extracted     : {stats['audio']}/{N_SAMPLES}")
    log(f"  Audio separated     : {stats['separation']}/{N_SAMPLES}")
    log(f"  Keyframes extracted : {stats['frames']}/{N_SAMPLES}")
    if stats["failed"]:
        log(f"  Failed downloads    : {len(stats['failed'])} — {', '.join(stats['failed'][:5])}" +
            (" ..." if len(stats["failed"]) > 5 else ""))
    log(f"  Metadata            : {META_FILE}")
    log(f"  Clips directory     : {CLIPS_DIR}")
    log(f"{'─'*55}")


if __name__ == "__main__":
    main()
