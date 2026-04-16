"""
Trailer Downloader
Searches YouTube for official trailers for 120 movies/TV shows,
downloads them, and runs the full multimodal pipeline:
  - clip.mp4        video
  - audio.wav       mixed audio
  - vocals.wav      isolated speech (demucs)
  - no_vocals.wav   isolated instrumental (demucs)
  - frame_0/1/2.jpg keyframes
"""

import json
import os
import ssl
import certifi
import subprocess
import shutil
from pathlib import Path

os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# ── Titles ────────────────────────────────────────────────────────────────────

MOVIES = [
    ("The Shawshank Redemption", 1994),
    ("The Godfather", 1972),
    ("The Dark Knight", 2008),
    ("The Godfather Part II", 1974),
    ("12 Angry Men", 1957),
    ("The Lord of the Rings: The Return of the King", 2003),
    ("Schindler's List", 1993),
    ("The Lord of the Rings: The Fellowship of the Ring", 2001),
    ("Pulp Fiction", 1994),
    ("The Good, the Bad and the Ugly", 1966),
    ("The Lord of the Rings: The Two Towers", 2002),
    ("Forrest Gump", 1994),
    ("Fight Club", 1999),
    ("Inception", 2010),
    ("Star Wars: The Empire Strikes Back", 1980),
    ("The Matrix", 1999),
    ("Goodfellas", 1990),
    ("Interstellar", 2014),
    ("One Flew Over the Cuckoo's Nest", 1975),
    ("Se7en", 1995),
    ("It's a Wonderful Life", 1946),
    ("The Silence of the Lambs", 1991),
    ("Seven Samurai", 1954),
    ("Saving Private Ryan", 1998),
    ("The Green Mile", 1999),
    ("City of God", 2002),
    ("Life Is Beautiful", 1997),
    ("Terminator 2: Judgment Day", 1991),
    ("Star Wars: A New Hope", 1977),
    ("Back to the Future", 1985),
    ("Spirited Away", 2001),
    ("The Pianist", 2002),
    ("Gladiator", 2000),
    ("Parasite", 2019),
    ("Grave of the Fireflies", 1988),
    ("Psycho", 1960),
    ("The Lion King", 1994),
    ("Harakiri", 1962),
    ("The Departed", 2006),
    ("Whiplash", 2014),
    ("Kill Bill: Vol. 1", 2003),
    ("The Prestige", 2006),
    ("American History X", 1998),
    ("Leon: The Professional", 1994),
    ("Spider-Man: Across the Spider-Verse", 2023),
    ("Cinema Paradiso", 1988),
    ("Casablanca", 1942),
    ("The Intouchables", 2011),
    ("The Usual Suspects", 1995),
    ("Alien", 1979),
    ("Django Unchained", 2012),
    ("Rear Window", 1954),
    ("Modern Times", 1936),
    ("City Lights", 1931),
    ("The Shining", 1980),
    ("Once Upon a Time in the West", 1968),
    ("Coco", 2017),
    ("Hamilton", 2020),
    ("Dr. Strangelove", 1964),
    ("Your Name", 2016),
]

TV_SHOWS = [
    ("Breaking Bad", 2008),
    ("Planet Earth II", 2016),
    ("Planet Earth", 2006),
    ("Band of Brothers", 2001),
    ("Chernobyl", 2019),
    ("The Wire", 2002),
    ("Avatar: The Last Airbender", 2005),
    ("The Sopranos", 1999),
    ("Blue Planet II", 2017),
    ("Cosmos: A Spacetime Odyssey", 2014),
    ("Cosmos", 1980),
    ("Our Planet", 2019),
    ("Game of Thrones", 2011),
    ("Bluey", 2018),
    ("The World at War", 1973),
    ("Fullmetal Alchemist: Brotherhood", 2009),
    ("Attack on Titan", 2013),
    ("Life", 2009),
    ("The Last Dance", 2020),
    ("The Twilight Zone", 1959),
    ("The Vietnam War", 2017),
    ("Rick and Morty", 2013),
    ("Sherlock", 2010),
    ("Batman: The Animated Series", 1992),
    ("Better Call Saul", 2015),
    ("Arcane", 2021),
    ("The Office", 2005),
    ("The Blue Planet", 2001),
    ("Clarkson's Farm", 2021),
    ("Scam 1992: The Harshad Mehta Story", 2020),
    ("Hunter x Hunter", 2011),
    ("Frozen Planet", 2011),
    ("Dexter: Resurrection", 2025),
    ("The Beatles Anthology", 1995),
    ("Only Fools and Horses", 1981),
    ("Human Planet", 2011),
    ("The Civil War", 1990),
    ("Firefly", 2002),
    ("As If", 2021),
    ("The Pitt", 2025),
    ("Gravity Falls", 2012),
    ("Death Note", 2006),
    ("Frieren: Beyond Journey's End", 2023),
    ("Taskmaster", 2015),
    ("Seinfeld", 1989),
    ("Dekalog", 1989),
    ("The Beatles: Get Back", 2021),
    ("Nathan for You", 2013),
    ("Cowboy Bebop", 1998),
    ("True Detective", 2014),
    ("Apocalypse: The Second World War", 2009),
    ("Fargo", 2014),
    ("Persona", 2018),
    ("It's Always Sunny in Philadelphia", 2005),
    ("Succession", 2018),
    ("Rome", 2005),
    ("Monty Python's Flying Circus", 1969),
    ("Friends", 1994),
    ("House", 2004),
    ("Only Murders in the Building", 2021),
]

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path(__file__).parent / "trailer_data"
META_FILE    = OUTPUT_DIR / "trailers_metadata.json"
CLIPS_DIR    = OUTPUT_DIR / "clips"
DEMUCS_MODEL = "htdemucs"
# ─────────────────────────────────────────────────────────────────────────────


def log(msg: str):
    print(msg, flush=True)


def run(cmd: list, timeout: int = 300) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="timed out")


def safe_dirname(title: str, year: int) -> str:
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    return f"{safe.strip().replace(' ', '_')}_{year}"


# ── Step 1: Search YouTube ────────────────────────────────────────────────────

def search_trailer(title: str, year: int, kind: str) -> dict | None:
    query = f"{title} {year} official trailer"
    result = run([
        "yt-dlp",
        "--quiet", "--no-warnings",
        "--dump-json",
        "--no-playlist",
        f"ytsearch1:{query}",
    ], timeout=30)

    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        info = json.loads(result.stdout.strip())
        return {
            "title":    title,
            "year":     year,
            "kind":     kind,
            "query":    query,
            "yt_id":    info.get("id"),
            "yt_title": info.get("title"),
            "yt_url":   info.get("webpage_url"),
            "duration": info.get("duration"),
        }
    except json.JSONDecodeError:
        return None


# ── Step 2: Download full trailer ─────────────────────────────────────────────

def download_trailer(info: dict, clip_dir: Path) -> bool:
    out_path = clip_dir / "clip.mp4"
    if out_path.exists() and out_path.stat().st_size > 10_000:
        log(f"    clip already exists — skipping download")
        return True

    tmp = clip_dir / "clip_tmp.mp4"
    result = run([
        "yt-dlp",
        "--quiet", "--no-warnings",
        "--format", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(tmp),
        info["yt_url"],
    ], timeout=600)

    if result.returncode != 0 or not tmp.exists():
        log(f"    FAILED (download): {result.stderr.strip()[:150]}")
        if tmp.exists(): tmp.unlink()
        return False

    tmp.rename(out_path)
    log(f"    clip.mp4  {out_path.stat().st_size / 1e6:.1f} MB")
    return True


# ── Step 3: Extract audio ─────────────────────────────────────────────────────

def extract_audio(clip_dir: Path) -> bool:
    dst = clip_dir / "audio.wav"
    if dst.exists():
        return True
    result = run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(clip_dir / "clip.mp4"),
        "-vn", "-ar", "44100", "-ac", "2", "-f", "wav", str(dst),
    ])
    if result.returncode != 0:
        log(f"    FAILED (audio): {result.stderr.strip()[:120]}")
        return False
    log(f"    audio.wav extracted")
    return True


# ── Step 4: Source separation ─────────────────────────────────────────────────

def separate_audio(clip_dir: Path) -> bool:
    vocals_dst    = clip_dir / "vocals.wav"
    no_vocals_dst = clip_dir / "no_vocals.wav"
    if vocals_dst.exists() and no_vocals_dst.exists():
        return True

    # Trim audio to max 5 min before separation to keep CPU time reasonable
    src      = clip_dir / "audio.wav"
    trimmed  = clip_dir / "audio_trim.wav"
    if not trimmed.exists():
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src), "-t", "300",
            "-ar", "44100", "-ac", "2", str(trimmed),
        ], capture_output=True)
    audio_in = trimmed if trimmed.exists() else src

    log(f"    Running demucs (CPU)...")
    tmp_out = clip_dir / "demucs_tmp"
    result = run([
        "python3", "-m", "demucs",
        "--two-stems", "vocals",
        "--device", "cpu",
        "-n", DEMUCS_MODEL,
        "--out", str(tmp_out),
        str(audio_in),
    ], timeout=900)

    if result.returncode != 0:
        log(f"    FAILED (demucs): {result.stderr.strip()[:200]}")
        shutil.rmtree(tmp_out, ignore_errors=True)
        return False

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


# ── Step 5: Keyframes ─────────────────────────────────────────────────────────

def extract_keyframes(clip_dir: Path) -> bool:
    # Use ffmpeg stderr to get duration (ffprobe not available)
    import re
    probe = subprocess.run(
        ["ffmpeg", "-i", str(clip_dir / "clip.mp4")],
        capture_output=True, text=True
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", probe.stderr)
    if match:
        h, m, s = match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)
    else:
        duration = 120.0

    all_ok = True
    for i, frac in enumerate((1/6, 3/6, 5/6)):
        dst = clip_dir / f"frame_{i}.jpg"
        if dst.exists():
            continue
        result = run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{duration * frac:.3f}",
            "-i", str(clip_dir / "clip.mp4"),
            "-frames:v", "1", "-q:v", "2", str(dst),
        ])
        if result.returncode != 0:
            all_ok = False

    if all_ok:
        log(f"    frame_0/1/2.jpg extracted")
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CLIPS_DIR.mkdir(exist_ok=True)

    all_titles = [(t, y, "movie") for t, y in MOVIES] + \
                 [(t, y, "tv")    for t, y in TV_SHOWS]

    # Load existing metadata if resuming
    if META_FILE.exists():
        with open(META_FILE) as f:
            metadata = json.load(f)
        done = {m["title"] for m in metadata}
        log(f"Resuming — {len(metadata)}/120 already processed.")
    else:
        metadata = []
        done     = set()

    stats = {"found": 0, "downloaded": 0, "audio": 0,
             "separated": 0, "frames": 0, "not_found": []}

    for i, (title, year, kind) in enumerate(all_titles, 1):
        if title in done:
            log(f"[{i}/120] {title} — already done, skipping")
            stats["found"] += 1; stats["downloaded"] += 1
            stats["audio"] += 1; stats["separated"]  += 1; stats["frames"] += 1
            continue

        log(f"\n[{i}/120] {title} ({year}) [{kind}]")

        # Search YouTube
        info = search_trailer(title, year, kind)
        if not info:
            log(f"    NOT FOUND on YouTube")
            stats["not_found"].append(title)
            metadata.append({"title": title, "year": year, "kind": kind, "error": "not found"})
            with open(META_FILE, "w") as f: json.dump(metadata, f, indent=2)
            continue

        log(f"    Found: \"{info['yt_title']}\"  ({info['duration']}s)  {info['yt_url']}")

        # Reject anything over 8 minutes — not a trailer
        if info["duration"] and info["duration"] > 480:
            log(f"    SKIPPED — too long ({info['duration']}s), likely not a trailer")
            metadata.append({"title": title, "year": year, "kind": kind, "error": "result too long, not a trailer"})
            with open(META_FILE, "w") as f: json.dump(metadata, f, indent=2)
            continue

        stats["found"] += 1

        clip_dir = CLIPS_DIR / safe_dirname(title, year)
        clip_dir.mkdir(exist_ok=True)
        info["local_dir"] = str(clip_dir)

        # Download
        ok_video = download_trailer(info, clip_dir)
        if not ok_video:
            info["error"] = "download failed"
            metadata.append(info)
            with open(META_FILE, "w") as f: json.dump(metadata, f, indent=2)
            continue
        stats["downloaded"] += 1

        # Audio extraction
        ok_audio = extract_audio(clip_dir)
        if ok_audio: stats["audio"] += 1

        # Source separation
        ok_sep = separate_audio(clip_dir) if ok_audio else False
        if ok_sep: stats["separated"] += 1

        # Keyframes
        ok_frames = extract_keyframes(clip_dir)
        if ok_frames: stats["frames"] += 1

        info["local_paths"] = {
            "clip":      str(clip_dir / "clip.mp4")      if ok_video  else None,
            "audio":     str(clip_dir / "audio.wav")     if ok_audio  else None,
            "vocals":    str(clip_dir / "vocals.wav")    if ok_sep    else None,
            "no_vocals": str(clip_dir / "no_vocals.wav") if ok_sep    else None,
            "frame_0":   str(clip_dir / "frame_0.jpg")   if ok_frames else None,
            "frame_1":   str(clip_dir / "frame_1.jpg")   if ok_frames else None,
            "frame_2":   str(clip_dir / "frame_2.jpg")   if ok_frames else None,
        }
        metadata.append(info)
        done.add(title)
        with open(META_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

    # Summary
    log(f"\n{'─'*55}")
    log(f"Complete.")
    log(f"  Trailers found      : {stats['found']}/120")
    log(f"  Videos downloaded   : {stats['downloaded']}/120")
    log(f"  Audio extracted     : {stats['audio']}/120")
    log(f"  Audio separated     : {stats['separated']}/120")
    log(f"  Keyframes extracted : {stats['frames']}/120")
    if stats["not_found"]:
        log(f"  Not found ({len(stats['not_found'])}): {', '.join(stats['not_found'][:5])}" +
            (" ..." if len(stats["not_found"]) > 5 else ""))
    log(f"  Metadata : {META_FILE}")
    log(f"  Clips    : {CLIPS_DIR}")
    log(f"{'─'*55}")


if __name__ == "__main__":
    main()
