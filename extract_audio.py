"""
extract_audio.py
================
Extracts audio from video clips associated with titles in the Genery DB.

How it works:
  1. Fetches titles + their top media item (by aesthetic_score) from Postgres
  2. Reads the HLS URL from media_items.file_path → scene → hls
  3. Converts b2:// storage URLs → CDN HTTP URLs
  4. Uses ffmpeg to download the HLS stream and extract audio as 16kHz mono WAV
  5. Saves one WAV file per title into the output directory

URL conversion:
  b2://genery-media-items/scene/...  →  {CDN_BASE}genery-media-items/scene/...

Output:
  audio/<title_cuid2>.wav

Usage:
  python extract_audio.py                   # 100 titles (default)
  python extract_audio.py --limit 20
  python extract_audio.py --output my_dir
  python extract_audio.py --cdn https://your-cdn.com/file/
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm

from db import pg

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CDN_BASE       = os.environ.get("CDN_BASE", "https://cdn.genery.online/file/")
B2_SCHEME      = "b2://"

AUDIO_SAMPLE_RATE = 16000   # Hz  — 16kHz is standard for speech/audio models
AUDIO_CHANNELS    = 1       # mono
FFMPEG_TIMEOUT    = 120     # seconds per clip before giving up


# ──────────────────────────────────────────────────────────────────────────────
# 1.  URL conversion
# ──────────────────────────────────────────────────────────────────────────────

def b2_to_cdn(b2_url: str, cdn_base: str = CDN_BASE) -> str | None:
    """
    Convert a b2:// storage URL to an HTTP CDN URL.

      b2://genery-media-items/scene/ttl_.../v/.../master.m3u8
      →  https://cdn.genery.online/file/genery-media-items/scene/ttl_.../v/.../master.m3u8
    """
    if not b2_url or not b2_url.startswith(B2_SCHEME):
        return None
    path = b2_url[len(B2_SCHEME):]          # strip "b2://"
    return cdn_base.rstrip("/") + "/" + path


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Database fetching
# ──────────────────────────────────────────────────────────────────────────────

def fetch_titles_with_clips(limit: int) -> list[dict]:
    """
    For each of the top `limit` titles (by popularity), fetch the single
    best media item (highest aesthetic_score) and its HLS URL.

    Returns list of dicts with keys:
      title_id, title_name, cuid2, type, year, hls_url
    """
    sql = text("""
        SELECT title_id, title_name, cuid2, type, year, file_path
        FROM (
            SELECT DISTINCT ON (t.id)
                t.id            AS title_id,
                t.title         AS title_name,
                t.cuid2         AS cuid2,
                t.type          AS type,
                t.year          AS year,
                t.popularity    AS popularity,
                mi.file_path    AS file_path
            FROM titles t
            JOIN media_items mi
              ON mi.title_id   = t.id
             AND mi.deleted_at IS NULL
             AND mi.file_path  IS NOT NULL
             AND mi.file_path::jsonb #>> '{scene,hls}' NOT LIKE '%/v/full\\_%'
            WHERE t.deleted_at IS NULL
            ORDER BY t.id, mi.aesthetic_score DESC NULLS LAST
        ) best_clip
        ORDER BY popularity DESC NULLS LAST
        LIMIT :limit
    """)
    with pg.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().fetchall()

    results = []
    for row in rows:
        fp = row["file_path"]
        if isinstance(fp, str):
            try:
                fp = json.loads(fp)
            except json.JSONDecodeError:
                continue

        hls_b2 = (fp.get("scene") or {}).get("hls")
        if not hls_b2:
            continue

        hls_url = b2_to_cdn(hls_b2)
        if not hls_url:
            continue

        results.append({
            "title_id":   int(row["title_id"]),
            "title_name": row["title_name"],
            "cuid2":      row["cuid2"],
            "type":       row["type"],
            "year":       row["year"],
            "hls_url":    hls_url,
        })

    return results


def fetch_hls_map(title_ids: list[int]) -> dict[str, str]:
    """
    Return {cuid2: hls_url} for a specific list of title IDs.
    Picks the best media item (highest aesthetic_score) per title.
    """
    sql = text("""
        SELECT DISTINCT ON (t.id)
            t.cuid2,
            mi.file_path
        FROM titles t
        JOIN media_items mi
          ON mi.title_id   = t.id
         AND mi.deleted_at IS NULL
         AND mi.file_path  IS NOT NULL
         AND mi.file_path::jsonb #>> '{scene,hls}' NOT LIKE '%/v/full\\_%'
        WHERE t.id = ANY(:ids)
          AND t.deleted_at IS NULL
        ORDER BY t.id, mi.aesthetic_score DESC NULLS LAST
    """)
    with pg.connect() as conn:
        rows = conn.execute(sql, {"ids": title_ids}).mappings().fetchall()

    result: dict[str, str] = {}
    for row in rows:
        fp = row["file_path"]
        if isinstance(fp, str):
            try:
                fp = json.loads(fp)
            except json.JSONDecodeError:
                continue
        hls_b2 = (fp.get("scene") or {}).get("hls")
        if not hls_b2:
            continue
        hls_url = b2_to_cdn(hls_b2)
        if hls_url:
            result[row["cuid2"]] = hls_url
    return result


def extract_missing_audio(
    titles: list[dict],
    out_dir: Path = Path("audio"),
    timeout: int = FFMPEG_TIMEOUT,
) -> None:
    """
    Extract audio for any titles that don't already have a WAV file.

    `titles` is a list of dicts that must contain at least:
      - "id"    (int)  — Postgres title ID
      - "cuid2" (str)  — used as the output filename stem

    Skips titles whose WAV already exists (idempotent).
    Intended to be called from title_vectors.py during --rebuild.
    """
    out_dir.mkdir(exist_ok=True)

    missing = [t for t in titles if not (out_dir / f"{t['cuid2']}.wav").exists()]
    already = len(titles) - len(missing)

    if not missing:
        print(f"  All {len(titles)} WAV files already present — skipping extraction.")
        return

    print(f"  {already} WAV files already exist. Extracting {len(missing)} missing …")

    hls_map = fetch_hls_map([int(t["id"]) for t in missing])
    no_hls  = sum(1 for t in missing if t["cuid2"] not in hls_map)
    if no_hls:
        print(f"  ⚠  {no_hls}/{len(missing)} titles have no HLS URL — will be skipped.")

    success, failed = 0, 0
    for t in tqdm(missing, unit="title"):
        cuid2   = t["cuid2"]
        hls_url = hls_map.get(cuid2)
        if not hls_url:
            continue
        out_path = out_dir / f"{cuid2}.wav"
        ok, err  = extract_audio(hls_url, out_path, timeout=timeout)
        if ok:
            success += 1
        else:
            failed += 1
            if out_path.exists():
                out_path.unlink()
            if err:
                title_name = t.get("title") or t.get("title_name") or cuid2
                print(f"  ✗  {title_name[:40]}: {err}")

    print(f"  Audio extraction — {success} ok · {failed} failed · {no_hls} no HLS")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Audio extraction via ffmpeg
# ──────────────────────────────────────────────────────────────────────────────

def _url_exists(url: str, timeout: int = 10) -> tuple[bool, str | None]:
    """
    Send a HEAD request to check whether the URL returns a 2xx/3xx status.
    Returns (True, None) if reachable or if SSL verification fails (let ffmpeg
    try with its own SSL stack), (False, reason) on definitive HTTP errors.
    """
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            if resp.status < 400:
                return True, None
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, str(e)


def extract_audio(hls_url: str, out_path: Path, timeout: int = FFMPEG_TIMEOUT) -> tuple[bool, str | None]:
    """
    Download an HLS stream and extract audio as a 16kHz mono WAV file.

    Performs a fast HEAD check before invoking ffmpeg so dead URLs are
    skipped immediately instead of waiting for the full ffmpeg timeout.

    ffmpeg command:
      ffmpeg -i <hls_url>          # input: HLS playlist (follows segments automatically)
             -vn                   # drop video stream
             -acodec pcm_s16le     # uncompressed 16-bit PCM
             -ar 16000             # resample to 16kHz
             -ac 1                 # mix to mono
             <out_path>

    Returns (True, None) on success, (False, error_message) on failure.
    """
    # Fast pre-check — skip ffmpeg entirely if the URL is already dead
    ok, err = _url_exists(hls_url)
    if not ok:
        return False, f"URL unreachable ({err})"

    cmd = [
        "ffmpeg",
        "-y",                        # overwrite output without asking
        "-loglevel", "error",        # suppress progress spam
        "-i", hls_url,
        "-vn",                       # no video
        "-acodec", "pcm_s16le",      # WAV codec
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS),
        str(out_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, None
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except FileNotFoundError:
        sys.exit("ffmpeg not found — please install it (brew install ffmpeg)")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────────────────────────────────────

def _diagnose(limit: int = 10) -> None:
    """
    Print a sample of raw file_path JSON values from media_items so you can
    verify the expected key structure (scene.hls, etc.) and actual URL values.
    """
    sql = text("""
        SELECT title_id, title, cuid2, file_path
        FROM (
            SELECT DISTINCT ON (t.id)
                t.id         AS title_id,
                t.title      AS title,
                t.cuid2      AS cuid2,
                t.popularity AS popularity,
                mi.file_path AS file_path
            FROM titles t
            JOIN media_items mi
              ON mi.title_id = t.id
             AND mi.deleted_at IS NULL
             AND mi.file_path IS NOT NULL
            WHERE t.deleted_at IS NULL
            ORDER BY t.id, mi.aesthetic_score DESC NULLS LAST
        ) best
        ORDER BY popularity DESC NULLS LAST
        LIMIT :limit
    """)
    with pg.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().fetchall()

    print(f"\n── Sampled {len(rows)} distinct titles ─────────────────────────────")
    for row in rows:
        fp = row["file_path"]
        if isinstance(fp, str):
            try:
                fp = json.loads(fp)
            except json.JSONDecodeError:
                pass
        print(f"\n  Title : {row['title']}")
        print(f"  cuid2 : {row['cuid2']}")
        if isinstance(fp, dict):
            for k, v in fp.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        print(f"    {k}.{sk}: {sv!r}")
                else:
                    print(f"    {k}: {v!r}")
        else:
            print(f"  file_path: {fp!r}")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio from title video clips.")
    parser.add_argument("--limit",    type=int, default=100,    help="Number of titles to process (default: 100)")
    parser.add_argument("--output",   type=str, default="audio", help="Output directory (default: ./audio)")
    parser.add_argument("--cdn",      type=str, default=CDN_BASE, help="CDN base URL for b2:// conversion")
    parser.add_argument("--timeout",  type=int, default=FFMPEG_TIMEOUT, help="ffmpeg timeout per clip in seconds")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print sample file_path JSON structures from DB and exit")
    args = parser.parse_args()

    if args.diagnose:
        _diagnose(limit=10)
        return  # unreachable (sys.exit inside), but keeps type checkers happy

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    # ── 1. Fetch titles + clip URLs ───────────────────────────────────────
    print(f"Fetching up to {args.limit} titles with video clips from Postgres …")
    titles = fetch_titles_with_clips(args.limit)
    if not titles:
        sys.exit("No titles with video clips found.")
    print(f"  Found {len(titles)} titles with HLS clips.")

    # ── 2. Extract audio for each title ──────────────────────────────────
    success, skipped, failed = [], [], []

    for t in tqdm(titles, unit="title"):
        out_path = out_dir / f"{t['cuid2']}.wav"

        # Skip if already extracted
        if out_path.exists():
            skipped.append(t)
            continue

        ok, err = extract_audio(t["hls_url"], out_path, timeout=args.timeout)

        if ok:
            success.append(t)
        else:
            failed.append({**t, "error": err})
            # Remove partial file if ffmpeg left one
            if out_path.exists():
                out_path.unlink()

    # ── 3. Summary ────────────────────────────────────────────────────────
    print(f"\n── Audio extraction summary ──────────────────────────────────")
    print(f"  Extracted : {len(success)}")
    print(f"  Skipped   : {len(skipped)}  (already existed)")
    print(f"  Failed    : {len(failed)}")
    print(f"  Output dir: {out_dir.resolve()}")

    if failed:
        print(f"\n── Failed titles ─────────────────────────────────────────────")
        for t in failed:
            print(f"  [{t['type']} {t['year']}] {t['title_name'][:50]}")
            print(f"    URL  : {t['hls_url']}")
            print(f"    Error: {t['error']}")

    if success:
        print(f"\n── Sample extracted files ────────────────────────────────────")
        for t in success[:5]:
            wav = out_dir / f"{t['cuid2']}.wav"
            size_kb = wav.stat().st_size // 1024
            print(f"  {wav.name}  ({size_kb} KB)  ← {t['title_name'][:40]}")


if __name__ == "__main__":
    main()
