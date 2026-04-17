"""
title_vectors.py
================
Builds one fixed-dimension vector per title from the Genery database,
indexes them in FAISS, and demonstrates similarity search + UMAP visualisation.

Vector layout  (1 693 dims, L2-normalised → cosine-equivalent search)
  [    0 :  384 ]  text     – mean-pooled all-MiniLM-L6-v2 on title+desc+genres+keywords+cast
  [  384 : 1152 ]  visual   – mean of SigLIP2 ViT-B vectors fetched from Qdrant (zeros if none)
  [ 1152 : 1664 ]  audio    – mean-pooled Whisper base encoder hidden states (zeros if no WAV)
  [ 1664 : 1693 ]  metadata – year · popularity · imdb_rating · is_adult
                              type one-hot (×5) · genre multi-hot (×20)

Usage
-----
  python title_vectors.py            # 100 titles (default)
  python title_vectors.py --limit 300
  python title_vectors.py --rebuild  # force rebuild even if cache exists

Audio pre-requisite:
  Run extract_audio.py first to populate the ./audio/ directory with WAV files.
  Titles without a WAV file get a zero audio sub-vector.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

import argparse
import json
import math
import re
import sys
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import text
from tqdm import tqdm

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from llama_index.core.node_parser import SentenceSplitter
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from transformers import ClapModel, ClapProcessor
    HAS_CLAP = True
except ImportError:
    HAS_CLAP = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from db import pg, qdrant, Collections


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

TEXT_DIM    = 384
VISUAL_DIM  = 768    # SigLIP2 ViT-B (Base) vectors stored in Qdrant STILLS collection
AUDIO_DIM   = 512    # CLAP larger_clap_general audio embedding
META_DIM    = 29     # breakdown: see encode_metadata
TOTAL_DIM   = TEXT_DIM + VISUAL_DIM + AUDIO_DIM + META_DIM   # 1 693

QDRANT_VECTOR_NAME = "siglip-768"

TEXT_MODEL_ID    = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_MAX_LEN     = 256   # max tokens per chunk fed to the transformer
CHUNK_SIZE       = 256   # LlamaIndex SentenceSplitter chunk size (tokens)
CHUNK_OVERLAP    = 32    # overlap between consecutive chunks (tokens)

SIGLIP2_MODEL_ID  = "google/siglip2-base-patch16-224"
CLAP_MODEL_ID     = "laion/larger_clap_general"   # CLAP → 512-dim shared text-audio space
AUDIO_DIR         = Path("audio")   # where extract_audio.py saves WAV files

# Vector slice boundaries
TEXT_SLICE   = slice(0,    384)    # all-MiniLM text sub-space
VISUAL_SLICE = slice(384,  1152)   # SigLIP2 visual sub-space
AUDIO_SLICE  = slice(1152, 1664)   # Whisper audio sub-space
META_SLICE   = slice(1664, 1693)   # metadata

# Query-time score combination weight
SEARCH_ALPHA = 0.5

YEAR_MIN, YEAR_MAX = 1900, 2025

KNOWN_TYPES = ["movie", "tvSeries", "tvEpisode", "musicVideo", "commercial"]

KNOWN_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "Thriller", "Western",
    "Family", "Biography", "History", "Sport", "War",
]

INDEX_PATH   = Path("title_index.faiss")
META_PATH    = Path("title_meta.json")
VECTORS_PATH = Path("title_vectors.npy")
UMAP_2D_DIR  = Path("2d_umaps")
UMAP_3D_DIR  = Path("3d_umaps")
UMAP_PATH    = UMAP_3D_DIR / "umap_titles.html"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Database fetching
# ──────────────────────────────────────────────────────────────────────────────

def fetch_titles(limit: int) -> list[dict]:
    """
    Fetch `limit` titles split evenly across movies, tvSeries, and musicVideos
    (one-third each, ordered by popularity within each type).
    If limit is not divisible by 3, the remainder goes to movies.
    """
    per_type, remainder = divmod(limit, 3)
    type_limits = {
        "movie":       per_type + remainder,
        "tvSeries":    per_type,
        "musicVideo":  per_type,
    }

    sql = text("""
        SELECT
            t.id,
            t.cuid2,
            t.title,
            t.type,
            t.description,
            t.year,
            t.popularity,
            t.is_adult,
            t.genres,
            t.keywords,
            t.ratings,
            COUNT(mi.id) AS media_count
        FROM titles t
        LEFT JOIN media_items mi
               ON mi.title_id = t.id
              AND mi.deleted_at IS NULL
        WHERE t.deleted_at IS NULL
          AND t.type = :type
        GROUP BY t.id
        ORDER BY t.popularity DESC NULLS LAST
        LIMIT :limit
    """)

    rows = []
    with pg.connect() as conn:
        for title_type, n in type_limits.items():
            result = conn.execute(sql, {"type": title_type, "limit": n}).mappings().fetchall()
            rows.extend(result)

    return [dict(r) for r in rows]


VISUAL_ITEMS_PER_TITLE = 50   # top-N stills by aesthetic_score to average per title
PERSONS_PER_TITLE      = 20   # max cast/crew entries to include per title


def fetch_visual_vectors(title_ids: list[int]) -> dict[int, np.ndarray]:
    """
    Return {title_id: mean_vector(768)} by fetching siglip-768 vectors from
    Qdrant for the top-N media_items (by aesthetic_score) per title, then averaging.

    Steps:
      1. Postgres: get top-N (title_id, point_id) pairs per title
      2. Qdrant:   retrieve the siglip-768 named vector for each point_id
      3. Average   all vectors that share the same title_id
    """
    if not title_ids:
        return {}

    # 1. Fetch top-N (title_id, point_id) per title, ranked by aesthetic_score
    sql = text("""
        SELECT title_id, point_id
        FROM (
            SELECT title_id, point_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY title_id
                       ORDER BY aesthetic_score DESC NULLS LAST
                   ) AS rn
            FROM media_items
            WHERE title_id = ANY(:ids)
              AND deleted_at IS NULL
              AND point_id IS NOT NULL
        ) ranked
        WHERE rn <= :n
    """)
    with pg.connect() as conn:
        rows = conn.execute(sql, {"ids": title_ids, "n": VISUAL_ITEMS_PER_TITLE}).fetchall()

    if not rows:
        return {}

    # Build lookup: point_id (str) → title_id
    point_to_title: dict[str, int] = {
        str(point_id): int(title_id) for title_id, point_id in rows
    }
    point_ids = list(point_to_title.keys())

    # 2. Fetch vectors from Qdrant in batches to avoid oversized requests
    BATCH_SIZE = 512
    accum: dict[int, list[np.ndarray]] = {}
    total_batches = (len(point_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Fetching {len(point_ids)} point vectors from Qdrant ({total_batches} batches) …")

    for i in tqdm(range(0, len(point_ids), BATCH_SIZE), total=total_batches, unit="batch"):
        batch = point_ids[i : i + BATCH_SIZE]
        results = qdrant.retrieve(
            collection_name=Collections.STILLS,
            ids=batch,
            with_vectors=[QDRANT_VECTOR_NAME],
        )
        for point in results:
            vec = point.vector
            if isinstance(vec, dict):
                vec = vec.get(QDRANT_VECTOR_NAME)
            if vec is None:
                continue
            arr = np.array(vec, dtype=np.float32)
            if arr.shape[0] == VISUAL_DIM:
                tid = point_to_title[str(point.id)]
                accum.setdefault(tid, []).append(arr)

    # 3. Average per title
    output = {}
    for tid, vecs in accum.items():
        # Calculate the mean (Magnitude shrinks here!)
        mean_vec = np.mean(vecs, axis=0)
        
        # FIX: Re-inflate to Unit Length (1.0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-9: 
            mean_vec = mean_vec / norm
            
        output[tid] = mean_vec
        
    return output


def fetch_title_persons(title_ids: list[int]) -> dict[int, list[dict]]:
    """
    Return {title_id: [{name, role, role_description}]} for cast & crew.
    Capped at PERSONS_PER_TITLE entries per title, prioritising directors
    and lead actors first.
    """
    if not title_ids:
        return {}

    sql = text("""
        SELECT
            tp.title_id,
            p.name,
            tp.role,
            tp.role_description
        FROM title_persons tp
        JOIN persons p ON p.id = tp.person_id
        WHERE tp.title_id = ANY(:ids)
          AND p.name IS NOT NULL
        ORDER BY
            tp.title_id,
            CASE tp.role
                WHEN 'Director'  THEN 1
                WHEN 'Actor'     THEN 2
                WHEN 'Writer'    THEN 3
                ELSE                  4
            END
    """)
    with pg.connect() as conn:
        rows = conn.execute(sql, {"ids": title_ids}).fetchall()

    result: dict[int, list[dict]] = {}
    counts: dict[int, int] = {}
    for title_id, name, role, role_description in rows:
        tid = int(title_id)
        if counts.get(tid, 0) >= PERSONS_PER_TITLE:
            continue
        result.setdefault(tid, []).append({
            "name":             name,
            "role":             role,
            "role_description": role_description,
        })
        counts[tid] = counts.get(tid, 0) + 1

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Text encoder
# ──────────────────────────────────────────────────────────────────────────────

_tokenizer  = None
_text_model = None
_splitter   = None


def _load_text_model():
    global _tokenizer, _text_model, _splitter
    if _tokenizer is None:
        if not HAS_TRANSFORMERS:
            raise RuntimeError("pip install transformers")
        print(f"Loading text model ({TEXT_MODEL_ID}) …")
        _tokenizer  = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        _text_model = AutoModel.from_pretrained(TEXT_MODEL_ID)
        if HAS_TORCH:
            _text_model.eval()
    if _splitter is None:
        if not HAS_LLAMA_INDEX:
            raise RuntimeError("pip install llama-index-core")
        _splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )


def _mean_pool(model_output, attention_mask) -> np.ndarray:
    token_emb   = model_output.last_hidden_state           # (1, seq, dim)
    mask_expand = attention_mask.unsqueeze(-1).float()     # (1, seq, 1)
    summed      = (token_emb * mask_expand).sum(dim=1)     # (1, dim)
    counts      = mask_expand.sum(dim=1).clamp(min=1e-9)   # (1, 1)
    return (summed / counts).squeeze(0).detach().cpu().numpy()


def _embed_chunk(chunk: str) -> np.ndarray:
    """Embed a single text chunk → (TEXT_DIM,) unit vector."""
    encoded = _tokenizer(
        chunk,
        padding=True,
        truncation=True,
        max_length=TEXT_MAX_LEN,
        return_tensors="pt" if HAS_TORCH else "np",
    )
    if HAS_TORCH:
        with torch.no_grad():
            output = _text_model(**encoded)
        vec = _mean_pool(output, encoded["attention_mask"])
    else:
        output = _text_model(**encoded)
        vec = output.last_hidden_state[0].mean(axis=0)
    return vec.astype(np.float32)


def encode_query_text(query: str) -> np.ndarray:
    """
    Encode a short ad-hoc query string → 384-dim unit vector.
    Bypasses chunking — the query is a single short string, not a
    multi-field document, so splitting adds no value.
    """
    _load_text_model()
    vec = _embed_chunk(query)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec


def encode_text(row: dict, persons: list[dict] | None = None) -> np.ndarray:
    """
    Build the full text for a title, split into overlapping chunks via
    LlamaIndex SentenceSplitter, embed each chunk, then mean-pool all
    chunk vectors into one TEXT_DIM unit vector.

    Text structure (no truncation before splitting):
      - Title
      - Description
      - Genres
      - Keywords
      - Cast & Crew
    """
    _load_text_model()

    # ── Build full text (no truncation) ──────────────────────────────────
    parts: list[str] = [row.get("title") or ""]
    if row.get("description"):
        parts.append(row["description"])
    genres = row.get("genres") or []
    if genres:
        parts.append("Genres: " + ", ".join(genres))
    keywords = row.get("keywords") or []
    if keywords:
        parts.append(", ".join(keywords))   # all keywords, no cap
    if persons:
        crew_parts: list[str] = []
        for p in persons:
            entry = f"{p['role']}: {p['name']}"
            if p.get("role_description"):
                entry += f" ({p['role_description']})"
            crew_parts.append(entry)
        if crew_parts:
            parts.append("Cast & Crew: " + ", ".join(crew_parts))

    full_text = " | ".join(p for p in parts if p)

    # ── Split into overlapping chunks ─────────────────────────────────────
    chunks = _splitter.split_text(full_text)

    # ── Embed each chunk and mean-pool ────────────────────────────────────
    chunk_vecs = np.stack([_embed_chunk(c) for c in chunks])   # (N_chunks, TEXT_DIM)
    vec = chunk_vecs.mean(axis=0)                               # (TEXT_DIM,)

    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SigLIP2 text encoder  (for query-time visual sub-space search)
# ──────────────────────────────────────────────────────────────────────────────

_siglip2_model     = None
_siglip2_processor = None


def _load_siglip2():
    global _siglip2_model, _siglip2_processor
    if _siglip2_model is None:
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            raise RuntimeError("pip install transformers torch")
        print(f"Loading SigLIP2 text encoder ({SIGLIP2_MODEL_ID}) …")
        from transformers import AutoProcessor
        _siglip2_processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
        _siglip2_model     = AutoModel.from_pretrained(SIGLIP2_MODEL_ID)
        _siglip2_model.eval()


def encode_query_visual(text: str) -> np.ndarray:
    """
    Encode a text query into the SigLIP2 visual embedding space (768-dim).
    Uses the text_model pooler_output directly — same 768-dim space as the
    siglip-768 image vectors stored in Qdrant.
    """
    _load_siglip2()
    inputs = _siglip2_processor(
        text=[text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        vec = _siglip2_model.text_model(**inputs).pooler_output[0]  # (768,)
    vec = vec.cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Audio encoder  (CLAP → 512-dim shared text-audio space)
# ──────────────────────────────────────────────────────────────────────────────

_clap_model     = None
_clap_processor = None


def _load_clap():
    global _clap_model, _clap_processor
    if _clap_model is None:
        if not HAS_CLAP:
            raise RuntimeError("pip install transformers  # ClapModel requires transformers>=4.31")
        if not HAS_TORCH:
            raise RuntimeError("pip install torch")
        print(f"Loading CLAP ({CLAP_MODEL_ID}) …")
        _clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        _clap_model     = ClapModel.from_pretrained(CLAP_MODEL_ID)
        _clap_model.eval()


def build_audio_map(cuid2s: list[str]) -> dict[str, np.ndarray]:
    """
    For each title cuid2, look for a WAV file in AUDIO_DIR and encode it
    using CLAP's audio encoder.

    Pipeline per file:
      1. Load WAV with scipy → float32 array at 16kHz mono
      2. Pass through ClapProcessor + ClapModel.get_audio_features → (512,)
      3. L2-normalise → unit vector

    Returns {cuid2: audio_vector(512)}.  Titles with no WAV are omitted.
    The resulting vectors share the same 512-dim space as encode_query_audio(),
    so text queries can be compared directly against stored audio vectors.
    """
    if not cuid2s:
        return {}

    _load_clap()

    import scipy.io.wavfile as wav_io
    import scipy.signal as sig_resample

    CLAP_SR = 48000   # CLAP was trained at 48kHz

    result: dict[str, np.ndarray] = {}
    for cuid2 in cuid2s:
        wav_path = AUDIO_DIR / f"{cuid2}.wav"
        if not wav_path.exists():
            continue
        try:
            sr, audio = wav_io.read(str(wav_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)           # stereo → mono
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio /= 32768.0                     # int16 → float32 [-1, 1]
            if sr != CLAP_SR:
                num_samples = int(len(audio) * CLAP_SR / sr)
                audio = sig_resample.resample(audio, num_samples)
            inputs = _clap_processor(
                audio=audio, sampling_rate=CLAP_SR, return_tensors="pt"
            )
            with torch.no_grad():
                audio_out = _clap_model.audio_model(**inputs)
                vec = _clap_model.audio_projection(audio_out.pooler_output)  # (1, 512)
            vec = vec[0].cpu().numpy().astype(np.float32)           # (512,)
            norm = np.linalg.norm(vec)
            result[cuid2] = (vec / norm) if norm > 0 else vec
        except Exception as e:
            print(f"  ⚠  Audio encoding failed for {cuid2}: {e}")

    return result


def encode_audio(cuid2: str, audio_map: dict[str, np.ndarray]) -> np.ndarray:
    """Return audio vector or zeros if no WAV exists for this title."""
    vec = audio_map.get(cuid2)
    if vec is None:
        return np.zeros(AUDIO_DIM, dtype=np.float32)
    return vec


def encode_query_audio(query: str) -> np.ndarray:
    """
    Encode a text query into CLAP's 512-dim audio embedding space.
    Because CLAP is trained contrastively on (audio, text) pairs, this vector
    is directly comparable to stored CLAP audio vectors via inner product.
    """
    _load_clap()
    inputs = _clap_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_out = _clap_model.text_model(**inputs)
        vec = _clap_model.text_projection(text_out.pooler_output)[0]  # (512,)
    vec = vec.cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Visual encoder  (Qdrant → mean SigLIP2 image vectors per title)
# ──────────────────────────────────────────────────────────────────────────────

def encode_visual(title_id: int, visual_map: dict[int, np.ndarray]) -> np.ndarray:
    """Return mean visual vector or a zero vector if no media items exist."""
    vec = visual_map.get(int(title_id))
    if vec is None:
        return np.zeros(VISUAL_DIM, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Metadata encoder
# ──────────────────────────────────────────────────────────────────────────────

def encode_metadata(row: dict) -> np.ndarray:
    """
    29-dim structured feature vector (all values in [0, 1]):
      [0]      year          – normalised to [YEAR_MIN, YEAR_MAX]
      [1]      popularity    – log-scaled, normalised
      [2]      imdb_rating   – /10
      [3]      is_adult      – binary
      [4:9]    type          – one-hot over KNOWN_TYPES (5 dims)
      [9:29]   genres        – multi-hot over KNOWN_GENRES (20 dims)
    """
    vec = np.zeros(META_DIM, dtype=np.float32)

    # year
    year = row.get("year")
    if year:
        vec[0] = float(np.clip((int(year) - YEAR_MIN) / (YEAR_MAX - YEAR_MIN), 0.0, 1.0))

    # popularity (log scale; cap at log(1001) ≈ 7)
    pop = row.get("popularity")
    if pop and pop > 0:
        vec[1] = float(np.clip(math.log1p(float(pop)) / math.log1p(1000), 0.0, 1.0))

    # imdb rating
    ratings = row.get("ratings")
    if ratings:
        if isinstance(ratings, str):
            try:
                ratings = json.loads(ratings)
            except json.JSONDecodeError:
                ratings = {}
        if isinstance(ratings, dict):
            imdb_rating = (ratings.get("imdb") or {}).get("rating")
            if imdb_rating is not None:
                vec[2] = float(np.clip(float(imdb_rating) / 10.0, 0.0, 1.0))

    # is_adult
    vec[3] = 1.0 if row.get("is_adult") else 0.0

    # type one-hot
    title_type = (row.get("type") or "").strip()
    if title_type in KNOWN_TYPES:
        vec[4 + KNOWN_TYPES.index(title_type)] = 1.0

    # genre multi-hot
    genres = row.get("genres") or []
    if isinstance(genres, str):
        try:
            genres = json.loads(genres)
        except json.JSONDecodeError:
            genres = []
    for g in genres:
        g_stripped = g.strip()
        if g_stripped in KNOWN_GENRES:
            vec[9 + KNOWN_GENRES.index(g_stripped)] = 1.0

    return vec


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Fusion
# ──────────────────────────────────────────────────────────────────────────────

def _l2_normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector; returns the zero vector unchanged."""
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32) if norm > 0 else v.astype(np.float32)


def build_title_vector(
    row: dict,
    visual_map: dict[int, np.ndarray],
    persons_map: dict[int, list[dict]],
    audio_map: dict[str, np.ndarray],
) -> np.ndarray:
    """
    L2-normalise each modality vector individually, concatenate, then
    L2-normalise the fused result.
    Missing modalities (no WAV, no Qdrant vector) contribute zero sub-vectors,
    so the final norm is driven by whichever modalities are present.
    """
    persons = persons_map.get(int(row["id"]))
    t_vec = _l2_normalise(encode_text(row, persons))       # (384,)
    v_vec = _l2_normalise(encode_visual(row["id"], visual_map))  # (768,)
    a_vec = _l2_normalise(encode_audio(row["cuid2"], audio_map)) # (512,)
    m_vec = _l2_normalise(encode_metadata(row))            # ( 29,)
    fused = np.concatenate([t_vec, v_vec, a_vec, m_vec])   # (1693,)
    norm  = np.linalg.norm(fused)
    return (fused / norm).astype(np.float32) if norm > 0 else fused.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  FAISS index
# ──────────────────────────────────────────────────────────────────────────────

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    IndexFlatIP (inner product) on L2-normalised vectors is equivalent to
    cosine similarity — exact nearest-neighbour, no approximation.
    """
    index = faiss.IndexFlatIP(TOTAL_DIM)
    index.add(vectors)
    return index


def build_sub_indices(
    vectors: np.ndarray,
) -> tuple[faiss.IndexFlatIP, faiss.IndexFlatIP, faiss.IndexFlatIP]:
    """
    Build three separate FAISS indices from sub-slices of the full vector:
      - text_index  : vectors[:, TEXT_SLICE]   (384-dim, all-MiniLM space)
      - visual_index: vectors[:, VISUAL_SLICE] (768-dim, SigLIP2 space)
      - audio_index : vectors[:, AUDIO_SLICE]  (512-dim, Whisper encoder space)

    Each sub-slice is L2-normalised independently so inner product = cosine.
    Note: audio_index is built for completeness; interactive_search only queries
    text and visual sub-spaces since there is no text→audio encoder.
    """
    def _normalise(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (mat / norms).astype(np.float32)

    text_vecs   = _normalise(vectors[:, TEXT_SLICE])
    visual_vecs = _normalise(vectors[:, VISUAL_SLICE])
    audio_vecs  = _normalise(vectors[:, AUDIO_SLICE])

    text_index   = faiss.IndexFlatIP(TEXT_DIM)
    visual_index = faiss.IndexFlatIP(VISUAL_DIM)
    audio_index  = faiss.IndexFlatIP(AUDIO_DIM)
    text_index.add(text_vecs)
    visual_index.add(visual_vecs)
    audio_index.add(audio_vecs)

    return text_index, visual_index, audio_index


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Similarity search demo
# ──────────────────────────────────────────────────────────────────────────────

def interactive_search(
    text_index:   faiss.IndexFlatIP,
    visual_index: faiss.IndexFlatIP,
    audio_index:  faiss.IndexFlatIP,
    titles: list[dict],
    vectors: np.ndarray,
    k: int = 5,
    alpha: float = SEARCH_ALPHA,
) -> None:
    """
    Interactive tri-encoder search loop.

    For each query:
      1. Encode with all-MiniLM  → search text sub-index   (384-dim)
      2. Encode with SigLIP2     → search visual sub-index (768-dim)
      3. Encode with CLAP        → search audio sub-index  (512-dim)
      4. Combine: alpha * text_score + beta * visual_score + (1-alpha-beta) * audio_score

    The query does not need to exist in the indexed set.
    Commands: 'quit' · 'list' · 'alpha <0-1>' · 'beta <0-1>'
    """
    beta = 1.0 - alpha   # visual weight; audio weight = 1 - alpha - beta
    print("\n" + "=" * 70)
    print("  TRI-ENCODER SIMILARITY SEARCH  (text + visual + audio)")
    print(f"  α={alpha:.2f} (text)  β={beta:.2f} (visual)  γ={max(0, 1-alpha-beta):.2f} (audio)")
    print("  Enter any title, name, or description to search.")
    print("  Commands: 'quit' · 'list' · 'alpha <value>' · 'beta <value>'")
    print("=" * 70)

    while True:
        try:
            raw = input("\n  Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            print("  Exiting.")
            break
        if raw.lower() == "list":
            print()
            for i, t in enumerate(titles):
                print(f"  {i+1:3}. {t['title'][:50]:<50}  {t.get('type','?'):12}  {t.get('year','?')}")
            continue
        if raw.lower().startswith("alpha "):
            try:
                alpha = float(raw.split()[1])
                alpha = max(0.0, min(1.0, alpha))
                beta  = max(0.0, min(1.0 - alpha, beta))
                print(f"  α={alpha:.2f} (text)  β={beta:.2f} (visual)  γ={max(0, 1-alpha-beta):.2f} (audio)")
            except (IndexError, ValueError):
                print("  Usage: alpha <0.0 – 1.0>")
            continue
        if raw.lower().startswith("beta "):
            try:
                beta = float(raw.split()[1])
                beta = max(0.0, min(1.0 - alpha, beta))
                print(f"  α={alpha:.2f} (text)  β={beta:.2f} (visual)  γ={max(0, 1-alpha-beta):.2f} (audio)")
            except (IndexError, ValueError):
                print("  Usage: beta <0.0 – 1.0>")
            continue

        print("  Encoding query …")

        # ── Text sub-space query (all-MiniLM) ────────────────────────────
        t_vec = encode_query_text(raw).reshape(1, -1)                 # (1, 384)
        text_scores, text_idxs = text_index.search(t_vec, len(titles))

        # ── Visual sub-space query (SigLIP2 text encoder) ────────────────
        v_vec = encode_query_visual(raw).reshape(1, -1)               # (1, 768)
        visual_scores, visual_idxs = visual_index.search(v_vec, len(titles))

        # ── Audio sub-space query (CLAP text encoder) ─────────────────────
        a_vec = encode_query_audio(raw).reshape(1, -1)                # (1, 512)
        audio_scores, audio_idxs = audio_index.search(a_vec, len(titles))

        # ── Combine scores ────────────────────────────────────────────────
        text_score_map  = {int(idx): float(s) for idx, s in zip(text_idxs[0],   text_scores[0])}
        visual_score_map= {int(idx): float(s) for idx, s in zip(visual_idxs[0], visual_scores[0])}
        audio_score_map = {int(idx): float(s) for idx, s in zip(audio_idxs[0],  audio_scores[0])}

        gamma = max(0.0, 1.0 - alpha - beta)
        combined = {
            i: alpha  * text_score_map.get(i,   0.0)
             + beta   * visual_score_map.get(i,  0.0)
             + gamma  * audio_score_map.get(i,   0.0)
            for i in range(len(titles))
        }
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

        # ── Display results ───────────────────────────────────────────────
        print(f"\n  Query  →  «{raw}»  [α={alpha:.2f} text · β={beta:.2f} visual · γ={gamma:.2f} audio]")
        print("  " + "-" * 75)
        print(f"  {'#':<4} {'score':<8} {'T-score':<8} {'V-score':<8} {'A-score':<8}  title")
        print("  " + "-" * 75)
        for rank, (idx, score) in enumerate(ranked, 1):
            t = titles[idx]
            visual_flag = "🖼" if int(t["id"]) in _visual_coverage else "  "
            print(
                f"  {rank:<4} {score:<8.4f} "
                f"{text_score_map.get(idx, 0):<8.4f} "
                f"{visual_score_map.get(idx, 0):<8.4f} "
                f"{audio_score_map.get(idx, 0):<8.4f} "
                f"{visual_flag} {t['title'][:40]:<40} "
                f"{t.get('type','?'):12} {t.get('year','?')}"
            )

        # ── UMAP: project query into the embedding space and save HTML ────
        if HAS_UMAP and HAS_PLOTLY:
            # Build a full 1693-dim query vector: text + visual + audio sub-vecs
            query_full = np.zeros(TOTAL_DIM, dtype=np.float32)
            query_full[TEXT_SLICE]   = t_vec.flatten()
            query_full[VISUAL_SLICE] = v_vec.flatten()
            query_full[AUDIO_SLICE]  = a_vec.flatten()
            norm = np.linalg.norm(query_full)
            if norm > 0:
                query_full /= norm

            result_idxs = [(rank, idx, score) for rank, (idx, score) in enumerate(ranked, 1)]
            safe_name = "".join(c if c.isalnum() else "_" for c in raw)[:40]
            umap_out_3d = UMAP_3D_DIR / f"umap_query_{safe_name}.html"
            visualise(vectors, titles, out_path=umap_out_3d,
                      query_vec=query_full, query_label=raw,
                      result_idxs=result_idxs)
            umap_out_2d = UMAP_2D_DIR / f"umap_query_{safe_name}.png"
            visualise_2d(vectors, titles, out_path=umap_out_2d,
                         query_vec=query_full, query_label=raw,
                         result_idxs=result_idxs)


# ──────────────────────────────────────────────────────────────────────────────
# 9.  UMAP visualisation
# ──────────────────────────────────────────────────────────────────────────────

def visualise(
    vectors: np.ndarray,
    titles: list[dict],
    out_path: Path = UMAP_PATH,
    query_vec: np.ndarray | None = None,
    query_label: str = "Query",
    result_idxs: list[tuple[int, int, float]] | None = None,
) -> None:
    # result_idxs: list of (rank, corpus_idx, score) for the top-k hits
    if not HAS_UMAP:
        print("  umap-learn not installed — skipping visualisation.  (pip install umap-learn)")
        return
    if not HAS_PLOTLY:
        print("  plotly not installed — skipping visualisation.  (pip install plotly)")
        return

    n = len(titles)
    n_neighbors = min(15, n - 1)

    # If a query vector is provided, append it so UMAP places it in the same space
    all_vecs = vectors if query_vec is None else np.vstack([vectors, query_vec.reshape(1, -1)])

    print(f"\nRunning UMAP on {len(all_vecs)} points (n_neighbors={n_neighbors}) …")
    reducer = umap_lib.UMAP(
        n_components=3,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
    )
    emb = reducer.fit_transform(all_vecs)   # (N [+1], 3)

    corpus_emb = emb[:n]
    query_emb  = emb[n] if query_vec is not None else None

    # ── colour by title type ──────────────────────────────────────────────
    type_labels  = [t.get("type") or "unknown" for t in titles]
    unique_types = sorted(set(type_labels))
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    type_color = {tp: palette[i % len(palette)] for i, tp in enumerate(unique_types)}

    traces = []

    # ── one trace per title type ──────────────────────────────────────────
    for tp in unique_types:
        idxs = [i for i, t in enumerate(titles) if (t.get("type") or "unknown") == tp]
        traces.append(go.Scatter3d(
            x=corpus_emb[idxs, 0],
            y=corpus_emb[idxs, 1],
            z=corpus_emb[idxs, 2],
            mode="markers+text",
            name=tp,
            text=[titles[i]["title"][:30] for i in idxs],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=5, color=type_color[tp], opacity=0.75),
        ))

    # ── top-k result highlights ───────────────────────────────────────────
    if result_idxs:
        res_x, res_y, res_z, res_text = [], [], [], []
        for rank, idx, score in result_idxs:
            res_x.append(corpus_emb[idx, 0])
            res_y.append(corpus_emb[idx, 1])
            res_z.append(corpus_emb[idx, 2])
            res_text.append(f"#{rank} {titles[idx]['title'][:30]} ({score:.3f})")
        traces.append(go.Scatter3d(
            x=res_x, y=res_y, z=res_z,
            mode="markers+text",
            name="Results",
            text=res_text,
            textposition="top center",
            textfont=dict(size=10, color="gold"),
            marker=dict(
                size=9, color="gold", symbol="circle",
                line=dict(color="black", width=2),
            ),
        ))

    # ── query point ───────────────────────────────────────────────────────
    if query_emb is not None:
        traces.append(go.Scatter3d(
            x=[query_emb[0]],
            y=[query_emb[1]],
            z=[query_emb[2]],
            mode="markers+text",
            name="Query",
            text=[f"Query: «{query_label}»"],
            textposition="top center",
            textfont=dict(size=11, color="red"),
            marker=dict(
                size=10, color="red", symbol="diamond",
                line=dict(color="black", width=2),
            ),
        ))

    title_str = f"Title embedding space — UMAP 3D  ({n} titles, {TOTAL_DIM}-dim vectors)"
    if query_vec is not None:
        title_str += f"  |  Query: «{query_label}»"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title_str,
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend_title="Title type",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    out_html = out_path.with_suffix(".html")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"  Saved → {out_html}")


def visualise_2d(
    vectors: np.ndarray,
    titles: list[dict],
    out_path: Path = UMAP_2D_DIR / "umap_titles.png",
    query_vec: np.ndarray | None = None,
    query_label: str = "Query",
    result_idxs: list[tuple[int, int, float]] | None = None,
) -> None:
    # result_idxs: list of (rank, corpus_idx, score) for the top-k hits
    if not HAS_UMAP:
        print("  umap-learn not installed — skipping 2D visualisation.  (pip install umap-learn)")
        return

    n = len(titles)
    n_neighbors = min(15, n - 1)

    all_vecs = vectors if query_vec is None else np.vstack([vectors, query_vec.reshape(1, -1)])

    print(f"\nRunning 2D UMAP on {len(all_vecs)} points (n_neighbors={n_neighbors}) …")
    reducer = umap_lib.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
    )
    emb = reducer.fit_transform(all_vecs)   # (N [+1], 2)

    corpus_emb = emb[:n]
    query_emb  = emb[n] if query_vec is not None else None

    type_labels  = [t.get("type") or "unknown" for t in titles]
    unique_types = sorted(set(type_labels))
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    type_color = {tp: palette[i % len(palette)] for i, tp in enumerate(unique_types)}

    fig, ax = plt.subplots(figsize=(12, 8))

    for tp in unique_types:
        idxs = [i for i, t in enumerate(titles) if (t.get("type") or "unknown") == tp]
        ax.scatter(
            corpus_emb[idxs, 0], corpus_emb[idxs, 1],
            c=type_color[tp], label=tp, s=20, alpha=0.75,
        )
        for i in idxs:
            ax.annotate(
                titles[i]["title"][:20],
                (corpus_emb[i, 0], corpus_emb[i, 1]),
                fontsize=5, alpha=0.6,
            )

    if result_idxs:
        for rank, idx, score in result_idxs:
            ax.scatter(
                corpus_emb[idx, 0], corpus_emb[idx, 1],
                c="gold", s=100, marker="*", zorder=4,
                edgecolors="black", linewidths=0.8,
            )
            ax.annotate(
                f"#{rank} {titles[idx]['title'][:25]}",
                (corpus_emb[idx, 0], corpus_emb[idx, 1]),
                fontsize=7, color="darkorange", fontweight="bold",
                xytext=(4, 4), textcoords="offset points",
            )
        ax.scatter([], [], c="gold", s=100, marker="*",
                   edgecolors="black", linewidths=0.8, label="Results")

    if query_emb is not None:
        ax.scatter(
            [query_emb[0]], [query_emb[1]],
            c="red", s=120, marker="D", zorder=5, label=f"Query: «{query_label}»",
            edgecolors="black", linewidths=1,
        )
        ax.annotate(
            f"Query: «{query_label}»",
            (query_emb[0], query_emb[1]),
            fontsize=8, color="red", fontweight="bold",
        )

    title_str = f"Title embedding space — UMAP 2D  ({n} titles, {TOTAL_DIM}-dim vectors)"
    if query_vec is not None:
        title_str += f"\nQuery: «{query_label}»"
    ax.set_title(title_str, fontsize=10)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=7, markerscale=1.5)
    fig.tight_layout()

    out_png = out_path.with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_png}")


# ──────────────────────────────────────────────────────────────────────────────
# 10.  Main
# ──────────────────────────────────────────────────────────────────────────────

# module-level set populated in main(); used by demo_search for the 🖼 flag
_visual_coverage: set[int] = set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build title vectors, index with FAISS, show similarity + UMAP.")
    parser.add_argument("--limit",   type=int, default=100, help="Number of titles to process (default: 100)")
    parser.add_argument("--rebuild", action="store_true",   help="Rebuild index even if cached files exist")
    args = parser.parse_args()

    # ── 1. Fetch titles ───────────────────────────────────────────────────
    print(f"Fetching up to {args.limit} titles from Postgres …")
    titles = fetch_titles(args.limit)
    if not titles:
        sys.exit("No titles found — check DATABASE_URL and that the titles table is populated.")
    if len(titles) < args.limit:
        print(f"  ⚠  Only {len(titles)} titles available (requested {args.limit}).")
    print(f"  Got {len(titles)} titles.")

    title_ids = [int(t["id"]) for t in titles]

    # ── 2. Fetch visual vectors ───────────────────────────────────────────
    print("Fetching visual vectors from Qdrant (siglip-768) …")
    visual_map = fetch_visual_vectors(title_ids)
    _visual_coverage.update(visual_map.keys())

    # ── 3. Fetch cast & crew ──────────────────────────────────────────────
    print("Fetching cast & crew from persons / title_persons …")
    persons_map = fetch_title_persons(title_ids)
    has_persons = sum(1 for tid in title_ids if tid in persons_map)
    print(f"  {has_persons}/{len(titles)} titles have cast/crew data.")

    # ── 4. Extract audio (on --rebuild) then encode with Whisper ─────────
    cuid2s = [t["cuid2"] for t in titles]

    if args.rebuild:
        from extract_audio import extract_missing_audio
        print("Extracting audio (music videos via yt-dlp) …")
        extract_missing_audio(titles, out_dir=AUDIO_DIR)

    wav_available = sum(1 for c in cuid2s if (AUDIO_DIR / f"{c}.wav").exists())
    if wav_available == 0:
        print("  No WAV files available — audio modality will be zeros.")
        audio_map: dict[str, np.ndarray] = {}
    else:
        print(f"Building audio embeddings (CLAP {CLAP_MODEL_ID}) …")
        print(f"  {wav_available}/{len(titles)} WAV files available.")
        audio_map = build_audio_map(cuid2s)

    # ── 5. Build or load vectors ──────────────────────────────────────────
    cache_valid = (
        INDEX_PATH.exists()
        and META_PATH.exists()
        and VECTORS_PATH.exists()
        and not args.rebuild
    )

    if cache_valid:
        print(f"Loading cached vectors from {VECTORS_PATH} …")
        vectors = np.load(VECTORS_PATH)
        print(f"Loading cached FAISS index from {INDEX_PATH} …")
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH) as f:
            saved_meta = json.load(f)
        # Reconcile: use DB titles but warn if counts differ
        if len(saved_meta) != len(titles):
            print(f"  ⚠  Cache has {len(saved_meta)} entries but DB returned {len(titles)}. "
                  "Run with --rebuild to refresh.")
    else:
        # Build from scratch
        print(f"\nBuilding {len(titles)} title vectors  "
              f"[text={TEXT_DIM}d · visual={VISUAL_DIM}d · audio={AUDIO_DIM}d · meta={META_DIM}d = {TOTAL_DIM}d] …")
        vectors_list: list[np.ndarray] = []
        for row in tqdm(titles, unit="title"):
            vectors_list.append(build_title_vector(row, visual_map, persons_map, audio_map))
        vectors = np.stack(vectors_list)   # (N, 1693)

        # Verify all L2-norms ≈ 1
        norms = np.linalg.norm(vectors, axis=1)
        print(f"  Vector norms — min: {norms.min():.4f}  max: {norms.max():.4f}  mean: {norms.mean():.4f}")

        print(f"Building FAISS IndexFlatIP ({TOTAL_DIM}-dim) …")
        index = build_index(vectors)

        # Persist
        np.save(VECTORS_PATH, vectors)
        faiss.write_index(index, str(INDEX_PATH))
        with open(META_PATH, "w") as f:
            json.dump(
                [
                    {
                        "id":    int(t["id"]),
                        "title": t["title"],
                        "type":  t.get("type"),
                        "year":  t.get("year"),
                    }
                    for t in titles
                ],
                f, indent=2, default=str,
            )
        print(f"  Saved → {VECTORS_PATH}  {INDEX_PATH}  {META_PATH}")

    # ── 6. Modality coverage report ───────────────────────────────────────
    covered     = len(visual_map)
    has_audio   = len(audio_map)
    print("\n── Modality coverage ─────────────────────────────────────────")
    has_desc    = sum(1 for t in titles if t.get("description"))
    has_rating  = sum(1 for t in titles if (t.get("ratings") or {}) and
                      (t.get("ratings") if isinstance(t.get("ratings"), dict) else
                       json.loads(t["ratings"]) if isinstance(t.get("ratings"), str) else {}
                      ).get("imdb"))
    n = len(titles)
    print(f"  Text (title)       : {n}/{n}  (100%)")
    print(f"  Text (description) : {has_desc}/{n}")
    print(f"  Cast & crew        : {has_persons}/{n}")
    print(f"  Visual embeddings  : {covered}/{n}  ({100*covered//n}%)")
    print(f"  Audio embeddings   : {has_audio}/{n}  ({100*has_audio//n}%)")
    print(f"  IMDB rating        : {has_rating}/{n}")

    # ── 7. Build sub-indices for dual-encoder search ──────────────────────
    print("Building text, visual, and audio sub-indices …")
    text_index, visual_index, audio_index = build_sub_indices(vectors)

    # ── 8. Interactive similarity search ─────────────────────────────────
    interactive_search(text_index, visual_index, audio_index, titles, vectors, k=5)


if __name__ == "__main__":
    main()
