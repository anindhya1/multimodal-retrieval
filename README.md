# Genery Demo — Multimodal Title Embeddings & Retrieval

A multimodal vector indexing and similarity search system for movie/TV titles. Each title is represented as a fused 1,693-dimensional vector built from text, visual, audio, and structured metadata, then indexed in FAISS for fast cosine-equivalent nearest-neighbour search.

---

## Vector Layout

```
[    0 :  384 ]  text     — mean-pooled all-MiniLM-L6-v2 on title + description + genres + keywords + cast
[  384 : 1152 ]  visual   — mean of SigLIP2 ViT-B (siglip-768) vectors from Qdrant (zeros if none)
[ 1152 : 1664 ]  audio    — mean-pooled Whisper base encoder hidden states (zeros if no WAV)
[ 1664 : 1693 ]  metadata — year · popularity · imdb_rating · is_adult
                            type one-hot (×5) · genre multi-hot (×20)
```

Each sub-vector is L2-normalised individually before concatenation. The final fused vector is L2-normalised again, making inner-product search equivalent to cosine similarity.

---

## Files

| File | Role |
|---|---|
| `title_vectors.py` | Main script — builds vectors, FAISS index, interactive search + per-query UMAP |
| `db.py` | Database clients (PostgreSQL via SQLAlchemy, Qdrant) |
| `extract_audio.py` | Extracts 16kHz mono WAV files from HLS streams via ffmpeg (called on `--rebuild`) |

---

## Setup

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and on `PATH` (`brew install ffmpeg` on macOS)
- A running PostgreSQL database populated with the Genery schema
- A running Qdrant instance with a `media_items` collection containing `siglip-768` vectors

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core packages used:

| Package | Purpose |
|---|---|
| `sqlalchemy` | PostgreSQL queries |
| `qdrant-client` | Fetch SigLIP2 visual vectors |
| `faiss-cpu` | Vector indexing and search |
| `transformers` | all-MiniLM-L6-v2 text encoder, SigLIP2 text encoder |
| `torch` | Model inference |
| `openai-whisper` | Audio encoder (hidden states, not transcription) |
| `llama-index-core` | `SentenceSplitter` for text chunking |
| `umap-learn` | Dimensionality reduction for per-query visualisation (optional) |
| `plotly` | Interactive 3D UMAP HTML output (optional, required alongside umap-learn) |
| `python-dotenv` | Environment variable loading |
| `tqdm` | Progress bars |

### Environment variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
QDRANT_ENDPOINT=https://your-qdrant-instance
QDRANT_API_KEY=your-api-key

# Optional — override default Qdrant collection names
QDRANT_STILLS_COLLECTION=media_items

# Optional — CDN base for b2:// → HTTP URL conversion (used by extract_audio.py)
CDN_BASE=https://cdn.genery.online/file/
```

---

## Usage

### Step 1 — Extract audio (optional, enables the audio modality)

`extract_audio.py` fetches each title's best video clip from Postgres, downloads its HLS stream, and extracts a 16kHz mono WAV file into `./audio/`. Titles without a WAV file receive a zero audio sub-vector.

```bash
python extract_audio.py                   # top 100 titles
python extract_audio.py --limit 500
python extract_audio.py --output my_dir   # custom output directory
python extract_audio.py --diagnose        # inspect raw file_path JSON from DB and exit
```

### Step 2 — Build vectors, index, and search

```bash
python title_vectors.py                   # top 100 titles (uses cache if present)
python title_vectors.py --limit 300
python title_vectors.py --rebuild         # force rebuild and re-extract missing audio
```

**What it does:**

1. Fetches titles from PostgreSQL ordered by popularity
2. Fetches SigLIP2 visual vectors from Qdrant (top 50 stills per title, averaged then re-normalised to unit length)
3. Fetches cast & crew from `title_persons` / `persons` (up to 20 per title, directors first)
4. Extracts and encodes audio with the Whisper encoder (if WAV files exist in `./audio/`)
5. Encodes text with all-MiniLM-L6-v2 (chunked via LlamaIndex `SentenceSplitter`, 256-token chunks with 32-token overlap)
6. Encodes structured metadata (year, popularity, IMDB rating, type, genres)
7. L2-normalises each modality vector individually, concatenates them → 1,693-dim vector, then L2-normalises the fused result
8. Builds a FAISS `IndexFlatIP` and persists it to disk
9. Launches an interactive dual-encoder similarity search loop

**Cached outputs** (`title_index.faiss`, `title_meta.json`, `title_vectors.npy`) are reused on subsequent runs unless `--rebuild` is passed.

---

## Pipeline Architecture

```
PostgreSQL titles
       │
       ├── text fields ──→ all-MiniLM-L6-v2 ──→ 384-dim ──→ L2-normalise ──┐
       │    (title, description,                                              │
       │     genres, keywords, cast)                                         │
       │                                                                     │
       ├── media_items ──→ Qdrant (siglip-768) ──→ mean+renorm ──→ 768-dim ──→ L2-normalise ──┤ concat → 1693-dim → L2-normalise
       │    (top 50 stills by aesthetic_score)                               │
       │                                                                     │
       ├── audio/<cuid2>.wav ──→ Whisper encoder ──→ 512-dim ──→ L2-normalise ──┤
       │    (populated by extract_audio.py)                                  │
       │                                                                     │
       └── structured fields ──→ normalise / one-hot / multi-hot ──→ 29-dim ──→ L2-normalise ──┘
            (year, popularity, IMDB rating, type, genres)
                                                                             │
                                                                       FAISS IndexFlatIP
                                                                  (inner product = cosine sim)
```

---

## Interactive Search

After building the index, `title_vectors.py` enters an interactive prompt that queries both the text sub-index (all-MiniLM) and the visual sub-index (SigLIP2) and combines scores:

```
combined_score = α × text_score + (1 − α) × visual_score
```

Default `α = 0.5`. Available commands:

```
Search: dark gritty crime thriller    # free-text query
Search: list                          # browse all indexed titles
Search: alpha 0.7                     # set text weight to 0.7, visual to 0.3
Search: quit
```

Results display the combined score, individual text and visual sub-scores, and a flag for titles that have Qdrant image vectors.

After each query, a 3D interactive UMAP visualisation is generated and saved as `umap_query_<query>.html`. The corpus titles are plotted and coloured by title type; the query is placed in the same space and shown as a red diamond, so you can see which region of the embedding space the query lands in relative to the titles.

---

## Database Schema (relevant tables)

| Table | Relevant columns |
|---|---|
| `titles` | `id`, `cuid2`, `title`, `type`, `description`, `year`, `popularity`, `is_adult`, `genres`, `keywords`, `ratings` |
| `media_items` | `title_id`, `point_id` (→ Qdrant), `file_path` (JSON with `scene.hls`), `aesthetic_score` |
| `title_persons` | `title_id`, `person_id`, `role`, `role_description` |
| `persons` | `id`, `name` |

---

## Generated Files

| File | Description |
|---|---|
| `audio/<cuid2>.wav` | 16kHz mono WAV files produced by `extract_audio.py` |
| `title_vectors.npy` | Raw `(N, 1693)` float32 vector matrix |
| `title_index.faiss` | Persisted FAISS `IndexFlatIP` |
| `title_meta.json` | Title metadata aligned to FAISS index positions |
| `umap_query_<query>.html` | Per-query interactive 3D UMAP plot (corpus + query point) |
