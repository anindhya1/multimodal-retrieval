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

All sub-vectors are L2-normalised before concatenation; the final fused vector is L2-normalised again, making inner-product search equivalent to cosine similarity.

---

## Files

| File | Role |
|---|---|
| `title_vectors.py` | Main script — builds vectors, FAISS index, UMAP plot, interactive search |
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
| `umap-learn` | 2D visualisation (optional) |
| `python-dotenv` | Environment variable loading |
| `tqdm` | Progress bars |
| `matplotlib` | UMAP plot rendering |

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
2. Fetches SigLIP2 visual vectors from Qdrant (top 50 stills per title, averaged)
3. Fetches cast & crew from `title_persons` / `persons` (up to 20 per title, directors first)
4. Extracts and encodes audio with the Whisper encoder (if WAV files exist in `./audio/`)
5. Encodes text with all-MiniLM-L6-v2 (chunked via LlamaIndex `SentenceSplitter`, 256-token chunks with 32-token overlap)
6. Encodes structured metadata (year, popularity, IMDB rating, type, genres)
7. Fuses all modalities → 1,693-dim L2-normalised vector per title
8. Builds a FAISS `IndexFlatIP` and persists it to disk
9. Renders a UMAP 2D scatter plot coloured by title type → `umap_titles.png`
10. Launches an interactive dual-encoder similarity search loop

**Cached outputs** (`title_index.faiss`, `title_meta.json`, `title_vectors.npy`) are reused on subsequent runs unless `--rebuild` is passed.

---

## Pipeline Architecture

```
PostgreSQL titles
       │
       ├── text fields ──→ all-MiniLM-L6-v2 ──→ 384-dim text vector
       │    (title, description,
       │     genres, keywords, cast)
       │
       ├── media_items ──→ Qdrant (siglip-768) ──→ mean pool ──→ 768-dim visual vector
       │    (top 50 stills by aesthetic_score)
       │
       ├── audio/<cuid2>.wav ──→ Whisper encoder ──→ mean pool ──→ 512-dim audio vector
       │    (populated by extract_audio.py)
       │
       └── structured fields ──→ normalise / one-hot / multi-hot ──→ 29-dim metadata vector
            (year, popularity, IMDB rating, type, genres)

All four sub-vectors concatenated → 1,693-dim fused vector → L2-normalise
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
| `umap_titles.png` | 2D UMAP scatter plot of the embedding space |
