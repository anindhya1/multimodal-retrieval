"""
Multimodal Movie Retrieval Pipeline — Path 1: SigLIP2 + CLAP (Late Fusion)
============================================================================

Architecture:
                                                          PROJECTION
    Keyframes  → SigLIP2 Vision Encoder → 1152-dim vec ──→ ┐
    Trailer    → SigLIP2 Vision Encoder → 1152-dim vec ──→ ┤ Project to
    IMDB text  → SigLIP2 Text Encoder   → 1152-dim vec ──→ ┤ common dim  → Late Fusion → FAISS
    Audio      → CLAP Audio Encoder     →  512-dim vec ──→ ┘ (e.g. 512)

Problem: SigLIP2 (1152-dim) and CLAP (512-dim) live in DIFFERENT vector spaces
          with DIFFERENT dimensionalities. We must align before fusion.

Solution: Project all vectors to a common dimensionality via learned linear
          projections (or simple PCA/truncation for a no-training baseline).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import numpy as np
import faiss
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
TRAILER_DIR  = PROJECT_DIR / "trailer_data"
META_FILE    = TRAILER_DIR / "trailers_metadata.json"
CLIPS_DIR    = TRAILER_DIR / "clips"
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoModel, AutoProcessor, ClapModel, ClapProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class MovieData:
    """Raw multimodal data for a single movie."""
    movie_id: str
    title: str
    year: int
    genres: list[str]
    video_path: str = ""                            # clip.mp4 — encoded by X-CLIP
    image_paths: list[str] = field(default_factory=list)   # frame_*.jpg + title_frame.jpg — encoded by SigLIP2
    text_chunks: list[str] = field(default_factory=list)   # encoded by SigLIP2 text
    audio_paths: list[str] = field(default_factory=list)   # encoded by CLAP


@dataclass
class MovieEmbedding:
    """Processed embedding for a single movie."""
    movie_id: str
    title: str
    year: int
    genres: list[str]
    fused_vector: np.ndarray                        # single movie-level vector (common dim)
    video_vector: Optional[np.ndarray] = None       # X-CLIP video embedding
    image_vectors: Optional[np.ndarray] = None      # SigLIP2 image embeddings
    text_vectors: Optional[np.ndarray] = None       # SigLIP2 text embeddings
    audio_vectors: Optional[np.ndarray] = None      # CLAP audio embeddings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENCODER 1: SigLIP2 (Text + Images)
# Handles: keyframes, trailer frames, IMDB text
# Output: 1152-dim vectors (for So400m variant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SigLIP2Encoder:
    """
    SigLIP2 dual-tower encoder.
    - Vision tower: image → 1152-dim vector
    - Text tower:   text  → 1152-dim vector
    Both share the same embedding space.
    """

    def __init__(self, model_name: str = "google/siglip2-so400m-patch16-naflex"):
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            raise ImportError("pip install transformers torch")

        print(f"  Loading SigLIP2: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embed_dim = self.model.config.vision_config.hidden_size
        print(f"  SigLIP2 embedding dim: {self.embed_dim}")

    def encode_images(self, images: list, batch_size: int = 8) -> np.ndarray:
        all_vectors = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                features = self.model.get_image_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
                features = F.normalize(features, dim=-1)
                all_vectors.append(features.cpu().numpy())
        return np.vstack(all_vectors)

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        all_vectors = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.processor(
                    text=batch, return_tensors="pt",
                    padding="max_length", max_length=64
                )
                features = self.model.get_text_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
                features = F.normalize(features, dim=-1)
                all_vectors.append(features.cpu().numpy())
        return np.vstack(all_vectors)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENCODER 2: CLAP (Audio)
# Handles: dialogue clips, soundtrack, sound effects
# Output: 512-dim vectors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CLAPEncoder:
    """
    CLAP (Contrastive Language-Audio Pretraining) encoder.
    - Audio tower: waveform → 512-dim vector
    - Text tower:  text    → 512-dim vector
    
    Uses HuggingFace transformers ClapModel.
    Audio must be 48kHz sample rate.
    """

    SAMPLE_RATE = 48000

    def __init__(self, model_name: str = "laion/clap-htsat-unfused"):
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            raise ImportError("pip install transformers torch")

        print(f"  Loading CLAP: {model_name}")
        self.model = ClapModel.from_pretrained(model_name)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embed_dim = self.model.config.projection_dim  # 512
        print(f"  CLAP embedding dim: {self.embed_dim}")

    def encode_audio_from_file(self, audio_path: str) -> np.ndarray:
        """Load audio file and encode → normalized 512-dim vector."""
        if not HAS_LIBROSA:
            raise ImportError("pip install librosa")

        # CLAP expects 48kHz audio
        waveform, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
        return self.encode_audio_waveform(waveform)

    def encode_audio_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Encode raw waveform array → normalized 512-dim vector."""
        with torch.no_grad():
            inputs = self.processor(
                audio=waveform, sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            )
            features = self.model.get_audio_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = F.normalize(features, dim=-1)
        return features.squeeze(0).cpu().numpy()

    def encode_audio_files(self, audio_paths: list[str]) -> np.ndarray:
        """Encode multiple audio files → array of 512-dim vectors."""
        return np.stack([self.encode_audio_from_file(p) for p in audio_paths])

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text through CLAP's text tower (for audio-text queries)."""
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            features = self.model.get_text_features(**inputs)
            features = F.normalize(features, dim=-1)
        return features.squeeze(0).cpu().numpy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENCODER 3: X-CLIP (Video)
# Handles: clip.mp4 → temporal video embedding
# Output: 512-dim vectors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class XCLIPEncoder:
    """
    X-CLIP video encoder — encodes clip.mp4 directly as a temporal sequence.
    Extracts SAMPLE_FRAMES frames at scene-cut boundaries (detected via HSV
    histogram correlation), passes them through the video transformer, and
    returns a single 512-dim embedding.

    Unlike image-based approaches, X-CLIP captures motion, pacing,
    and scene transitions — essential for trailer content.
    """

    SAMPLE_FRAMES   = 8     # X-CLIP standard: fixed frame count expected by the model
    SCENE_THRESHOLD = 0.40  # HSV histogram CORREL below this → scene cut

    def __init__(self, model_name: str = "microsoft/xclip-base-patch32"):
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            raise ImportError("pip install transformers torch")
        if not HAS_CV2:
            raise ImportError("pip install opencv-python")

        print(f"  Loading X-CLIP: {model_name}")
        from transformers import XCLIPModel, XCLIPProcessor
        self.model     = XCLIPModel.from_pretrained(model_name)
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embed_dim = self.model.config.projection_dim  # 512
        print(f"  X-CLIP embedding dim: {self.embed_dim}")

    def _sample_frames(self, video_path: str) -> list[list]:
        """
        For each detected scene, extract exactly SAMPLE_FRAMES evenly spaced frames.
        Returns a list of per-scene frame lists: [[scene0_frames], [scene1_frames], ...]

        Steps:
          1. Scan the video at ~2 fps, computing a 2-D HSV histogram per sample.
          2. A scene cut is declared when the CORREL score between consecutive
             histograms drops below SCENE_THRESHOLD.
          3. Each consecutive pair of scene-boundary indices defines a scene's
             frame range: [scene_idxs[i], scene_idxs[i+1] - 1].
          4. SAMPLE_FRAMES frames are sampled evenly within each scene's range.
          5. If fewer than 2 scenes are detected (very short or static video),
             the entire clip is treated as one scene.
        """
        from PIL import Image

        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
        if total == 0:
            cap.release()
            return []

        # ── Pass 1: detect scene-boundary frame indices ──────────────────────
        stride     = max(1, int(fps / 2))   # sample ~2 frames per second
        scene_idxs = []                      # frame index of each scene start
        prev_hist  = None

        frame_idx = 0
        while frame_idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            is_cut = (prev_hist is None) or (
                cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < self.SCENE_THRESHOLD
            )
            if is_cut:
                scene_idxs.append(frame_idx)

            prev_hist  = hist
            frame_idx += stride

        # ── Fallback: treat entire clip as one scene ──────────────────────────
        if len(scene_idxs) < 2:
            scene_idxs = [0, total - 1]

        # ── Pass 2: sample SAMPLE_FRAMES evenly within each scene's range ────
        scenes = []
        for i in range(len(scene_idxs)):
            start = scene_idxs[i]
            end   = scene_idxs[i + 1] - 1 if i + 1 < len(scene_idxs) else total - 1
            if end <= start:
                end = start + 1

            target_idxs = np.linspace(start, end, self.SAMPLE_FRAMES, dtype=int)
            frames = []
            for idx in target_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))

            if len(frames) == self.SAMPLE_FRAMES:
                scenes.append(frames)

        cap.release()
        return scenes

    def _encode_scenes(self, scenes: list[list]) -> np.ndarray:
        """
        Encode pre-extracted scenes through X-CLIP → mean-pooled clip vector.
        Each scene must contain exactly SAMPLE_FRAMES PIL Images.
        """
        image_proc = getattr(self.processor, "image_processor", None) or \
                     getattr(self.processor, "feature_extractor", None)

        scene_vectors = []
        with torch.no_grad():
            for frames in scenes:
                # get_video_features expects [batch, num_frames, C, H, W]
                pixel_values = image_proc(images=frames, return_tensors="pt").pixel_values
                pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:]).unsqueeze(0)
                features = self.model.get_video_features(pixel_values=pixel_values)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
                features = F.normalize(features, dim=-1)
                scene_vectors.append(features.squeeze(0).cpu().numpy())

        # Mean-pool scene vectors → single clip-level vector, then re-normalize
        clip_vector = np.stack(scene_vectors).mean(axis=0)
        clip_vector = clip_vector / (np.linalg.norm(clip_vector) + 1e-8)
        return clip_vector

    def encode_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract scenes from video_path and encode → normalized 512-dim vector.
        Returns None if video cannot be read.
        """
        scenes = self._sample_frames(video_path)
        if not scenes:
            print(f"    WARNING: could not read frames from {video_path}")
            return None
        return self._encode_scenes(scenes)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIMENSION ALIGNMENT
# SigLIP2 = 1152-dim, CLAP = 512-dim, X-CLIP = 512-dim
# X-CLIP and CLAP need no projection (already 512-dim)
# We must project both to a common space before
# late fusion can combine them.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DimensionAligner:
    """
    Projects vectors from different encoders into a common dimensionality.
    
    Two modes:
        1. "linear" (default): Random linear projection (no training needed).
           Good enough for late fusion where the fusion step absorbs alignment.
        2. "learned": Trained linear layer (requires paired data).
           Better if you need cross-modal similarity at the individual vector level.
    
    Why this is needed:
        SigLIP2 outputs 1152-dim vectors in Space A.
        CLAP outputs 512-dim vectors in Space B.
        You cannot average a 1152-dim vector with a 512-dim vector.
        Even if they were the same size, they'd be in different spaces.
        
        The projection gives them the same size. L2-normalization after
        projection puts them on the same scale. The late fusion weights
        then control how much each modality contributes.
    """

    def __init__(self, source_dims: dict[str, int], target_dim: int = 512):
        """
        Args:
            source_dims: mapping of modality name → original embedding dim
                         e.g. {"siglip2": 1152, "clap": 512}
            target_dim:  common dimension to project to
        """
        self.target_dim = target_dim
        self.projections: dict[str, np.ndarray] = {}

        np.random.seed(42)
        for name, dim in source_dims.items():
            if dim == target_dim:
                # No projection needed — identity
                self.projections[name] = None
            else:
                # Random orthogonal projection (preserves distances better than random)
                random_matrix = np.random.randn(dim, target_dim).astype(np.float32)
                # QR decomposition → orthogonal columns
                q, _ = np.linalg.qr(random_matrix)
                self.projections[name] = q[:, :target_dim]

        print(f"  Dimension aligner: target_dim={target_dim}")
        for name, dim in source_dims.items():
            if self.projections[name] is None:
                print(f"    {name}: {dim} → {target_dim} (identity, no projection)")
            else:
                print(f"    {name}: {dim} → {target_dim} (orthogonal projection)")

    def project(self, vectors: np.ndarray, source: str) -> np.ndarray:
        """
        Project vectors from a source encoder to the common dimension.
        Re-normalizes after projection.
        """
        proj = self.projections[source]
        if proj is None:
            return vectors  # already correct dimension

        projected = vectors @ proj  # (N, source_dim) @ (source_dim, target_dim) → (N, target_dim)
        # Re-normalize after projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8
        return projected / norms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LATE FUSION AGGREGATOR
# Now handles projected vectors from both encoders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LateFusionAggregator:
    """
    Combines projected vectors from SigLIP2 + CLAP into one movie vector.
    
    Steps:
        1. Mean-pool within each modality
        2. L2-normalize each modality summary
        3. Weighted average across modalities
        4. Final L2-normalization
    """

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or {
            "video": 0.35,   # X-CLIP: full temporal video signal
            "image": 0.15,   # SigLIP2: keyframes + title card
            "text":  0.30,   # SigLIP2: whisper transcript + OCR
            "audio": 0.20,   # CLAP: instrumental + vocals stems
        }

    def aggregate_modality(self, vectors: np.ndarray) -> np.ndarray:
        """Mean-pool → L2-normalize."""
        if vectors is None or len(vectors) == 0:
            return None
        mean_vec = vectors.mean(axis=0)
        return mean_vec / (np.linalg.norm(mean_vec) + 1e-8)

    def fuse(
        self,
        video_vector: Optional[np.ndarray] = None,
        image_vectors: Optional[np.ndarray] = None,
        text_vectors: Optional[np.ndarray] = None,
        audio_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fuse all modalities into a single movie-level vector."""
        components = {}

        if video_vector is not None:
            # video_vector is already a single 512-dim vector from X-CLIP
            v = video_vector.reshape(1, -1) if video_vector.ndim == 1 else video_vector
            components["video"] = self.aggregate_modality(v)
        if image_vectors is not None and len(image_vectors) > 0:
            components["image"] = self.aggregate_modality(image_vectors)
        if text_vectors is not None and len(text_vectors) > 0:
            components["text"] = self.aggregate_modality(text_vectors)
        if audio_vectors is not None and len(audio_vectors) > 0:
            components["audio"] = self.aggregate_modality(audio_vectors)

        if not components:
            raise ValueError("At least one modality must have vectors")

        # Redistribute weights for available modalities
        available_weight = sum(self.weights[k] for k in components)
        norm_weights = {k: self.weights[k] / available_weight for k in components}

        # Weighted combination
        fused = sum(norm_weights[k] * components[k] for k in components)

        # Final L2 normalization
        return fused / (np.linalg.norm(fused) + 1e-8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FAISS INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MovieVectorIndex:
    """FAISS index for movie-level vectors with cosine similarity."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.index = faiss.IndexFlatIP(embed_dim)
        self.movie_metadata: list[dict] = []

    def add_movie(self, embedding: MovieEmbedding):
        vec = embedding.fused_vector.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        self.movie_metadata.append({
            "movie_id": embedding.movie_id,
            "title": embedding.title,
            "year": embedding.year,
            "genres": embedding.genres,
        })

    def add_movies(self, embeddings: list[MovieEmbedding]):
        for emb in embeddings:
            self.add_movie(emb)
        print(f"  FAISS index: {self.index.ntotal} movies indexed")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = self.movie_metadata[idx].copy()
            result["similarity_score"] = float(score)
            results.append(result)
        return results

    def get_all_vectors(self) -> np.ndarray:
        return faiss.rev_swig_ptr(
            self.index.get_xb(), self.index.ntotal * self.embed_dim
        ).reshape(self.index.ntotal, self.embed_dim).copy()

    def save(self, path: str):
        """
        Save FAISS index + metadata to disk.
        Writes two files:
            {path}.faiss  — the FAISS binary index
            {path}.json   — movie metadata list (titles, years, genres, ids)
        """
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.json", "w") as f:
            json.dump({
                "embed_dim": self.embed_dim,
                "movie_metadata": self.movie_metadata,
            }, f, indent=2)
        print(f"  Index saved → {path}.faiss + {path}.json")

    @classmethod
    def load(cls, path: str) -> "MovieVectorIndex":
        """
        Load a previously saved index from disk.
        Usage:
            index = MovieVectorIndex.load("trailer_data/faiss_index")
        """
        index_path = f"{path}.faiss"
        meta_path  = f"{path}.json"

        if not Path(index_path).exists() or not Path(meta_path).exists():
            raise FileNotFoundError(f"Index files not found at {path}(.faiss/.json)")

        with open(meta_path) as f:
            data = json.load(f)

        instance = cls(embed_dim=data["embed_dim"])
        instance.index = faiss.read_index(index_path)
        instance.movie_metadata = data["movie_metadata"]
        print(f"  Index loaded ← {index_path}  ({instance.index.ntotal} movies)")
        return instance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UMAP VISUALIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MovieSpaceVisualizer:
    """UMAP projection + matplotlib plot of the movie embedding space."""

    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1):
        if not HAS_UMAP:
            raise ImportError("pip install umap-learn")
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist,
            n_components=3, metric="cosine", random_state=42,
        )

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(vectors)

    def plot(
        self, vectors_3d: np.ndarray, metadata: list[dict],
        query_point_3d: Optional[np.ndarray] = None,
        query_label: str = "Query",
        save_path: Optional[str] = None,
        title: str = "Movie Embedding Space (UMAP Projection)",
        highlight_ids: Optional[list] = None,
    ):
        if not HAS_PLOTLY:
            raise ImportError("pip install plotly")

        highlight_ids = set(highlight_ids or [])
        genre_set = sorted({m["genres"][0] for m in metadata if m["genres"]})
        # Map genres to a discrete Plotly color sequence
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        genre_colors = {g: palette[i % len(palette)] for i, g in enumerate(genre_set)}

        traces = []

        # ── Corpus movies: two traces per genre (regular + retrieved) ──
        for genre in genre_set:
            all_indices = [
                i for i, m in enumerate(metadata)
                if (m["genres"][0] if m["genres"] else "Unknown") == genre
            ]
            regular   = [i for i in all_indices if metadata[i].get("movie_id") not in highlight_ids]
            retrieved = [i for i in all_indices if metadata[i].get("movie_id") in highlight_ids]

            if regular:
                traces.append(go.Scatter3d(
                    x=vectors_3d[regular, 0],
                    y=vectors_3d[regular, 1],
                    z=vectors_3d[regular, 2],
                    mode="markers+text",
                    name=genre,
                    text=[f"{metadata[i]['title']} ({metadata[i]['year']})" for i in regular],
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(size=6, color=genre_colors[genre], opacity=0.7),
                ))

            if retrieved:
                traces.append(go.Scatter3d(
                    x=vectors_3d[retrieved, 0],
                    y=vectors_3d[retrieved, 1],
                    z=vectors_3d[retrieved, 2],
                    mode="markers+text",
                    name=f"{genre} (retrieved)",
                    text=[f"<b>{metadata[i]['title']} ({metadata[i]['year']})</b>" for i in retrieved],
                    textposition="top center",
                    textfont=dict(size=10),
                    marker=dict(
                        size=10, color=genre_colors[genre], opacity=0.9,
                        line=dict(color="gold", width=2),
                    ),
                ))

        # ── Query point ──
        if query_point_3d is not None:
            traces.append(go.Scatter3d(
                x=[query_point_3d[0]],
                y=[query_point_3d[1]],
                z=[query_point_3d[2]],
                mode="markers+text",
                name="Query",
                text=[query_label],
                textposition="top center",
                textfont=dict(size=11, color="red"),
                marker=dict(size=14, color="red", symbol="diamond",
                            line=dict(color="black", width=2)),
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="UMAP-1",
                yaxis_title="UMAP-2",
                zaxis_title="UMAP-3",
            ),
            legend_title="Primary Genre",
            margin=dict(l=0, r=0, b=0, t=40),
        )

        if save_path:
            fig.write_html(save_path)
            print(f"  Saved: {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultimodalMovieRetrieval:
    """
    Complete pipeline with X-CLIP + SigLIP2 + CLAP:

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                         INGESTION PIPELINE                              │
    │                                                                         │
    │  clip.mp4  ──→ X-CLIP Video   ──→  512-d ──→ (identity)→ 512-d ──┐     │
    │  Text      ──→ SigLIP2 Text   ──→ 1152-d ──→ Project  ──→ 512-d ──┤Fuse │→ FAISS
    │  Audio     ──→ CLAP Audio     ──→  512-d ──→ (identity)→ 512-d ──┘     │
    └──────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                          QUERY PIPELINE                                 │
    │                                                                         │
    │  Text query ──→ SigLIP2 Text ──→ 1152-d ──→ Project ──→ 512-d          │
    │                                                  ↓                      │
    │                                            FAISS search                 │
    │                                                  ↓                      │
    │                                           Top-K movies                  │
    └──────────────────────────────────────────────────────────────────────────┘
    """

    SIGLIP2_DIM = 1152  # So400m — text only
    XCLIP_DIM   = 512   # X-CLIP video encoder
    CLAP_DIM    = 512   # CLAP audio encoder
    COMMON_DIM  = 512   # common projection target

    def __init__(
        self,
        siglip2_model: str = "google/siglip2-so400m-patch16-naflex",
        xclip_model:   str = "microsoft/xclip-base-patch32",
        clap_model:    str = "laion/clap-htsat-unfused",
        use_real_models: bool = False,
    ):
        print("Initializing Multimodal Movie Retrieval Pipeline")
        print("=" * 55)

        if use_real_models:
            self.siglip2 = SigLIP2Encoder(siglip2_model)
            self.xclip   = XCLIPEncoder(xclip_model)
            self.clap    = CLAPEncoder(clap_model)
            self.SIGLIP2_DIM = self.siglip2.embed_dim
            self.XCLIP_DIM   = self.xclip.embed_dim
            self.CLAP_DIM    = self.clap.embed_dim
        else:
            self.siglip2 = None
            self.xclip   = None
            self.clap    = None

        # Only SigLIP2 (1152-d) needs projection; X-CLIP and CLAP are already 512-d
        self.aligner = DimensionAligner(
            source_dims={
                "siglip2": self.SIGLIP2_DIM,
                "xclip":   self.XCLIP_DIM,
                "clap":    self.CLAP_DIM,
            },
            target_dim=self.COMMON_DIM,
        )

        self.aggregator = LateFusionAggregator()
        self.index      = MovieVectorIndex(self.COMMON_DIM)
        self.visualizer = MovieSpaceVisualizer(n_neighbors=5, min_dist=0.3) if HAS_UMAP else None
        self.embeddings: list[MovieEmbedding] = []
        print("=" * 55)
        print()

    def ingest_movie(self, movie: MovieData) -> MovieEmbedding:
        """Full ingestion: encode → project → fuse → index."""
        print(f"  Ingesting: {movie.title} ({movie.year})")

        proj_video_vec  = None
        proj_image_vecs = None
        proj_text_vecs  = None
        proj_audio_vecs = None

        # ── X-CLIP: extract scenes once, encode for video + reuse frames for SigLIP2 ──
        scene_frames = None  # flat list of PIL Images shared with SigLIP2 below
        if movie.video_path and self.xclip:
            try:
                scenes = self.xclip._sample_frames(movie.video_path)
                if scenes:
                    raw = self.xclip._encode_scenes(scenes)
                    proj_video_vec = self.aligner.project(raw.reshape(1, -1), "xclip")[0]
                    scene_frames = [frame for scene in scenes for frame in scene]
            except Exception as e:
                print(f"    WARNING: video encoding failed ({e}), skipping video modality")

        # ── SigLIP2: encode scene frames (from video) or fallback to image_paths ──
        if self.siglip2:
            try:
                if scene_frames:
                    images = scene_frames
                else:
                    from PIL import Image as PILImage
                    images = [PILImage.open(p).convert("RGB") for p in movie.image_paths]
                if images:
                    raw = self.siglip2.encode_images(images)
                    proj_image_vecs = self.aligner.project(raw, "siglip2")
            except Exception as e:
                print(f"    WARNING: image encoding failed ({e}), skipping image modality")

        # ── SigLIP2: encode text chunks, project 1152 → 512 ──
        if movie.text_chunks and self.siglip2:
            try:
                raw = self.siglip2.encode_texts(movie.text_chunks)
                proj_text_vecs = self.aligner.project(raw, "siglip2")
            except Exception as e:
                print(f"    WARNING: text encoding failed ({e}), skipping text modality")

        # ── CLAP: encode audio stems (already 512-d, identity projection) ──
        if movie.audio_paths and self.clap:
            try:
                raw = self.clap.encode_audio_files(movie.audio_paths)
                proj_audio_vecs = self.aligner.project(raw, "clap")
            except Exception as e:
                print(f"    WARNING: audio encoding failed ({e}), skipping audio modality")

        # ── Late fusion ──
        fused = self.aggregator.fuse(
            video_vector=proj_video_vec,
            image_vectors=proj_image_vecs,
            text_vectors=proj_text_vecs,
            audio_vectors=proj_audio_vecs,
        )

        embedding = MovieEmbedding(
            movie_id=movie.movie_id,
            title=movie.title,
            year=movie.year,
            genres=movie.genres,
            fused_vector=fused,
            video_vector=proj_video_vec,
            image_vectors=proj_image_vecs,
            text_vectors=proj_text_vecs,
            audio_vectors=proj_audio_vecs,
        )

        self.index.add_movie(embedding)
        self.embeddings.append(embedding)
        return embedding

    def save_index(self, path: str = None):
        """Save FAISS index + metadata. Defaults to trailer_data/faiss_index."""
        path = path or str(TRAILER_DIR / "faiss_index")
        self.index.save(path)

    def load_index(self, path: str = None):
        """Load a previously saved FAISS index. Replaces the current index."""
        path = path or str(TRAILER_DIR / "faiss_index")
        self.index = MovieVectorIndex.load(path)

    def search_by_text(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Text query:
            1. SigLIP2 text encoder → 1152-dim
            2. Project → 512-dim
            3. FAISS search
        """
        if self.siglip2:
            raw = self.siglip2.encode_text(query)
            query_vec = self.aligner.project(raw.reshape(1, -1), "siglip2")[0]
        else:
            raise RuntimeError("SigLIP2 not loaded (use_real_models=False)")
        return self.index.search(query_vec, top_k)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a text query → projected 512-dim vector (same path as search_by_text)."""
        if not self.siglip2:
            raise RuntimeError("SigLIP2 not loaded (use_real_models=False)")
        raw = self.siglip2.encode_text(query)
        return self.aligner.project(raw.reshape(1, -1), "siglip2")[0]

    def visualize(
        self, query_vec: Optional[np.ndarray] = None,
        query_label: str = "Query",
        save_path: str = "movie_embedding_space.html",
        highlight_ids: Optional[list] = None,
    ):
        """UMAP visualization of fused movie vectors."""
        if not self.visualizer:
            print("UMAP not available")
            return

        all_vectors = self.index.get_all_vectors()

        if query_vec is not None:
            combined = np.vstack([all_vectors, query_vec.reshape(1, -1)])
            projected = self.visualizer.fit_transform(combined)
            self.visualizer.plot(
                projected[:-1], self.index.movie_metadata,
                query_point_3d=projected[-1], query_label=query_label,
                save_path=save_path, highlight_ids=highlight_ids,
            )
        else:
            projected = self.visualizer.fit_transform(all_vectors)
            self.visualizer.plot(
                projected, self.index.movie_metadata,
                save_path=save_path, highlight_ids=highlight_ids,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADER — reads from trailer_data/
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_trailer_dataset() -> list[MovieData]:
    """
    Reads trailers_metadata.json and builds a list of MovieData objects,
    mapping our dataset's files to the pipeline's expected fields:

        video_path   ← video/clip.mp4                        (X-CLIP video encoder)
        text_chunks  ← [whisper_transcript, title_ocr]       (SigLIP2 text encoder)
        audio_paths  ← [audio/no_vocals.wav, audio/vocals.wav] (CLAP audio encoder)
    """
    if not META_FILE.exists():
        raise FileNotFoundError(f"Metadata not found: {META_FILE}")

    with open(META_FILE) as f:
        metadata = json.load(f)

    movies = []
    skipped = 0

    def _resolve(p):
        """Resolve a path that might be relative to PROJECT_DIR."""
        if not p:
            return None
        path = Path(p)
        if path.exists():
            return str(path)
        resolved = PROJECT_DIR / path
        if resolved.exists():
            return str(resolved)
        return None

    for entry in metadata:
        # Skip entries with no downloaded trailer
        if "error" in entry and not entry.get("local_paths"):
            skipped += 1
            continue

        lp    = entry.get("local_paths", {})
        title = entry.get("title", "Unknown")
        year  = entry.get("year", 0)
        kind  = entry.get("kind", "movie")   # "movie" or "tv"

        # ── Video path (X-CLIP) ───────────────────────────────────────────────
        video_path = _resolve(lp.get("clip")) or ""

        # ── Image paths (SigLIP2 vision) ──────────────────────────────────────
        image_paths = [
            r for r in [
                _resolve(lp.get("frame_0")), _resolve(lp.get("frame_1")),
                _resolve(lp.get("frame_2")), _resolve(entry.get("title_frame")),
            ] if r
        ]

        # ── Text chunks (SigLIP2) ─────────────────────────────────────────────
        text_chunks = []
        transcript = entry.get("whisper_transcript", "").strip()
        ocr        = entry.get("title_ocr", "").strip()
        if transcript:
            text_chunks.append(transcript)
        if ocr:
            text_chunks.append(ocr)

        # ── Audio paths (CLAP) ────────────────────────────────────────────────
        # Both stems fed to CLAP:
        #   no_vocals.wav → captures music/mood/soundtrack
        #   vocals.wav    → captures speech prosody, tone, pacing
        audio_paths = []
        for key in ("no_vocals", "vocals"):
            r = _resolve(lp.get(key))
            if r:
                audio_paths.append(r)

        # Generate a stable movie_id from title + year
        movie_id = f"{title.lower().replace(' ', '-').replace(':', '').replace(',', '')}_{year}"

        movies.append(MovieData(
            movie_id=movie_id,
            title=title,
            year=year,
            genres=[kind],
            video_path=video_path,
            image_paths=image_paths,
            text_chunks=text_chunks,
            audio_paths=audio_paths,
        ))

    print(f"  Loaded {len(movies)} titles from trailer_data/ ({skipped} skipped)")
    return movies


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYNTHETIC INGESTION (for demo without GPU)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _simulate_encoded_vectors(
    n_vecs: int, genre_centroids: dict, genres: list[str],
    source_dim: int, noise: float = 0.1,
) -> np.ndarray:
    """Generate synthetic vectors clustered by genre (simulates encoder output)."""
    base = sum(genre_centroids[g] for g in genres) / len(genres)
    vecs = np.array([base + np.random.randn(source_dim) * noise for _ in range(n_vecs)])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return (vecs / norms).astype(np.float32)


def demo_with_synthetic_data():
    """
    End-to-end demo using synthetic vectors.
    Simulates what SigLIP2 and CLAP would produce, then runs
    projection → fusion → FAISS indexing → search → UMAP visualization.
    """
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  DEMO: SigLIP2 + CLAP Multimodal Movie Retrieval         ║")
    print("╚" + "═" * 58 + "╝")
    print()

    SIGLIP2_DIM = 1152
    CLAP_DIM = 512
    COMMON_DIM = 512

    movies = [
        {"id": "dark-knight", "title": "The Dark Knight", "year": 2008,
         "genres": ["Action", "Thriller"], "n_kf": 10, "n_txt": 3, "n_aud": 5},
        {"id": "inception", "title": "Inception", "year": 2010,
         "genres": ["Sci-Fi", "Thriller"], "n_kf": 8, "n_txt": 4, "n_aud": 4},
        {"id": "interstellar", "title": "Interstellar", "year": 2014,
         "genres": ["Sci-Fi", "Drama"], "n_kf": 12, "n_txt": 5, "n_aud": 6},
        {"id": "joker", "title": "Joker", "year": 2019,
         "genres": ["Drama", "Thriller"], "n_kf": 7, "n_txt": 3, "n_aud": 3},
        {"id": "endgame", "title": "Avengers: Endgame", "year": 2019,
         "genres": ["Action", "Sci-Fi"], "n_kf": 15, "n_txt": 6, "n_aud": 8},
        {"id": "parasite", "title": "Parasite", "year": 2019,
         "genres": ["Drama", "Thriller"], "n_kf": 9, "n_txt": 4, "n_aud": 3},
        {"id": "dune", "title": "Dune", "year": 2021,
         "genres": ["Sci-Fi", "Adventure"], "n_kf": 11, "n_txt": 5, "n_aud": 5},
        {"id": "batman-begins", "title": "Batman Begins", "year": 2005,
         "genres": ["Action", "Thriller"], "n_kf": 8, "n_txt": 3, "n_aud": 4},
        {"id": "mad-max", "title": "Mad Max: Fury Road", "year": 2015,
         "genres": ["Action", "Adventure"], "n_kf": 13, "n_txt": 4, "n_aud": 7},
        {"id": "blade-runner", "title": "Blade Runner 2049", "year": 2017,
         "genres": ["Sci-Fi", "Drama"], "n_kf": 10, "n_txt": 5, "n_aud": 4},
        {"id": "matrix", "title": "The Matrix", "year": 1999,
         "genres": ["Sci-Fi", "Action"], "n_kf": 9, "n_txt": 4, "n_aud": 5},
        {"id": "no-country", "title": "No Country for Old Men", "year": 2007,
         "genres": ["Thriller", "Drama"], "n_kf": 6, "n_txt": 3, "n_aud": 2},
    ]

    # Genre centroids in BOTH encoder spaces
    np.random.seed(42)
    siglip2_centroids = {g: np.random.randn(SIGLIP2_DIM) * 0.3 for g in
                         ["Action", "Sci-Fi", "Drama", "Thriller", "Adventure"]}
    clap_centroids = {g: np.random.randn(CLAP_DIM) * 0.3 for g in
                      ["Action", "Sci-Fi", "Drama", "Thriller", "Adventure"]}

    # ── Initialize pipeline components ──
    aligner = DimensionAligner(
        source_dims={"siglip2": SIGLIP2_DIM, "clap": CLAP_DIM},
        target_dim=COMMON_DIM,
    )
    aggregator = LateFusionAggregator()
    index = MovieVectorIndex(COMMON_DIM)
    embeddings = []

    print("\n── Ingesting movies ──")
    for m in movies:
        # Simulate SigLIP2 outputs (1152-dim) for images and text
        raw_image = _simulate_encoded_vectors(m["n_kf"], siglip2_centroids, m["genres"], SIGLIP2_DIM)
        raw_text = _simulate_encoded_vectors(m["n_txt"], siglip2_centroids, m["genres"], SIGLIP2_DIM)

        # Simulate CLAP outputs (512-dim) for audio
        raw_audio = _simulate_encoded_vectors(m["n_aud"], clap_centroids, m["genres"], CLAP_DIM)

        # ── Project to common 512-dim space ──
        proj_image = aligner.project(raw_image, "siglip2")
        proj_text = aligner.project(raw_text, "siglip2")
        proj_audio = aligner.project(raw_audio, "clap")  # identity (already 512)

        # ── Late fusion ──
        fused = aggregator.fuse(
            image_vectors=proj_image,
            text_vectors=proj_text,
            audio_vectors=proj_audio,
        )

        emb = MovieEmbedding(
            movie_id=m["id"], title=m["title"], year=m["year"],
            genres=m["genres"], fused_vector=fused,
            image_vectors=proj_image, text_vectors=proj_text,
            audio_vectors=proj_audio,
        )
        index.add_movie(emb)
        embeddings.append(emb)
        print(f"  ✓ {m['title']} ({m['year']}) — "
              f"{m['n_kf']} keyframes + {m['n_txt']} texts + {m['n_aud']} audio clips")

    # ── Similarity Search ──
    print(f"\n── Similarity Search ──")

    # Simulate a text query (SigLIP2 text encoder → project)
    query_raw = _simulate_encoded_vectors(1, siglip2_centroids, ["Action", "Thriller"], SIGLIP2_DIM)
    query_vec = aligner.project(query_raw, "siglip2")[0]

    print(f'  Query: "dark superhero action thriller"')
    print(f"  Query path: SigLIP2 text encoder → 1152-d → project → 512-d → FAISS search")
    print()
    results = index.search(query_vec, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r['title']} ({r['year']}) "
              f"[{', '.join(r['genres'])}] "
              f"— cosine sim: {r['similarity_score']:.4f}")

    # ── UMAP Visualization ──
    if HAS_UMAP:
        print(f"\n── UMAP Visualization ──")
        visualizer = MovieSpaceVisualizer(n_neighbors=5, min_dist=0.3)
        all_vectors = index.get_all_vectors()
        combined = np.vstack([all_vectors, query_vec.reshape(1, -1)])
        projected = visualizer.fit_transform(combined)
        visualizer.plot(
            projected[:-1], index.movie_metadata,
            query_point_3d=projected[-1],
            query_label='Query: "dark superhero action thriller"',
            save_path="movie_embedding_space_siglip2_clap.html",
            title="Movie Embedding Space — SigLIP2 + CLAP (UMAP Projection)",
        )

    # ── Print pipeline summary ──
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  PIPELINE SUMMARY                                        ║")
    print("╠" + "═" * 58 + "╣")
    print("║                                                          ║")
    print("║  ENCODERS:                                               ║")
    print("║    SigLIP2  → keyframes, trailer frames, text (1152-d)   ║")
    print("║    CLAP     → audio dialogue/soundtrack      ( 512-d)    ║")
    print("║                                                          ║")
    print("║  ALIGNMENT:                                              ║")
    print("║    SigLIP2 1152-d ──→ orthogonal proj ──→ 512-d          ║")
    print("║    CLAP     512-d ──→ identity         ──→ 512-d         ║")
    print("║                                                          ║")
    print("║  FUSION:                                                 ║")
    print("║    Weighted avg (text 35%, image 25%, trailer 20%,       ║")
    print("║                  audio 20%) → L2-normalize → 512-d       ║")
    print("║                                                          ║")
    print("║  INDEX:                                                  ║")
    print("║    FAISS IndexFlatIP (cosine sim on normalized vecs)      ║")
    print("║                                                          ║")
    print("║  VISUALIZATION:                                          ║")
    print("║    UMAP (512-d → 2-d) with genre coloring               ║")
    print("║                                                          ║")
    print("╚" + "═" * 58 + "╝")


def run_search_loop(retrieval: "MultimodalMovieRetrieval"):
    """Interactive search loop — takes queries from the user via terminal."""
    print("\n── Interactive Search (type 'quit' to exit) ──")
    while True:
        try:
            query = input("\nEnter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        try:
            query_vec = retrieval.encode_query(query)
            results   = retrieval.index.search(query_vec, top_k=5)
            print()
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['title']} ({r['year']}) — cosine sim: {r['similarity_score']:.4f}")

            # UMAP: all corpus movies + query point + top-K results highlighted
            if retrieval.visualizer:
                safe_name = "".join(c if c.isalnum() else "_" for c in query)[:40]
                save_path = f"umap_{safe_name}.html"
                highlight_ids = [r["movie_id"] for r in results]
                retrieval.visualize(
                    query_vec=query_vec,
                    query_label=f'Query: "{query}"',
                    save_path=save_path,
                    highlight_ids=highlight_ids,
                )
                print(f"  UMAP saved → {save_path}")
        except Exception as e:
            print(f"  Search failed: {e}")


def run_real_pipeline(reindex: bool = False):
    """
    Full pipeline using real models and trailer data.
    Loads trailers_metadata.json, encodes each title with X-CLIP + SigLIP2 + CLAP,
    builds a FAISS index, saves it, then enters an interactive search loop.

    Supports resuming: if a saved index already exists, already-encoded movies are
    skipped and the index is checkpointed every SAVE_EVERY movies.
    Pass reindex=True (or --reindex on the CLI) to ignore any existing index and
    re-encode everything from scratch.
    """
    SAVE_EVERY = 10  # checkpoint frequency (movies)

    print()
    print("╔" + "═" * 58 + "╗")
    print("║  REAL PIPELINE: X-CLIP + SigLIP2 + CLAP                 ║")
    print("╚" + "═" * 58 + "╝")
    print()

    index_path = str(TRAILER_DIR / "faiss_index")

    # ── Load dataset ──
    print("── Loading trailer dataset ──")
    movies = load_trailer_dataset()

    # ── Initialize retrieval system with real models ──
    print("\n── Loading models ──")
    retrieval = MultimodalMovieRetrieval(use_real_models=True)

    # ── Resume: load existing index and skip already-encoded movies ──
    already_indexed: set[str] = set()
    if not reindex and Path(index_path + ".faiss").exists():
        print(f"\n── Resuming from existing index: {index_path} ──")
        retrieval.load_index(index_path)
        already_indexed = {m["movie_id"] for m in retrieval.index.movie_metadata}
        print(f"  {len(already_indexed)} movies already indexed, skipping")
    elif reindex:
        print("\n── Reindex requested — existing index will be overwritten ──")

    # ── Ingest remaining movies ──
    remaining = [m for m in movies if m.movie_id not in already_indexed]
    print(f"\n── Ingesting {len(remaining)} titles ({len(already_indexed)} skipped) ──")
    newly_ingested = 0
    for i, movie in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {movie.title} ({movie.year})")
        try:
            retrieval.ingest_movie(movie)
            newly_ingested += 1
            if newly_ingested % SAVE_EVERY == 0:
                retrieval.index.save(index_path)
                print(f"  Checkpoint saved ({retrieval.index.index.ntotal} movies total)")
        except Exception as e:
            print(f"  FAILED: {e}")

    if retrieval.index.index.ntotal == 0:
        print("ERROR: No movies were indexed. Check file paths in metadata.")
        return

    # ── Final save ──
    print(f"\n── Saving index → {index_path} ──")
    retrieval.index.save(index_path)

    run_search_loop(retrieval)


def run_search_only():
    """
    Skip ingestion — load the saved FAISS index and go straight to search.
    Use this after the index has already been built.
    """
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  SEARCH MODE: loading saved index                        ║")
    print("╚" + "═" * 58 + "╝")
    print()

    index_path = str(TRAILER_DIR / "faiss_index")
    print(f"── Loading index from {index_path} ──")

    retrieval = MultimodalMovieRetrieval(use_real_models=True)
    retrieval.load_index(index_path)

    print(f"  {retrieval.index.index.ntotal} titles loaded into FAISS index.")

    run_search_loop(retrieval)


if __name__ == "__main__":
    import sys
    if "--search-only" in sys.argv:
        run_search_only()
    else:
        run_real_pipeline(reindex="--reindex" in sys.argv)