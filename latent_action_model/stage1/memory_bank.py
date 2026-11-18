"""Global latent memory bank and compatible-pool nearest-neighbour lookup.

The stage-1 loss queries a snapshot index with the current (live) posterior
mean, while candidate posterior means are read from a stop-gradient CPU bank.
The bank is updated online, but its FAISS indices are intentionally rebuilt
only at epoch boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


LATENT_BANK_SCHEMA = "lawam-stage1-latent-bank-v1"
LATENT_BANK_VERSION = 1
DEFAULT_FULL_REFRESH_EPOCHS = (5, 10, 15, 20, 25, 30, 35)
DEFAULT_TEACHER_BANDWIDTH_SCALE = 0.30


class ArtifactValidationError(ValueError):
    """Raised when a persisted artifact does not match the active dataset."""


class FaissUnavailableError(RuntimeError):
    """Raised when production FAISS lookup was requested but is unavailable."""


def _as_float32_2d(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return np.ascontiguousarray(array)


def _as_strings(values: Sequence[Any], *, name: str) -> np.ndarray:
    result = np.asarray([str(value) for value in values], dtype=np.str_)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return result


def sample_id_digest(sample_ids: Sequence[Any]) -> str:
    """Order-sensitive, unambiguous digest used by every stage-1 artifact."""

    digest = hashlib.sha256()
    for sample_id in sample_ids:
        encoded = str(sample_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_savez(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_metadata(archive: Any, *, expected_schema: str) -> dict[str, Any]:
    if "metadata_json" not in archive.files:
        raise ArtifactValidationError("Artifact has no metadata_json field.")
    try:
        metadata = json.loads(str(archive["metadata_json"].item()))
    except Exception as exc:  # pragma: no cover - defensive corruption path
        raise ArtifactValidationError("Artifact metadata_json is invalid.") from exc
    if metadata.get("schema") != expected_schema:
        raise ArtifactValidationError(
            f"Artifact schema mismatch: expected {expected_schema!r}, got {metadata.get('schema')!r}."
        )
    if int(metadata.get("version", -1)) != LATENT_BANK_VERSION:
        raise ArtifactValidationError(
            f"Unsupported {expected_schema} version {metadata.get('version')!r}; expected {LATENT_BANK_VERSION}."
        )
    return metadata


class FlatL2Index:
    """Exact squared-L2 index backed by FAISS or an explicit test fallback.

    ``backend='faiss'`` is the production default.  NumPy lookup is deliberately
    opt-in so a missing FAISS installation cannot silently turn a full training
    run into an O(N^2) job.
    """

    def __init__(
        self,
        vectors: Any,
        *,
        ids: Sequence[int] | None = None,
        backend: str = "faiss",
        allow_exact_fallback: bool = False,
    ) -> None:
        self.vectors = _as_float32_2d(vectors, name="vectors")
        if ids is None:
            self.ids = np.arange(len(self.vectors), dtype=np.int64)
        else:
            self.ids = np.asarray(ids, dtype=np.int64)
        if self.ids.shape != (len(self.vectors),):
            raise ValueError("ids must contain exactly one integer for each vector.")
        if len(np.unique(self.ids)) != len(self.ids):
            raise ValueError("ids must be unique.")

        normalized_backend = str(backend).lower()
        if normalized_backend not in {"faiss", "auto", "exact"}:
            raise ValueError("backend must be one of: faiss, auto, exact.")
        self.backend = normalized_backend
        self._index: Any | None = None
        if normalized_backend in {"faiss", "auto"}:
            try:
                import faiss  # type: ignore
            except ImportError as exc:
                if not allow_exact_fallback:
                    raise FaissUnavailableError(
                        "FAISS is required for the full stage-1 memory bank. Install faiss-cpu/faiss-gpu, "
                        "or set backend='exact', allow_exact_fallback=True only for small synthetic tests."
                    ) from exc
                self.backend = "exact"
            else:
                index = faiss.IndexFlatL2(int(self.vectors.shape[1]))
                index.add(self.vectors)
                self._index = index
                self.backend = "faiss"
        elif not allow_exact_fallback:
            raise ValueError("backend='exact' is restricted to tests; pass allow_exact_fallback=True explicitly.")

    def search(self, queries: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        query_array = _as_float32_2d(queries, name="queries")
        if query_array.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"Query dimension {query_array.shape[1]} does not match index dimension {self.vectors.shape[1]}."
            )
        if not 1 <= int(k) <= len(self.vectors):
            raise ValueError(f"k must be in [1, {len(self.vectors)}], got {k}.")
        k = int(k)
        if self.backend == "faiss":
            distances, local_indices = self._index.search(query_array, k)
            return distances.astype(np.float32, copy=False), self.ids[local_indices]

        all_distances = np.empty((len(query_array), k), dtype=np.float32)
        all_ids = np.empty((len(query_array), k), dtype=np.int64)
        for row, query in enumerate(query_array):
            delta = self.vectors - query[None, :]
            distances = np.einsum("nd,nd->n", delta, delta, dtype=np.float32)
            # Stable global-id tie breaking makes the fallback fully reproducible.
            order = np.lexsort((self.ids, distances))[:k]
            all_distances[row] = distances[order]
            all_ids[row] = self.ids[order]
        return all_distances, all_ids


class PoolL2Index:
    """A full-library index plus one exact FAISS index per compatible pool."""

    def __init__(
        self,
        vectors: Any,
        pool_ids: Sequence[Any],
        *,
        backend: str = "faiss",
        allow_exact_fallback: bool = False,
    ) -> None:
        self.vectors = _as_float32_2d(vectors, name="vectors")
        self.pool_ids = _as_strings(pool_ids, name="pool_ids")
        if len(self.pool_ids) != len(self.vectors):
            raise ValueError("pool_ids length must match vectors.")
        self.global_index = FlatL2Index(
            self.vectors,
            backend=backend,
            allow_exact_fallback=allow_exact_fallback,
        )
        self.pool_members: dict[str, np.ndarray] = {}
        self.pool_indices: dict[str, FlatL2Index] = {}
        for pool_id in sorted(set(self.pool_ids.tolist())):
            members = np.flatnonzero(self.pool_ids == pool_id).astype(np.int64)
            self.pool_members[pool_id] = members
            self.pool_indices[pool_id] = FlatL2Index(
                self.vectors[members],
                ids=members,
                backend=backend,
                allow_exact_fallback=allow_exact_fallback,
            )

    def search_global(self, queries: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.global_index.search(queries, k)

    def search_pool_filtered(
        self,
        query: Any,
        *,
        pool_id: str,
        k: int,
        excluded_ids: Iterable[int] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        if pool_id not in self.pool_indices:
            raise KeyError(f"Unknown compatible pool {pool_id!r}.")
        excluded = set(int(index) for index in excluded_ids)
        pool_size = len(self.pool_members[pool_id])
        if pool_size - len(excluded.intersection(self.pool_members[pool_id].tolist())) < k:
            raise ValueError(f"Pool {pool_id!r} has fewer than {k} valid candidates after exclusions.")

        index = self.pool_indices[pool_id]
        requested = min(pool_size, max(int(k) + 32, 4 * int(k)))
        while True:
            distances, indices = index.search(np.asarray(query, dtype=np.float32).reshape(1, -1), requested)
            keep = np.asarray([int(candidate) not in excluded for candidate in indices[0]], dtype=bool)
            kept_indices = indices[0][keep]
            kept_distances = distances[0][keep]
            if len(kept_indices) >= k:
                return kept_distances[:k], kept_indices[:k]
            if requested == pool_size:
                raise ValueError(f"Pool {pool_id!r} has fewer than {k} searchable candidates after exclusions.")
            requested = min(pool_size, requested * 2)


def windows_overlap(
    anchor: int,
    candidates: np.ndarray,
    episode_keys: np.ndarray,
    start_steps: np.ndarray,
    end_steps: np.ndarray,
) -> np.ndarray:
    """Return candidates whose inclusive windows overlap the anchor episode."""

    same_episode = episode_keys[candidates] == episode_keys[int(anchor)]
    temporal_overlap = np.maximum(start_steps[candidates], start_steps[int(anchor)]) <= np.minimum(
        end_steps[candidates], end_steps[int(anchor)]
    )
    return same_episode & temporal_overlap


class LatentMemoryBank:
    """CPU FP32 bank initialized from release posterior means."""

    def __init__(
        self,
        sample_ids: Sequence[Any],
        latents: Any,
        pool_ids: Sequence[Any],
        episode_keys: Sequence[Any],
        start_steps: Sequence[int],
        end_steps: Sequence[int],
        *,
        backend: str = "faiss",
        allow_exact_fallback: bool = False,
        full_refresh_epochs: Sequence[int] = DEFAULT_FULL_REFRESH_EPOCHS,
        build_index: bool = True,
    ) -> None:
        self.sample_ids = _as_strings(sample_ids, name="sample_ids")
        if len(set(self.sample_ids.tolist())) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique.")
        self.latents = _as_float32_2d(latents, name="latents").copy()
        self.pool_ids = _as_strings(pool_ids, name="pool_ids")
        self.episode_keys = _as_strings(episode_keys, name="episode_keys")
        self.start_steps = np.asarray(start_steps, dtype=np.int64)
        self.end_steps = np.asarray(end_steps, dtype=np.int64)
        expected = (len(self.sample_ids),)
        for name, value in (
            ("pool_ids", self.pool_ids),
            ("episode_keys", self.episode_keys),
            ("start_steps", self.start_steps),
            ("end_steps", self.end_steps),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}.")
        if len(self.latents) != len(self.sample_ids):
            raise ValueError("latents length must match sample_ids.")
        if np.any(self.end_steps < self.start_steps):
            raise ValueError("Every end_step must be greater than or equal to start_step.")

        self.backend = str(backend)
        self.allow_exact_fallback = bool(allow_exact_fallback)
        self.full_refresh_epochs = tuple(int(epoch) for epoch in full_refresh_epochs)
        self.index_epoch = -1
        self.full_refresh_epoch = 0
        self.online_updates = 0
        self.dirty = True
        self._sample_to_index = {sample_id: index for index, sample_id in enumerate(self.sample_ids.tolist())}
        self._index: PoolL2Index | None = None
        if build_index:
            self.rebuild_index(epoch=0)

    @classmethod
    def initialize_release(cls, *args: Any, **kwargs: Any) -> "LatentMemoryBank":
        """Named constructor documenting that epoch-zero values come from release."""

        return cls(*args, **kwargs)

    @property
    def latent_dim(self) -> int:
        return int(self.latents.shape[1])

    def indices_for_ids(self, sample_ids: Sequence[Any]) -> np.ndarray:
        try:
            return np.asarray([self._sample_to_index[str(sample_id)] for sample_id in sample_ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"Unknown sample_id {exc.args[0]!r}.") from exc

    def update(self, indices: Sequence[int], live_mu: Any) -> None:
        index_array = np.asarray(indices, dtype=np.int64)
        values = _as_float32_2d(live_mu, name="live_mu")
        if index_array.ndim != 1 or len(index_array) != len(values):
            raise ValueError("indices and live_mu must have the same batch length.")
        if values.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent dimension {self.latent_dim}, got {values.shape[1]}.")
        if np.any(index_array < 0) or np.any(index_array >= len(self.latents)):
            raise IndexError("Memory-bank update index is out of range.")
        self.latents[index_array] = values
        self.online_updates += int(len(index_array))
        self.dirty = True

    def replace_all(self, full_mu: Any, *, epoch: int) -> None:
        values = _as_float32_2d(full_mu, name="full_mu")
        if values.shape != self.latents.shape:
            raise ValueError(f"full_mu must have shape {self.latents.shape}, got {values.shape}.")
        self.latents[...] = values
        self.full_refresh_epoch = int(epoch)
        self.dirty = True

    def should_full_refresh(self, epoch: int) -> bool:
        return int(epoch) in self.full_refresh_epochs

    def rebuild_index(self, epoch: int) -> None:
        self._index = PoolL2Index(
            self.latents.copy(),
            self.pool_ids,
            backend=self.backend,
            allow_exact_fallback=self.allow_exact_fallback,
        )
        self.index_epoch = int(epoch)
        self.dirty = False

    def _excluded_for_anchor(self, anchor: int) -> np.ndarray:
        if self._index is None:
            raise RuntimeError("Latent index has not been built.")
        pool_id = str(self.pool_ids[anchor])
        pool_members = self._index.pool_members[pool_id]
        overlap = windows_overlap(anchor, pool_members, self.episode_keys, self.start_steps, self.end_steps)
        excluded = pool_members[overlap]
        if anchor not in excluded:
            excluded = np.concatenate((excluded, np.asarray([anchor], dtype=np.int64)))
        return np.unique(excluded)

    def query(self, anchor_indices: Sequence[int], live_mu: Any, *, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Latent index has not been built.")
        anchors = np.asarray(anchor_indices, dtype=np.int64)
        queries = _as_float32_2d(live_mu, name="live_mu")
        if anchors.ndim != 1 or len(anchors) != len(queries):
            raise ValueError("anchor_indices and live_mu must have the same batch length.")
        if queries.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent dimension {self.latent_dim}, got {queries.shape[1]}.")
        if np.any(anchors < 0) or np.any(anchors >= len(self.latents)):
            raise IndexError("Anchor index is out of range.")

        distances = np.empty((len(anchors), int(k)), dtype=np.float32)
        neighbours = np.empty((len(anchors), int(k)), dtype=np.int64)
        for row, (anchor, query) in enumerate(zip(anchors.tolist(), queries)):
            distance, neighbour = self._index.search_pool_filtered(
                query,
                pool_id=str(self.pool_ids[anchor]),
                k=int(k),
                excluded_ids=self._excluded_for_anchor(anchor),
            )
            distances[row] = distance
            neighbours[row] = neighbour
        return distances, neighbours

    def save(self, path: str | Path) -> None:
        metadata = {
            "schema": LATENT_BANK_SCHEMA,
            "version": LATENT_BANK_VERSION,
            "sample_count": len(self.sample_ids),
            "sample_id_sha256": sample_id_digest(self.sample_ids),
            "latent_dim": self.latent_dim,
            "index_epoch": self.index_epoch,
            "full_refresh_epoch": self.full_refresh_epoch,
            "online_updates": self.online_updates,
            "full_refresh_epochs": list(self.full_refresh_epochs),
        }
        _atomic_savez(
            path,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            sample_ids=self.sample_ids,
            latents=self.latents,
            pool_ids=self.pool_ids,
            episode_keys=self.episode_keys,
            start_steps=self.start_steps,
            end_steps=self.end_steps,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sample_ids: Sequence[Any],
        backend: str = "faiss",
        allow_exact_fallback: bool = False,
        build_index: bool = True,
        full_refresh_epochs: Sequence[int] | None = None,
    ) -> "LatentMemoryBank":
        with np.load(path, allow_pickle=False) as archive:
            metadata = _read_metadata(archive, expected_schema=LATENT_BANK_SCHEMA)
            sample_ids = archive["sample_ids"].astype(np.str_)
            expected = _as_strings(expected_sample_ids, name="expected_sample_ids")
            if not np.array_equal(sample_ids, expected):
                raise ArtifactValidationError(
                    "Latent bank sample_id order does not match the active train manifest; refusing unsafe load."
                )
            if metadata.get("sample_id_sha256") != sample_id_digest(sample_ids):
                raise ArtifactValidationError("Latent bank sample_id digest is corrupt.")
            bank = cls(
                sample_ids,
                archive["latents"],
                archive["pool_ids"],
                archive["episode_keys"],
                archive["start_steps"],
                archive["end_steps"],
                backend=backend,
                allow_exact_fallback=allow_exact_fallback,
                full_refresh_epochs=(
                    metadata.get("full_refresh_epochs", DEFAULT_FULL_REFRESH_EPOCHS)
                    if full_refresh_epochs is None
                    else full_refresh_epochs
                ),
                build_index=build_index,
            )
        bank.index_epoch = int(metadata.get("index_epoch", -1))
        bank.full_refresh_epoch = int(metadata.get("full_refresh_epoch", 0))
        bank.online_updates = int(metadata.get("online_updates", 0))
        return bank


@dataclass(frozen=True)
class NeighborBatch:
    candidate_indices: np.ndarray
    candidate_mask: np.ndarray
    source_mask: np.ndarray
    teacher_distances: np.ndarray
    bandwidth: np.ndarray
    candidate_bank_mu: np.ndarray


class Stage1NeighborProvider:
    """Join fixed teacher graph candidates with the epoch latent index."""

    EFFECT = np.uint8(1)
    LATENT = np.uint8(2)
    MIDDLE = np.uint8(4)
    FAR = np.uint8(8)

    def __init__(
        self,
        graph: Any,
        bank: LatentMemoryBank,
        *,
        k_z: int = 8,
        middle_count: int = 8,
        far_count: int = 8,
        seed: int = 20260803,
        teacher_bandwidth_scale: float = DEFAULT_TEACHER_BANDWIDTH_SCALE,
        teacher_bandwidth_floor: float = 0.0,
    ) -> None:
        if not np.array_equal(graph.sample_ids, bank.sample_ids):
            raise ArtifactValidationError("Teacher graph and latent bank sample_id order differ.")
        if not np.array_equal(graph.pool_ids, bank.pool_ids):
            raise ArtifactValidationError("Teacher graph and latent bank compatible-pool definitions differ.")
        if not np.array_equal(graph.episode_keys, bank.episode_keys):
            raise ArtifactValidationError("Teacher graph and latent bank episode keys differ.")
        if not np.array_equal(graph.start_steps, bank.start_steps) or not np.array_equal(
            graph.end_steps, bank.end_steps
        ):
            raise ArtifactValidationError("Teacher graph and latent bank window boundaries differ.")
        self.graph = graph
        self.bank = bank
        self.k_z = int(k_z)
        self.middle_count = int(middle_count)
        self.far_count = int(far_count)
        self.seed = int(seed)
        self.teacher_bandwidth_scale = float(teacher_bandwidth_scale)
        if not np.isfinite(self.teacher_bandwidth_scale) or self.teacher_bandwidth_scale <= 0:
            raise ValueError("teacher_bandwidth_scale must be finite and positive")
        self.teacher_bandwidth_floor = float(teacher_bandwidth_floor)
        if not np.isfinite(self.teacher_bandwidth_floor) or self.teacher_bandwidth_floor < 0:
            raise ValueError("teacher_bandwidth_floor must be finite and non-negative")

    def get(self, anchor_indices: Sequence[int], live_mu: Any, *, epoch: int) -> NeighborBatch:
        anchors = np.asarray(anchor_indices, dtype=np.int64)
        query_mu = _as_float32_2d(live_mu, name="live_mu")
        if len(anchors) != len(query_mu):
            raise ValueError("anchor_indices and live_mu must have the same batch length.")
        _, latent_neighbours = self.bank.query(anchors, query_mu, k=self.k_z)
        middle, far = self.graph.sample_mid_far(
            anchors,
            epoch=int(epoch),
            middle_count=self.middle_count,
            far_count=self.far_count,
            seed=self.seed,
        )
        effect = self.graph.knn_indices[anchors]
        width = int(effect.shape[1] + self.k_z + self.middle_count + self.far_count)
        candidate_indices = np.full((len(anchors), width), -1, dtype=np.int64)
        candidate_mask = np.zeros((len(anchors), width), dtype=bool)
        source_mask = np.zeros((len(anchors), width), dtype=np.uint8)
        teacher_distances = np.zeros((len(anchors), width), dtype=np.float32)
        candidate_bank_mu = np.zeros((len(anchors), width, self.bank.latent_dim), dtype=np.float32)

        for row, anchor in enumerate(anchors.tolist()):
            ordered: list[int] = []
            sources: dict[int, np.uint8] = {}
            for values, source in (
                (effect[row], self.EFFECT),
                (latent_neighbours[row], self.LATENT),
                (middle[row], self.MIDDLE),
                (far[row], self.FAR),
            ):
                for candidate in values.tolist():
                    candidate = int(candidate)
                    if candidate not in sources:
                        ordered.append(candidate)
                        sources[candidate] = source
                    else:
                        sources[candidate] = np.uint8(sources[candidate] | source)
            count = len(ordered)
            selected = np.asarray(ordered, dtype=np.int64)
            candidate_indices[row, :count] = selected
            candidate_mask[row, :count] = True
            source_mask[row, :count] = np.asarray([sources[index] for index in ordered], dtype=np.uint8)
            teacher_distances[row, :count] = self.graph.distances(anchor, selected)
            candidate_bank_mu[row, :count] = self.bank.latents[selected]

        return NeighborBatch(
            candidate_indices=candidate_indices,
            candidate_mask=candidate_mask,
            source_mask=source_mask,
            teacher_distances=teacher_distances,
            # The rank-based bandwidth remains anchor-adaptive. A single
            # train-calibrated scale prevents the top-16 set becoming nearly
            # uniform on the real candidate distribution.
            bandwidth=np.maximum(
                self.graph.knn_distances[anchors, self.graph.k_effect - 1]
                * self.teacher_bandwidth_scale,
                self.teacher_bandwidth_floor,
            ).astype(np.float32, copy=True),
            candidate_bank_mu=candidate_bank_mu,
        )

    def update_bank(self, anchor_indices: Sequence[int], live_mu: Any) -> None:
        self.bank.update(anchor_indices, live_mu)

    def rebuild_index(self, epoch: int) -> None:
        self.bank.rebuild_index(epoch=int(epoch))
