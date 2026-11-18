"""Frozen DINO-residual teacher, train-only PCA, and compatible-pool graph.

V1 invariants implemented here are intentionally strict:

* final post-LayerNorm DINO tokens, with CLS + four register tokens removed;
* per-patch vector L2 normalization followed by ``end - start``;
* train-only, non-whitened PCA and one global RMS scale (never sample norm);
* mean Huber distance with delta=1;
* fixed effect top-16 plus per-anchor 30--60% and 70--95% rank bands.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .memory_bank import ArtifactValidationError, FlatL2Index, sample_id_digest, windows_overlap


PCA_SCHEMA = "lawam-stage1-dino-residual-pca-v1"
TEACHER_GRAPH_SCHEMA = "lawam-stage1-teacher-graph-v1"
RESIDUAL_STORE_SCHEMA = "lawam-stage1-dino-residual-store-v1"
ARTIFACT_VERSION = 1
SPECIAL_TOKEN_COUNT = 5  # one CLS plus four DINOv3 register tokens
DEFAULT_EPS = 1.0e-6


class CompressionValidationError(RuntimeError):
    """Raised when no configured PCA dimension preserves O16 >= 0.80."""


class PCAUnavailableError(RuntimeError):
    """Raised when the requested scalable PCA implementation is unavailable."""


def _float32_2d(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return np.ascontiguousarray(array)


def _strings(values: Sequence[Any], *, name: str) -> np.ndarray:
    array = np.asarray([str(value) for value in values], dtype=np.str_)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


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


def _metadata_from_archive(archive: Any, schema: str) -> dict[str, Any]:
    if "metadata_json" not in archive.files:
        raise ArtifactValidationError("Artifact has no metadata_json field.")
    try:
        metadata = json.loads(str(archive["metadata_json"].item()))
    except Exception as exc:  # pragma: no cover - corrupt archive path
        raise ArtifactValidationError("Artifact metadata_json is invalid.") from exc
    if metadata.get("schema") != schema or int(metadata.get("version", -1)) != ARTIFACT_VERSION:
        raise ArtifactValidationError(
            f"Expected schema/version {schema}/{ARTIFACT_VERSION}, got "
            f"{metadata.get('schema')}/{metadata.get('version')}."
        )
    return metadata


def final_post_ln_patch_tokens(last_hidden_state: Any, *, special_token_count: int = SPECIAL_TOKEN_COUNT) -> Any:
    """Strip CLS/register tokens from a final post-LN DINO hidden state."""

    if getattr(last_hidden_state, "ndim", None) != 3:
        raise ValueError(
            f"DINO last_hidden_state must have shape [B, 5+P, D], got "
            f"{getattr(last_hidden_state, 'shape', None)}."
        )
    if int(last_hidden_state.shape[1]) <= int(special_token_count):
        raise ValueError("DINO output does not contain any patch tokens after removing five special tokens.")
    return last_hidden_state[:, int(special_token_count) :, :]


def patch_l2_residual(start_patch_tokens: Any, end_patch_tokens: Any, *, eps: float = DEFAULT_EPS) -> Any:
    """Compute flattened per-patch-L2 ``end - start`` without sample norm."""

    if tuple(start_patch_tokens.shape) != tuple(end_patch_tokens.shape):
        raise ValueError(
            f"Start/end patch layouts differ: {tuple(start_patch_tokens.shape)} vs {tuple(end_patch_tokens.shape)}."
        )
    if getattr(start_patch_tokens, "ndim", None) != 3:
        raise ValueError("Patch tokens must have shape [B, P, D].")

    # Preserve torch tensors for GPU extraction; NumPy is used by offline tests.
    if hasattr(start_patch_tokens, "detach"):
        import torch
        import torch.nn.functional as torch_functional

        start = torch_functional.normalize(start_patch_tokens.float(), p=2, dim=-1, eps=float(eps))
        end = torch_functional.normalize(end_patch_tokens.float(), p=2, dim=-1, eps=float(eps))
        return (end - start).reshape(start.shape[0], -1).to(dtype=torch.float32)

    start = np.asarray(start_patch_tokens, dtype=np.float32)
    end = np.asarray(end_patch_tokens, dtype=np.float32)
    start_norm = np.sqrt(np.sum(start * start, axis=-1, keepdims=True, dtype=np.float32))
    end_norm = np.sqrt(np.sum(end * end, axis=-1, keepdims=True, dtype=np.float32))
    start_unit = start / np.maximum(start_norm, np.float32(eps))
    end_unit = end / np.maximum(end_norm, np.float32(eps))
    return np.ascontiguousarray((end_unit - start_unit).reshape(start.shape[0], -1), dtype=np.float32)


def patch_mean_context(start_patch_tokens: Any, *, eps: float = DEFAULT_EPS) -> Any:
    """Return an L2-normalized mean of per-patch-normalized start tokens."""

    if getattr(start_patch_tokens, "ndim", None) != 3:
        raise ValueError("Patch tokens must have shape [B, P, D].")
    if hasattr(start_patch_tokens, "detach"):
        import torch.nn.functional as torch_functional

        patches = torch_functional.normalize(start_patch_tokens.float(), p=2, dim=-1, eps=float(eps))
        return torch_functional.normalize(patches.mean(dim=1), p=2, dim=-1, eps=float(eps))

    patches = np.asarray(start_patch_tokens, dtype=np.float32)
    patch_norm = np.sqrt(np.sum(patches * patches, axis=-1, keepdims=True, dtype=np.float32))
    pooled = np.mean(patches / np.maximum(patch_norm, np.float32(eps)), axis=1, dtype=np.float32)
    pooled_norm = np.sqrt(np.sum(pooled * pooled, axis=-1, keepdims=True, dtype=np.float32))
    return np.ascontiguousarray(pooled / np.maximum(pooled_norm, np.float32(eps)), dtype=np.float32)


def extract_final_patch_residual_and_context(
    model: Any,
    start_pixels: Any,
    end_pixels: Any,
    *,
    expected_patch_count: int | None = None,
) -> tuple[Any, Any]:
    """Run frozen DINO once and return residual plus start-context descriptor."""

    import torch

    if tuple(start_pixels.shape) != tuple(end_pixels.shape):
        raise ValueError("Start/end pixels must have identical [B,C,H,W] layouts.")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        pixels = torch.cat((start_pixels, end_pixels), dim=0)
        outputs = model(pixel_values=pixels, output_hidden_states=False, return_dict=True)
        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError("DINO model output has no final post-LN last_hidden_state.")
        patches = final_post_ln_patch_tokens(outputs.last_hidden_state)
        if expected_patch_count is not None and int(patches.shape[1]) != int(expected_patch_count):
            raise ValueError(
                f"Expected {expected_patch_count} patch tokens after removing five special tokens, "
                f"got {patches.shape[1]}."
            )
        start_patches, end_patches = patches.chunk(2, dim=0)
        return patch_l2_residual(start_patches, end_patches).detach(), patch_mean_context(start_patches).detach()


def extract_final_patch_residual(
    model: Any,
    start_pixels: Any,
    end_pixels: Any,
    *,
    expected_patch_count: int | None = None,
) -> Any:
    """Run a permanently frozen DINO model and return FP32 patch residuals.

    ``start_pixels`` and ``end_pixels`` must already have undergone the exact
    same deterministic crop/resize pipeline.  Concatenating them into one
    forward pass also prevents endpoint-specific model paths.
    """

    residual, _ = extract_final_patch_residual_and_context(
        model,
        start_pixels,
        end_pixels,
        expected_patch_count=expected_patch_count,
    )
    return residual


class ResidualStore:
    """Memmapped residual matrix with an order-checked sample-ID sidecar."""

    def __init__(self, root: str | Path, *, expected_sample_ids: Sequence[Any] | None = None) -> None:
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactValidationError(f"Invalid residual-store metadata: {metadata_path}.") from exc
        if metadata.get("schema") != RESIDUAL_STORE_SCHEMA or int(metadata.get("version", -1)) != ARTIFACT_VERSION:
            raise ArtifactValidationError("Residual-store schema/version mismatch.")
        self.metadata = metadata
        self.sample_ids = np.load(self.root / "sample_ids.npy", allow_pickle=False).astype(np.str_)
        self.residuals = np.load(self.root / "residuals.npy", mmap_mode="r", allow_pickle=False)
        if self.residuals.dtype != np.float32 or self.residuals.ndim != 2:
            raise ArtifactValidationError("Residual matrix must be a [N,D] float32 .npy array.")
        if len(self.sample_ids) != len(self.residuals):
            raise ArtifactValidationError("Residual matrix and sample_ids length differ.")
        if metadata.get("sample_id_sha256") != sample_id_digest(self.sample_ids):
            raise ArtifactValidationError("Residual-store sample_id digest is corrupt.")
        if expected_sample_ids is not None and not np.array_equal(
            self.sample_ids, _strings(expected_sample_ids, name="expected_sample_ids")
        ):
            raise ArtifactValidationError("Residual store does not match the active train manifest sample_id order.")
        context_filename = metadata.get("start_context_file")
        self.start_context = None
        if context_filename is not None:
            context = np.load(self.root / str(context_filename), mmap_mode="r", allow_pickle=False)
            if context.dtype != np.float32 or context.ndim != 2 or len(context) != len(self.residuals):
                raise ArtifactValidationError("Start-context matrix must be [N,D] FP32 and align with residuals.")
            for start in range(0, len(context), 4096):
                if not np.isfinite(context[start : start + 4096]).all():
                    raise ArtifactValidationError("Start-context matrix contains NaN or infinity.")
            self.start_context = context

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        sample_ids: Sequence[Any],
        feature_dim: int,
        source_metadata: dict[str, Any],
    ) -> tuple[np.memmap, "ResidualStoreWriter"]:
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=False)
        ids = _strings(sample_ids, name="sample_ids")
        np.save(destination / "sample_ids.npy", ids, allow_pickle=False)
        matrix = np.lib.format.open_memmap(
            destination / "residuals.npy", mode="w+", dtype=np.float32, shape=(len(ids), int(feature_dim))
        )
        writer = ResidualStoreWriter(destination, ids, matrix, source_metadata)
        return matrix, writer


class ResidualStoreWriter:
    def __init__(
        self,
        root: Path,
        sample_ids: np.ndarray,
        matrix: np.memmap,
        source_metadata: dict[str, Any],
    ) -> None:
        self.root = root
        self.sample_ids = sample_ids
        self.matrix = matrix
        self.source_metadata = source_metadata

    def finalize(self) -> None:
        self.matrix.flush()
        metadata = {
            "schema": RESIDUAL_STORE_SCHEMA,
            "version": ARTIFACT_VERSION,
            "sample_count": len(self.sample_ids),
            "sample_id_sha256": sample_id_digest(self.sample_ids),
            "shape": list(self.matrix.shape),
            "dtype": "float32",
            "split": str(self.source_metadata.get("source_split", "unknown")),
            **self.source_metadata,
        }
        (self.root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_slices(length: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, int(length), int(batch_size)):
        yield slice(start, min(int(length), start + int(batch_size)))


@dataclass(frozen=True)
class FrozenPCA:
    mean: np.ndarray
    components: np.ndarray
    global_rms: float
    sample_id_sha256: str
    fit_count: int
    backend: str
    overlap_at_16: float | None = None
    validation_sample_count: int | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        components = np.asarray(self.components, dtype=np.float32)
        if mean.ndim != 1 or components.ndim != 2 or components.shape[1] != len(mean):
            raise ValueError("PCA mean/components shapes are inconsistent.")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(components)):
            raise ValueError("PCA parameters contain NaN or infinity.")
        if not np.isfinite(self.global_rms) or float(self.global_rms) <= 0:
            raise ValueError("PCA global RMS must be finite and positive.")
        object.__setattr__(self, "mean", np.ascontiguousarray(mean))
        object.__setattr__(self, "components", np.ascontiguousarray(components))

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, residuals: Any, *, scale: bool = True) -> np.ndarray:
        values = _float32_2d(residuals, name="residuals")
        if values.shape[1] != self.input_dim:
            raise ValueError(f"Expected residual dimension {self.input_dim}, got {values.shape[1]}.")
        projected = (values - self.mean[None, :]) @ self.components.T
        projected = np.asarray(projected, dtype=np.float32)
        if scale:
            projected /= np.float32(self.global_rms + DEFAULT_EPS)
        return np.ascontiguousarray(projected)

    def save(self, path: str | Path) -> None:
        metadata = {
            "schema": PCA_SCHEMA,
            "version": ARTIFACT_VERSION,
            "fit_split": "train_only",
            "whiten": False,
            "sample_id_sha256": self.sample_id_sha256,
            "fit_count": int(self.fit_count),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "global_rms": float(self.global_rms),
            "backend": self.backend,
            "overlap_at_16": self.overlap_at_16,
            "validation_sample_count": self.validation_sample_count,
        }
        _atomic_savez(
            path,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            mean=self.mean,
            components=self.components,
        )

    @classmethod
    def load(cls, path: str | Path, *, expected_sample_ids: Sequence[Any] | None = None) -> "FrozenPCA":
        with np.load(path, allow_pickle=False) as archive:
            metadata = _metadata_from_archive(archive, PCA_SCHEMA)
            expected_digest = None if expected_sample_ids is None else sample_id_digest(expected_sample_ids)
            if expected_digest is not None and metadata.get("sample_id_sha256") != expected_digest:
                raise ArtifactValidationError(
                    "PCA was fitted on a different train sample_id set/order; refusing unsafe load."
                )
            if metadata.get("fit_split") != "train_only" or bool(metadata.get("whiten", True)):
                raise ArtifactValidationError("PCA artifact violates train-only/no-whitening V1 invariants.")
            return cls(
                mean=archive["mean"],
                components=archive["components"],
                global_rms=float(metadata["global_rms"]),
                sample_id_sha256=str(metadata["sample_id_sha256"]),
                fit_count=int(metadata["fit_count"]),
                backend=str(metadata["backend"]),
                overlap_at_16=metadata.get("overlap_at_16"),
                validation_sample_count=metadata.get("validation_sample_count"),
            )


def _fit_components(
    residuals: Any,
    n_components: int,
    *,
    backend: str,
    batch_size: int,
    seed: int,
    allow_numpy_test_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    count, feature_dim = residuals.shape
    if int(n_components) > min(int(count), int(feature_dim)):
        raise ValueError(
            f"PCA-{n_components} requires at least {n_components} samples and source dimensions; "
            f"got shape {residuals.shape}."
        )
    backend = str(backend).lower()
    if backend == "torch_randomized_cuda":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - production dependency
            raise PCAUnavailableError("PyTorch is required for torch_randomized_cuda PCA.") from exc
        if not torch.cuda.is_available():
            raise PCAUnavailableError("torch_randomized_cuda PCA requires an available CUDA device.")

        def report(stage: str) -> None:
            print(
                f"[pca:{n_components}] {stage} rows={count} source_dim={feature_dim}",
                flush=True,
            )

        report("mean:start")
        mean_accumulator = np.zeros(feature_dim, dtype=np.float64)
        for batch_slice in _iter_slices(count, int(batch_size)):
            mean_accumulator += np.sum(np.asarray(residuals[batch_slice], dtype=np.float32), axis=0, dtype=np.float64)
        mean = (mean_accumulator / float(count)).astype(np.float32)
        report("mean:done")

        device = torch.device("cuda")
        torch.manual_seed(int(seed))
        rank = min(int(n_components) + 32, count, feature_dim)
        omega = torch.randn(feature_dim, rank, device=device, dtype=torch.float32)
        omega.mul_(rank**-0.5)
        mean_gpu = torch.from_numpy(mean).to(device)
        y = torch.empty((count, rank), device=device, dtype=torch.float32)
        report("randomized_range:start")
        for batch_slice in _iter_slices(count, int(batch_size)):
            batch = torch.from_numpy(np.asarray(residuals[batch_slice], dtype=np.float32)).to(device)
            y[batch_slice] = (batch - mean_gpu) @ omega
        del omega
        report("randomized_range:qr")
        q, _ = torch.linalg.qr(y, mode="reduced")
        del y

        # One subspace iteration materially improves the tail of PCA-256 while
        # keeping the large source memmap out of GPU memory.
        feature_basis = torch.zeros((feature_dim, rank), device=device, dtype=torch.float32)
        report("subspace_basis:start")
        for batch_slice in _iter_slices(count, int(batch_size)):
            batch = torch.from_numpy(np.asarray(residuals[batch_slice], dtype=np.float32)).to(device)
            feature_basis.add_((batch - mean_gpu).T @ q[batch_slice])
        report("subspace_basis:qr")
        feature_basis, _ = torch.linalg.qr(feature_basis, mode="reduced")
        y = torch.empty((count, rank), device=device, dtype=torch.float32)
        report("subspace_projection:start")
        for batch_slice in _iter_slices(count, int(batch_size)):
            batch = torch.from_numpy(np.asarray(residuals[batch_slice], dtype=np.float32)).to(device)
            y[batch_slice] = (batch - mean_gpu) @ feature_basis
        report("subspace_projection:qr")
        q, _ = torch.linalg.qr(y, mode="reduced")
        del y, feature_basis

        compressed = torch.zeros((rank, feature_dim), device=device, dtype=torch.float32)
        report("compressed_basis:start")
        for batch_slice in _iter_slices(count, int(batch_size)):
            batch = torch.from_numpy(np.asarray(residuals[batch_slice], dtype=np.float32)).to(device)
            compressed.add_(q[batch_slice].T @ (batch - mean_gpu))
        report("compressed_basis:svd")
        if not torch.isfinite(compressed).all():
            raise RuntimeError("Randomized PCA compressed basis contains NaN or infinity.")
        try:
            _, _, right = torch.linalg.svd(compressed, full_matrices=False, driver="gesvdj")
        except RuntimeError as exc:
            if "CUSOLVER_STATUS_INVALID_VALUE" not in str(exc):
                raise
            # gesvdj rejects some very wide high-rank matrices (for example
            # PCA-8192 over 196608-D DINO residuals) even when inputs are finite.
            report("compressed_basis:svd_fallback_gesvd")
            _, _, right = torch.linalg.svd(compressed, full_matrices=False, driver="gesvd")
        components = right[: int(n_components)].float().cpu().numpy()
        report("done")
        return mean, np.ascontiguousarray(components, dtype=np.float32), backend

    if backend == "numpy":
        if not allow_numpy_test_fallback:
            raise ValueError("NumPy full SVD is test-only; explicitly set allow_numpy_test_fallback=True.")
        values = _float32_2d(residuals, name="residuals")
        mean = np.mean(values, axis=0, dtype=np.float32)
        _, _, right = np.linalg.svd(values - mean[None, :], full_matrices=False)
        return mean, np.asarray(right[:n_components], dtype=np.float32), "numpy_full_svd_test_only"

    try:
        if backend == "sklearn_randomized":
            from sklearn.decomposition import PCA

            estimator = PCA(
                n_components=int(n_components),
                whiten=False,
                svd_solver="randomized",
                random_state=int(seed),
            )
            estimator.fit(np.asarray(residuals, dtype=np.float32))
        elif backend == "sklearn_incremental":
            from sklearn.decomposition import IncrementalPCA

            effective_batch = max(int(batch_size), int(n_components))
            estimator = IncrementalPCA(n_components=int(n_components), whiten=False, batch_size=effective_batch)
            starts = list(range(0, count, effective_batch))
            if len(starts) > 1 and count - starts[-1] < n_components:
                starts.pop()
            for position, start in enumerate(starts):
                end = starts[position + 1] if position + 1 < len(starts) else count
                estimator.partial_fit(np.asarray(residuals[start:end], dtype=np.float32))
        else:
            raise ValueError(
                "PCA backend must be sklearn_incremental, sklearn_randomized, torch_randomized_cuda, or numpy."
            )
    except ImportError as exc:
        raise PCAUnavailableError(
            "scikit-learn is required for scalable stage-1 PCA. Install scikit-learn, or use the explicit "
            "NumPy fallback only in small synthetic tests."
        ) from exc
    return (
        np.asarray(estimator.mean_, dtype=np.float32),
        np.asarray(estimator.components_, dtype=np.float32),
        backend,
    )


def fit_train_only_pca(
    residuals: Any,
    sample_ids: Sequence[Any],
    *,
    n_components: int = 256,
    backend: str = "sklearn_incremental",
    batch_size: int = 512,
    seed: int = 20260803,
    allow_numpy_test_fallback: bool = False,
) -> FrozenPCA:
    """Fit no-whitening PCA and compute one train-global descriptor RMS."""

    if getattr(residuals, "ndim", None) != 2 or np.dtype(residuals.dtype) != np.dtype(np.float32):
        raise ValueError("Train residuals must be a [N,D] FP32 matrix.")
    ids = _strings(sample_ids, name="sample_ids")
    if len(ids) != len(residuals):
        raise ValueError("sample_ids length must match train residuals.")
    mean, components, fitted_backend = _fit_components(
        residuals,
        int(n_components),
        backend=backend,
        batch_size=int(batch_size),
        seed=int(seed),
        allow_numpy_test_fallback=allow_numpy_test_fallback,
    )
    square_sum = np.float64(0.0)
    element_count = 0
    if fitted_backend == "torch_randomized_cuda":
        import torch

        device = torch.device("cuda")
        mean_gpu = torch.from_numpy(mean).to(device)
        components_gpu = torch.from_numpy(components).to(device)
        for batch_slice in _iter_slices(len(residuals), int(batch_size)):
            batch = torch.from_numpy(np.asarray(residuals[batch_slice], dtype=np.float32)).to(device)
            projected = (batch - mean_gpu) @ components_gpu.T
            square_sum += float(torch.sum(projected.double().square()).item())
            element_count += int(projected.numel())
    else:
        for batch_slice in _iter_slices(len(residuals), int(batch_size)):
            batch = np.asarray(residuals[batch_slice], dtype=np.float32)
            projected = np.asarray((batch - mean[None, :]) @ components.T, dtype=np.float32)
            square_sum += np.sum(projected * projected, dtype=np.float64)
            element_count += int(projected.size)
    global_rms = float(np.sqrt(square_sum / float(element_count)))
    return FrozenPCA(
        mean=mean,
        components=components,
        global_rms=global_rms,
        sample_id_sha256=sample_id_digest(ids),
        fit_count=len(ids),
        backend=fitted_backend,
    )


def _topk_without_self(
    values: np.ndarray,
    *,
    k: int,
    backend: str,
    allow_exact_fallback: bool,
) -> np.ndarray:
    if backend == "torch_cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch_cuda kNN validation requires CUDA")
        device = torch.device("cuda")
        vectors = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(device)
        norms = vectors.square().sum(dim=1)
        result = np.empty((len(values), int(k)), dtype=np.int64)
        query_batch = 32 if values.shape[1] > 8192 else 512
        for start in range(0, len(values), query_batch):
            stop = min(len(values), start + query_batch)
            distances = norms[start:stop, None] + norms[None, :] - 2.0 * (vectors[start:stop] @ vectors.T)
            row = torch.arange(stop - start, device=device)
            distances[row, torch.arange(start, stop, device=device)] = float("inf")
            result[start:stop] = torch.topk(distances, k=int(k), dim=1, largest=False, sorted=True).indices.cpu().numpy()
        return result
    index = FlatL2Index(values, backend=backend, allow_exact_fallback=allow_exact_fallback)
    _, raw = index.search(values, min(len(values), int(k) + 1))
    result = np.empty((len(values), int(k)), dtype=np.int64)
    for anchor, row in enumerate(raw):
        filtered = row[row != anchor]
        if len(filtered) < k:
            raise ValueError("Not enough non-self samples for kNN overlap validation.")
        result[anchor] = filtered[:k]
    return result


def knn_overlap(
    raw_residuals: Any,
    projected_residuals: Any,
    *,
    k: int = 16,
    backend: str = "faiss",
    allow_exact_fallback: bool = False,
) -> float:
    raw = _float32_2d(raw_residuals, name="raw_residuals")
    projected = _float32_2d(projected_residuals, name="projected_residuals")
    if len(raw) != len(projected):
        raise ValueError("Raw/projected validation sets must have equal length.")
    if len(raw) <= int(k):
        raise ValueError(f"O{k} validation requires more than {k} samples.")
    raw_knn = _topk_without_self(raw, k=int(k), backend=backend, allow_exact_fallback=allow_exact_fallback)
    pca_knn = _topk_without_self(projected, k=int(k), backend=backend, allow_exact_fallback=allow_exact_fallback)
    overlap = [len(set(left.tolist()).intersection(right.tolist())) / float(k) for left, right in zip(raw_knn, pca_knn)]
    return float(np.mean(np.asarray(overlap, dtype=np.float32), dtype=np.float32))


def fit_validated_pca(
    residuals: Any,
    sample_ids: Sequence[Any],
    *,
    primary_dim: int = 256,
    fallback_dim: int = 512,
    maximum_dim: int = 1024,
    candidate_dimensions: Sequence[int] | None = None,
    validation_samples: int = 2000,
    overlap_k: int = 16,
    minimum_overlap: float = 0.80,
    pca_backend: str = "sklearn_incremental",
    index_backend: str = "faiss",
    batch_size: int = 512,
    seed: int = 20260803,
    allow_test_fallbacks: bool = False,
) -> FrozenPCA:
    """Enforce O16 >= .80 across an ordered dimension ladder."""

    count = len(residuals)
    validation_count = min(int(validation_samples), count)
    if validation_count <= int(overlap_k):
        raise ValueError("PCA validation subset is too small for the requested k.")
    rng = np.random.default_rng(int(seed))
    validation_indices = np.sort(rng.choice(count, size=validation_count, replace=False))
    raw_validation = np.asarray(residuals[validation_indices], dtype=np.float32)
    attempts: list[tuple[int, float]] = []
    configured_dimensions = (
        (int(primary_dim), int(fallback_dim), int(maximum_dim))
        if candidate_dimensions is None
        else tuple(int(value) for value in candidate_dimensions)
    )
    dimensions = tuple(dict.fromkeys(configured_dimensions))
    if not dimensions or any(value <= 0 for value in dimensions):
        raise ValueError("PCA candidate dimensions must be a non-empty sequence of positive integers.")
    for attempt_index, dimension in enumerate(dimensions):
        print(
            f"[pca-validation] candidate={dimension} "
            f"attempt={attempt_index + 1}/{len(dimensions)}",
            flush=True,
        )
        try:
            pca = fit_train_only_pca(
                residuals,
                sample_ids,
                n_components=dimension,
                backend=pca_backend,
                batch_size=batch_size,
                seed=seed,
                allow_numpy_test_fallback=allow_test_fallbacks,
            )
        except ValueError as exc:
            if attempt_index == len(dimensions) - 1:
                raise CompressionValidationError(
                    f"Lower PCA dimensions failed O{overlap_k}; PCA-{dimension} cannot be fitted: {exc}"
                ) from exc
            raise
        projected = pca.transform(raw_validation, scale=False)
        overlap = knn_overlap(
            raw_validation,
            projected,
            k=int(overlap_k),
            backend=index_backend,
            allow_exact_fallback=allow_test_fallbacks,
        )
        attempts.append((dimension, overlap))
        print(
            f"[pca-validation] candidate={dimension} O{overlap_k}={overlap:.6f} "
            f"required={minimum_overlap:.6f}",
            flush=True,
        )
        pca = FrozenPCA(
            mean=pca.mean,
            components=pca.components,
            global_rms=pca.global_rms,
            sample_id_sha256=pca.sample_id_sha256,
            fit_count=pca.fit_count,
            backend=pca.backend,
            overlap_at_16=overlap,
            validation_sample_count=validation_count,
        )
        if overlap >= float(minimum_overlap):
            return pca
    raise CompressionValidationError(
        f"PCA compression rejected for candidates {dimensions}: {attempts}. "
        f"Required O{overlap_k} >= {minimum_overlap:.2f}; "
        "the threshold is locked and must not be reduced."
    )


def mean_huber_distance(anchor: Any, candidates: Any, *, delta: float = 1.0) -> np.ndarray:
    """FP32 mean Huber dissimilarity over globally RMS-scaled descriptors."""

    if float(delta) != 1.0:
        raise ValueError("Locked V1 requires Huber delta=1.0.")
    anchor_array = np.asarray(anchor, dtype=np.float32)
    candidate_array = _float32_2d(candidates, name="candidates")
    if anchor_array.shape != (candidate_array.shape[1],):
        raise ValueError("Anchor/candidate descriptor dimensions differ.")
    absolute = np.abs(candidate_array - anchor_array[None, :], dtype=np.float32)
    huber = np.where(absolute <= 1.0, 0.5 * absolute * absolute, absolute - 0.5).astype(np.float32)
    return np.mean(huber, axis=1, dtype=np.float32)


def compatible_pool_ids(
    effect_families: Sequence[Any],
    *,
    views: Sequence[Any] | None = None,
    horizons: Sequence[Any] | None = None,
) -> np.ndarray:
    """Create collision-safe exact (family, view, horizon) compatibility keys."""

    family = _strings(effect_families, name="effect_families")
    count = len(family)
    view = _strings(views if views is not None else ["single_view"] * count, name="views")
    horizon = _strings(horizons if horizons is not None else ["single_horizon"] * count, name="horizons")
    if len(view) != count or len(horizon) != count:
        raise ValueError("effect_families, views, and horizons must have equal length.")
    return np.asarray(
        [json.dumps([f, v, h], separators=(",", ":"), ensure_ascii=False) for f, v, h in zip(family, view, horizon)],
        dtype=np.str_,
    )


def episode_keys(source_ids: Sequence[Any], episode_ids: Sequence[Any]) -> np.ndarray:
    source = _strings(source_ids, name="source_ids")
    episode = _strings(episode_ids, name="episode_ids")
    if len(source) != len(episode):
        raise ValueError("source_ids and episode_ids must have equal length.")
    return np.asarray(
        [json.dumps([s, e], separators=(",", ":"), ensure_ascii=False) for s, e in zip(source, episode)], dtype=np.str_
    )


def _rank_band(sorted_candidates: np.ndarray, low: float, high: float) -> np.ndarray:
    count = len(sorted_candidates)
    start = int(math.ceil(float(low) * count))
    stop = int(math.ceil(float(high) * count))
    return sorted_candidates[start:stop]


class TeacherGraph:
    def __init__(
        self,
        *,
        sample_ids: Sequence[Any],
        descriptors: Any,
        pool_ids: Sequence[Any],
        episode_keys: Sequence[Any],
        start_steps: Sequence[int],
        end_steps: Sequence[int],
        knn_indices: Any,
        knn_distances: Any,
        middle_offsets: Any,
        middle_indices: Any,
        far_offsets: Any,
        far_indices: Any,
        global_rms: float,
        pca_sample_id_sha256: str,
    ) -> None:
        self.sample_ids = _strings(sample_ids, name="sample_ids")
        self.descriptors = _float32_2d(descriptors, name="descriptors")
        self.pool_ids = _strings(pool_ids, name="pool_ids")
        self.episode_keys = _strings(episode_keys, name="episode_keys")
        self.start_steps = np.asarray(start_steps, dtype=np.int64)
        self.end_steps = np.asarray(end_steps, dtype=np.int64)
        self.knn_indices = np.asarray(knn_indices, dtype=np.int64)
        self.knn_distances = np.asarray(knn_distances, dtype=np.float32)
        self.middle_offsets = np.asarray(middle_offsets, dtype=np.int64)
        self.middle_indices = np.asarray(middle_indices, dtype=np.int64)
        self.far_offsets = np.asarray(far_offsets, dtype=np.int64)
        self.far_indices = np.asarray(far_indices, dtype=np.int64)
        self.global_rms = float(global_rms)
        self.pca_sample_id_sha256 = str(pca_sample_id_sha256)
        self._validate()

    def _validate(self) -> None:
        count = len(self.sample_ids)
        if len(set(self.sample_ids.tolist())) != count:
            raise ArtifactValidationError("Teacher graph sample_ids are not unique.")
        for name, value in (
            ("descriptors", self.descriptors),
            ("pool_ids", self.pool_ids),
            ("episode_keys", self.episode_keys),
            ("start_steps", self.start_steps),
            ("end_steps", self.end_steps),
            ("knn_indices", self.knn_indices),
            ("knn_distances", self.knn_distances),
        ):
            if len(value) != count:
                raise ArtifactValidationError(f"Teacher graph {name} length does not match sample_ids.")
        if self.knn_indices.ndim != 2 or self.knn_distances.shape != self.knn_indices.shape:
            raise ArtifactValidationError("Teacher kNN indices/distances shapes are invalid.")
        if self.middle_offsets.shape != (count + 1,) or self.far_offsets.shape != (count + 1,):
            raise ArtifactValidationError("Teacher rank-band CSR offsets are invalid.")
        if self.middle_offsets[-1] != len(self.middle_indices) or self.far_offsets[-1] != len(self.far_indices):
            raise ArtifactValidationError("Teacher rank-band CSR payload is truncated.")
        if np.any(self.knn_indices < 0) or np.any(self.knn_indices >= count):
            raise ArtifactValidationError("Teacher kNN index is out of range.")
        if not np.isfinite(self.global_rms) or self.global_rms <= 0:
            raise ArtifactValidationError("Teacher global RMS is invalid.")

    @property
    def k_effect(self) -> int:
        return int(self.knn_indices.shape[1])

    def middle_candidates(self, anchor: int) -> np.ndarray:
        return self.middle_indices[self.middle_offsets[anchor] : self.middle_offsets[anchor + 1]]

    def far_candidates(self, anchor: int) -> np.ndarray:
        return self.far_indices[self.far_offsets[anchor] : self.far_offsets[anchor + 1]]

    def distances(self, anchor: int, candidates: Sequence[int]) -> np.ndarray:
        candidate_array = np.asarray(candidates, dtype=np.int64)
        return mean_huber_distance(self.descriptors[int(anchor)], self.descriptors[candidate_array])

    @staticmethod
    def _anchor_rng(seed: int, epoch: int, sample_id: str, band: str) -> np.random.Generator:
        payload = f"{seed}|{epoch}|{sample_id}|{band}".encode("utf-8")
        generated_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="little")
        return np.random.default_rng(generated_seed)

    def sample_mid_far(
        self,
        anchor_indices: Sequence[int],
        *,
        epoch: int,
        middle_count: int = 8,
        far_count: int = 8,
        seed: int = 20260803,
    ) -> tuple[np.ndarray, np.ndarray]:
        anchors = np.asarray(anchor_indices, dtype=np.int64)
        middle = np.empty((len(anchors), int(middle_count)), dtype=np.int64)
        far = np.empty((len(anchors), int(far_count)), dtype=np.int64)
        for row, anchor in enumerate(anchors.tolist()):
            middle_pool = self.middle_candidates(anchor)
            far_pool = self.far_candidates(anchor)
            if len(middle_pool) < middle_count or len(far_pool) < far_count:
                raise ValueError(f"Anchor {self.sample_ids[anchor]} has an undersized middle/far rank band.")
            middle[row] = self._anchor_rng(seed, epoch, str(self.sample_ids[anchor]), "middle").choice(
                middle_pool, size=int(middle_count), replace=False
            )
            far[row] = self._anchor_rng(seed, epoch, str(self.sample_ids[anchor]), "far").choice(
                far_pool, size=int(far_count), replace=False
            )
        return middle, far

    def save(self, path: str | Path) -> None:
        metadata = {
            "schema": TEACHER_GRAPH_SCHEMA,
            "version": ARTIFACT_VERSION,
            "sample_count": len(self.sample_ids),
            "sample_id_sha256": sample_id_digest(self.sample_ids),
            "pca_sample_id_sha256": self.pca_sample_id_sha256,
            "descriptor_dim": int(self.descriptors.shape[1]),
            "global_rms": self.global_rms,
            "distance": "mean_huber_delta_1",
            "effect_knn": self.k_effect,
            "middle_quantile": [0.30, 0.60],
            "far_quantile": [0.70, 0.95],
            "same_episode_overlap_excluded": True,
        }
        _atomic_savez(
            path,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            sample_ids=self.sample_ids,
            descriptors=self.descriptors,
            pool_ids=self.pool_ids,
            episode_keys=self.episode_keys,
            start_steps=self.start_steps,
            end_steps=self.end_steps,
            knn_indices=self.knn_indices,
            knn_distances=self.knn_distances,
            middle_offsets=self.middle_offsets,
            middle_indices=self.middle_indices,
            far_offsets=self.far_offsets,
            far_indices=self.far_indices,
        )

    @classmethod
    def load(cls, path: str | Path, *, expected_sample_ids: Sequence[Any]) -> "TeacherGraph":
        with np.load(path, allow_pickle=False) as archive:
            metadata = _metadata_from_archive(archive, TEACHER_GRAPH_SCHEMA)
            ids = archive["sample_ids"].astype(np.str_)
            expected = _strings(expected_sample_ids, name="expected_sample_ids")
            if not np.array_equal(ids, expected):
                raise ArtifactValidationError(
                    "Teacher graph sample_id order does not match the active train manifest; refusing unsafe load."
                )
            if metadata.get("sample_id_sha256") != sample_id_digest(ids):
                raise ArtifactValidationError("Teacher graph sample_id digest is corrupt.")
            return cls(
                sample_ids=ids,
                descriptors=archive["descriptors"],
                pool_ids=archive["pool_ids"],
                episode_keys=archive["episode_keys"],
                start_steps=archive["start_steps"],
                end_steps=archive["end_steps"],
                knn_indices=archive["knn_indices"],
                knn_distances=archive["knn_distances"],
                middle_offsets=archive["middle_offsets"],
                middle_indices=archive["middle_indices"],
                far_offsets=archive["far_offsets"],
                far_indices=archive["far_indices"],
                global_rms=float(metadata["global_rms"]),
                pca_sample_id_sha256=str(metadata["pca_sample_id_sha256"]),
            )


def build_teacher_graph(
    *,
    sample_ids: Sequence[Any],
    descriptors: Any,
    pool_ids: Sequence[Any],
    episode_keys: Sequence[Any],
    start_steps: Sequence[int],
    end_steps: Sequence[int],
    global_rms: float,
    pca_sample_id_sha256: str,
    effect_knn: int = 16,
    middle_quantile: tuple[float, float] = (0.30, 0.60),
    far_quantile: tuple[float, float] = (0.70, 0.95),
    minimum_middle_count: int = 8,
    minimum_far_count: int = 8,
    backend: str = "numpy",
    anchor_batch_size: int = 32,
    candidate_batch_size: int = 1024,
) -> TeacherGraph:
    """Build exact mean-Huber neighbours and rank bands inside each pool."""

    if tuple(middle_quantile) != (0.30, 0.60) or tuple(far_quantile) != (0.70, 0.95):
        raise ValueError("Locked V1 rank bands are middle=[.30,.60], far=[.70,.95].")
    ids = _strings(sample_ids, name="sample_ids")
    features = _float32_2d(descriptors, name="descriptors")
    pools = _strings(pool_ids, name="pool_ids")
    episodes = _strings(episode_keys, name="episode_keys")
    starts = np.asarray(start_steps, dtype=np.int64)
    ends = np.asarray(end_steps, dtype=np.int64)
    count = len(ids)
    for name, values in (
        ("descriptors", features),
        ("pool_ids", pools),
        ("episode_keys", episodes),
        ("start_steps", starts),
        ("end_steps", ends),
    ):
        if len(values) != count:
            raise ValueError(f"{name} length must match sample_ids.")

    knn_indices = np.empty((count, int(effect_knn)), dtype=np.int64)
    knn_distances = np.empty((count, int(effect_knn)), dtype=np.float32)
    middle_lists: list[np.ndarray] = []
    far_lists: list[np.ndarray] = []
    members_by_pool = {pool: np.flatnonzero(pools == pool).astype(np.int64) for pool in sorted(set(pools.tolist()))}

    if backend == "torch_cuda":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError("torch_cuda teacher graph backend requires PyTorch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("torch_cuda teacher graph backend requires CUDA")
        if anchor_batch_size <= 0 or candidate_batch_size <= 0:
            raise ValueError("GPU graph batch sizes must be positive")

        device = torch.device("cuda")
        features_gpu = torch.from_numpy(features).to(device)
        middle_by_anchor: list[np.ndarray | None] = [None] * count
        far_by_anchor: list[np.ndarray | None] = [None] * count
        pool_count = len(members_by_pool)
        for pool_number, (pool_id, members) in enumerate(members_by_pool.items(), start=1):
            print(
                f"[teacher-graph] pool={pool_number}/{pool_count} "
                f"members={len(members)} id={pool_id}",
                flush=True,
            )
            pool_features = features_gpu[torch.from_numpy(members).to(device)]
            pool_size = len(members)
            for anchor_start in range(0, pool_size, int(anchor_batch_size)):
                anchor_stop = min(pool_size, anchor_start + int(anchor_batch_size))
                anchors = members[anchor_start:anchor_stop]
                anchor_features = features_gpu[torch.from_numpy(anchors).to(device)]
                distances_gpu = torch.empty((len(anchors), pool_size), device=device, dtype=torch.float32)
                for candidate_start in range(0, pool_size, int(candidate_batch_size)):
                    candidate_stop = min(pool_size, candidate_start + int(candidate_batch_size))
                    absolute = torch.abs(
                        anchor_features[:, None, :] - pool_features[None, candidate_start:candidate_stop, :]
                    )
                    huber = torch.where(absolute <= 1.0, 0.5 * absolute.square(), absolute - 0.5)
                    distances_gpu[:, candidate_start:candidate_stop] = huber.mean(dim=-1)

                eligible = np.ones((len(anchors), pool_size), dtype=bool)
                for row, anchor in enumerate(anchors.tolist()):
                    eligible[row] &= ~windows_overlap(anchor, members, episodes, starts, ends)
                    eligible[row, members == anchor] = False
                    if int(np.sum(eligible[row])) < effect_knn:
                        raise ValueError(
                            f"Compatible pool {pool_id!r} for anchor {ids[anchor]!r} has too few candidates."
                        )
                eligible_gpu = torch.from_numpy(eligible).to(device)
                distances_gpu.masked_fill_(~eligible_gpu, float("inf"))
                order = torch.argsort(distances_gpu, dim=1, stable=True)
                top_local = order[:, :effect_knn]
                top_distances = torch.gather(distances_gpu, 1, top_local).cpu().numpy()
                order_cpu = order.cpu().numpy()
                for row, anchor in enumerate(anchors.tolist()):
                    valid_count = int(np.sum(eligible[row]))
                    sorted_candidates = members[order_cpu[row, :valid_count]]
                    knn_indices[anchor] = sorted_candidates[:effect_knn]
                    knn_distances[anchor] = top_distances[row]
                    middle = _rank_band(sorted_candidates, *middle_quantile)
                    far = _rank_band(sorted_candidates, *far_quantile)
                    if len(middle) < minimum_middle_count or len(far) < minimum_far_count:
                        raise ValueError(
                            f"Anchor {ids[anchor]!r} rank bands are too small: middle={len(middle)}, far={len(far)}."
                        )
                    middle_by_anchor[anchor] = middle
                    far_by_anchor[anchor] = far
        if any(value is None for value in middle_by_anchor) or any(value is None for value in far_by_anchor):
            raise RuntimeError("GPU teacher graph did not process every anchor")
        middle_lists = [value for value in middle_by_anchor if value is not None]
        far_lists = [value for value in far_by_anchor if value is not None]

    elif backend != "numpy":
        raise ValueError("teacher graph backend must be 'numpy' or 'torch_cuda'")

    else:
        for anchor in range(count):
            candidates = members_by_pool[str(pools[anchor])]
            excluded = windows_overlap(anchor, candidates, episodes, starts, ends)
            candidates = candidates[~excluded]
            candidates = candidates[candidates != anchor]
            if len(candidates) < effect_knn:
                raise ValueError(
                    f"Compatible pool for anchor {ids[anchor]!r} has only {len(candidates)} valid candidates; "
                    f"effect_knn={effect_knn}."
                )
            distances = mean_huber_distance(features[anchor], features[candidates])
            order = np.lexsort((candidates, distances))
            sorted_candidates = candidates[order]
            sorted_distances = distances[order]
            knn_indices[anchor] = sorted_candidates[:effect_knn]
            knn_distances[anchor] = sorted_distances[:effect_knn]
            middle = _rank_band(sorted_candidates, *middle_quantile)
            far = _rank_band(sorted_candidates, *far_quantile)
            if len(middle) < minimum_middle_count or len(far) < minimum_far_count:
                raise ValueError(
                    f"Anchor {ids[anchor]!r} rank bands are too small: middle={len(middle)}, far={len(far)}."
                )
            middle_lists.append(middle)
            far_lists.append(far)

    middle_offsets = np.zeros(count + 1, dtype=np.int64)
    far_offsets = np.zeros(count + 1, dtype=np.int64)
    for index in range(count):
        middle_offsets[index + 1] = middle_offsets[index] + len(middle_lists[index])
        far_offsets[index + 1] = far_offsets[index] + len(far_lists[index])
    middle_indices = np.concatenate(middle_lists).astype(np.int64, copy=False)
    far_indices = np.concatenate(far_lists).astype(np.int64, copy=False)
    return TeacherGraph(
        sample_ids=ids,
        descriptors=features,
        pool_ids=pools,
        episode_keys=episodes,
        start_steps=starts,
        end_steps=ends,
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        middle_offsets=middle_offsets,
        middle_indices=middle_indices,
        far_offsets=far_offsets,
        far_indices=far_indices,
        global_rms=float(global_rms),
        pca_sample_id_sha256=str(pca_sample_id_sha256),
    )
