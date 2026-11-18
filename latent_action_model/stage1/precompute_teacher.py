"""Offline CLI for the locked stage-1 DINO residual teacher.

Typical usage::

    python -m latent_action_model.stage1.precompute_teacher extract-residuals ...
    python -m latent_action_model.stage1.precompute_teacher fit-pca ...
    python -m latent_action_model.stage1.precompute_teacher build-graph ...

The three artifacts are deliberately separate: the large raw residual memmap
is reusable, while PCA and graph files remain small and versioned.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .memory_bank import sample_id_digest
from .teacher import (
    FrozenPCA,
    ResidualStore,
    ResidualStoreWriter,
    build_teacher_graph,
    compatible_pool_ids,
    episode_keys,
    extract_final_patch_residual_and_context,
    fit_validated_pca,
)


LOCKED_SEED = 20260803
LOCKED_IMAGE_SIZE = 256
LOCKED_PATCH_SIZE = 16


def _residual_store_paths(output: Path, *, resume: bool = False) -> tuple[Path, Path]:
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"immutable residual store already exists: {destination}")
    temporary = destination.with_name(
        f".{destination.name}.work" if resume else f".{destination.name}.{os.getpid()}.tmp"
    )
    if temporary.exists() and not resume:
        raise FileExistsError(f"stale residual-store temporary directory exists: {temporary}")
    return destination, temporary


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(path: str | Path, *, require_train: bool = False) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the stage-1 parquet manifest.") from exc
    table = parquet.read_table(path)
    result = table.to_pydict()
    required = {
        "sample_id",
        "effect_family",
        "source_id",
        "episode_id",
        "start_step",
        "end_step",
        "split",
        "shard_path",
        "shard_row",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}.")
    split_values = set(str(value) for value in result["split"])
    if len(split_values) != 1:
        raise ValueError(f"Manifest must contain exactly one split, got {sorted(split_values)}.")
    if require_train and split_values != {"train"}:
        raise ValueError("Teacher PCA/graph fitting accepts the train manifest only.")
    sample_ids = [str(value) for value in result["sample_id"]]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Manifest sample_id values are not unique.")
    return result


def _resolve_shard(manifest_path: str | Path, value: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        candidate = raw
    else:
        candidate = Path(manifest_path).expanduser().resolve().parent.parent / raw
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate.resolve()


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for residual extraction but is missing from the LaWAM environment. "
            "Install h5py before running extract-residuals."
        ) from exc
    return h5py


def _normalize_images(images: np.ndarray, *, device: str, image_size: int) -> Any:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.ascontiguousarray(images)).to(device=device, dtype=torch.float32).mul_(1.0 / 255.0)
    if tuple(tensor.shape[-2:]) != (int(image_size), int(image_size)):
        tensor = functional.interpolate(
            tensor,
            size=(int(image_size), int(image_size)),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return tensor.sub_(mean).div_(std)


def extract_residuals(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel

    h5py = _require_h5py()
    manifest = _load_manifest(args.manifest)
    sample_ids = [str(value) for value in manifest["sample_id"]]
    image_size = int(args.image_size)
    if image_size % LOCKED_PATCH_SIZE:
        raise ValueError(f"image-size must be divisible by DINO patch size {LOCKED_PATCH_SIZE}.")

    model = AutoModel.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        dtype=torch.float32,
        local_files_only=True,
    ).to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_size = int(getattr(model.config, "hidden_size"))
    patch_count = (image_size // LOCKED_PATCH_SIZE) ** 2
    feature_dim = patch_count * hidden_size
    resume = bool(getattr(args, "resume", False))
    destination, temporary = _residual_store_paths(Path(args.output), resume=resume)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_path = temporary / "extraction_state.json"
    source_metadata = {
        "dino_model": str(Path(args.model).resolve()),
        "dino_layer": "final_post_ln",
        "special_tokens_removed": 5,
        "patch_feature_l2_norm": True,
        "residual_direction": "end_minus_start",
        "residual_sample_l2_norm": False,
        "image_size": image_size,
        "patch_size": LOCKED_PATCH_SIZE,
        "preprocess": "deterministic_resize_then_imagenet_normalize",
        "source_split": str(manifest["split"][0]),
        "start_context_file": "start_context.npy",
        "start_context_dim": hidden_size,
        "start_context": "l2norm(mean(per_patch_l2norm(final_post_ln_start_tokens)))",
    }
    work_contract = {
        "schema": "lawam-stage1-residual-extraction-work-v1",
        "sample_count": len(sample_ids),
        "sample_id_sha256": sample_id_digest(sample_ids),
        "feature_dim": feature_dim,
        "source_metadata": source_metadata,
    }
    try:
        if temporary.exists():
            if not state_path.is_file():
                raise ValueError(f"resumable residual work directory lacks state: {temporary}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if {key: state.get(key) for key in work_contract} != work_contract:
                raise ValueError(f"residual extraction work contract changed: {temporary}")
            state_shards = state.get("completed_shards")
            if not isinstance(state_shards, list) or len(set(state_shards)) != len(state_shards):
                raise ValueError("residual extraction work has invalid completed_shards")
            stored_ids = np.load(temporary / "sample_ids.npy", allow_pickle=False).astype(np.str_)
            if not np.array_equal(stored_ids, np.asarray(sample_ids, dtype=np.str_)):
                raise ValueError("residual extraction work sample IDs differ from the manifest")
            matrix = np.load(temporary / "residuals.npy", mmap_mode="r+", allow_pickle=False)
            context_matrix = np.load(
                temporary / "start_context.npy", mmap_mode="r+", allow_pickle=False
            )
            if matrix.shape != (len(sample_ids), feature_dim) or matrix.dtype != np.float32:
                raise ValueError("residual extraction work matrix contract mismatch")
            if context_matrix.shape != (len(sample_ids), hidden_size) or context_matrix.dtype != np.float32:
                raise ValueError("residual extraction work context contract mismatch")
            completed_shards = set(str(value) for value in state_shards)
            print(
                f"[residuals:{manifest['split'][0]}] resuming completed_shards={len(completed_shards)}",
                flush=True,
            )
        else:
            matrix, _ = ResidualStore.create(
                temporary,
                sample_ids=sample_ids,
                feature_dim=feature_dim,
                source_metadata=source_metadata,
            )
            context_matrix = np.lib.format.open_memmap(
                temporary / "start_context.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(sample_ids), hidden_size),
            )
            completed_shards: set[str] = set()
            _atomic_json(state_path, {**work_contract, "completed_shards": []})

        # Group by shard for sequential HDF5 access while writing back to manifest
        # positions, preserving the exact sample-ID order in the artifact.
        by_shard: dict[str, list[int]] = {}
        for position, shard_path in enumerate(manifest["shard_path"]):
            by_shard.setdefault(str(shard_path), []).append(position)
        unknown_completed = completed_shards.difference(by_shard)
        if unknown_completed:
            raise ValueError(
                f"residual extraction work references unknown shards: {sorted(unknown_completed)[:8]}"
            )
        processed_count = sum(len(by_shard[path]) for path in completed_shards)
        source_split = str(manifest["split"][0])
        with torch.inference_mode():
            for shard_path in sorted(by_shard):
                if shard_path in completed_shards:
                    continue
                positions = np.asarray(by_shard[shard_path], dtype=np.int64)
                resolved_shard = _resolve_shard(args.manifest, shard_path)
                with h5py.File(resolved_shard, "r") as handle:
                    if "rgb_start" not in handle or "rgb_end" not in handle:
                        raise ValueError(f"Shard {resolved_shard} lacks rgb_start/rgb_end datasets.")
                    for offset in range(0, len(positions), int(args.batch_size)):
                        selected_positions = positions[offset : offset + int(args.batch_size)]
                        rows = np.asarray(
                            [manifest["shard_row"][index] for index in selected_positions],
                            dtype=np.int64,
                        )
                        # h5py fancy indexing requires increasing row indices.
                        order = np.argsort(rows, kind="stable")
                        inverse = np.argsort(order, kind="stable")
                        start = np.asarray(handle["rgb_start"][rows[order]], dtype=np.uint8)[inverse]
                        end = np.asarray(handle["rgb_end"][rows[order]], dtype=np.uint8)[inverse]
                        start_pixels = _normalize_images(start, device=args.device, image_size=image_size)
                        end_pixels = _normalize_images(end, device=args.device, image_size=image_size)
                        residual, start_context = extract_final_patch_residual_and_context(
                            model,
                            start_pixels,
                            end_pixels,
                            expected_patch_count=patch_count,
                        )
                        if not torch.isfinite(residual).all() or not torch.isfinite(start_context).all():
                            raise RuntimeError(
                                f"DINO produced NaN/Inf for shard {resolved_shard}, rows {rows.tolist()}"
                            )
                        matrix[selected_positions] = residual.float().cpu().numpy()
                        context_matrix[selected_positions] = start_context.float().cpu().numpy()
                matrix.flush()
                context_matrix.flush()
                completed_shards.add(shard_path)
                _atomic_json(
                    state_path,
                    {**work_contract, "completed_shards": sorted(completed_shards)},
                )
                processed_count += len(positions)
                print(
                    f"[residuals:{source_split}] {processed_count}/{len(sample_ids)} "
                    f"shard={shard_path}",
                    flush=True,
                )
        context_matrix.flush()
        if len(completed_shards) != len(by_shard):
            raise RuntimeError("residual extraction did not commit every source shard")
        writer = ResidualStoreWriter(
            temporary,
            np.asarray(sample_ids, dtype=np.str_),
            matrix,
            source_metadata,
        )
        writer.finalize()
        state_path.unlink()
        ResidualStore(temporary, expected_sample_ids=sample_ids)
        os.replace(temporary, destination)
    except Exception:
        if not resume:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {"residual_store": str(destination), "samples": len(sample_ids), "dim": feature_dim}
        )
    )


def fit_pca(args: argparse.Namespace) -> None:
    store = ResidualStore(args.residual_store)
    if store.metadata.get("source_split") != "train":
        raise ValueError("PCA must be fitted from a train-split residual store.")
    pca = fit_validated_pca(
        store.residuals,
        store.sample_ids,
        candidate_dimensions=args.pca_dimensions,
        validation_samples=2000,
        overlap_k=16,
        minimum_overlap=0.80,
        pca_backend=args.pca_backend,
        index_backend=args.index_backend,
        batch_size=int(args.batch_size),
        seed=LOCKED_SEED,
        allow_test_fallbacks=bool(args.allow_exact_test_fallback),
    )
    pca.save(args.output)
    print(
        json.dumps(
            {
                "pca": str(Path(args.output).resolve()),
                "dimension": pca.output_dim,
                "O16": pca.overlap_at_16,
                "global_rms": pca.global_rms,
            },
            sort_keys=True,
        )
    )


def _manifest_values(manifest: dict[str, list[Any]], column: str, *, fixed: str | None = None) -> list[Any]:
    if column in manifest:
        return manifest[column]
    if fixed is not None:
        return [fixed] * len(manifest["sample_id"])
    raise ValueError(f"Manifest has no {column!r} column and no fixed replacement was supplied.")


def _transform_residual_store(
    pca: FrozenPCA,
    residuals: Any,
    *,
    batch_size: int,
    label: str,
    use_cuda: bool,
) -> np.ndarray:
    descriptors = np.empty((len(residuals), pca.output_dim), dtype=np.float32)
    if not use_cuda:
        for start in range(0, len(residuals), int(batch_size)):
            stop = min(len(residuals), start + int(batch_size))
            descriptors[start:stop] = pca.transform(residuals[start:stop], scale=True)
        return descriptors

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA descriptor projection was requested, but CUDA is unavailable.")
    device = torch.device("cuda")
    mean = torch.from_numpy(pca.mean).to(device)
    components = torch.from_numpy(pca.components).to(device)
    scale = float(pca.global_rms) + 1.0e-6
    with torch.inference_mode():
        for start in range(0, len(residuals), int(batch_size)):
            stop = min(len(residuals), start + int(batch_size))
            batch = torch.from_numpy(np.asarray(residuals[start:stop], dtype=np.float32)).to(device)
            projected = ((batch - mean) @ components.T).div_(scale)
            if not torch.isfinite(projected).all():
                raise RuntimeError(f"PCA descriptor projection produced NaN/Inf for rows {start}:{stop}.")
            descriptors[start:stop] = projected.cpu().numpy()
            if stop == len(residuals) or stop % (int(batch_size) * 16) == 0:
                print(f"[descriptors:{label}] {stop}/{len(residuals)} dim={pca.output_dim}", flush=True)
    return descriptors


def build_graph(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest, require_train=True)
    sample_ids = [str(value) for value in manifest["sample_id"]]
    store = ResidualStore(args.residual_store, expected_sample_ids=sample_ids)
    pca = FrozenPCA.load(args.pca, expected_sample_ids=sample_ids)
    descriptors = _transform_residual_store(
        pca,
        store.residuals,
        batch_size=int(args.batch_size),
        label="train",
        use_cuda=args.backend == "torch_cuda",
    )

    views = _manifest_values(manifest, args.view_column, fixed=args.fixed_view)
    horizons = _manifest_values(manifest, args.horizon_column, fixed=None)
    if args.pool_column not in manifest:
        raise ValueError(
            f"Manifest has no pool column {args.pool_column!r}; "
            f"available columns are {sorted(manifest)}"
        )
    pools = compatible_pool_ids(manifest[args.pool_column], views=views, horizons=horizons)
    episodes = episode_keys(manifest["source_id"], manifest["episode_id"])
    graph = build_teacher_graph(
        sample_ids=sample_ids,
        descriptors=descriptors,
        pool_ids=pools,
        episode_keys=episodes,
        start_steps=manifest["start_step"],
        end_steps=manifest["end_step"],
        global_rms=pca.global_rms,
        pca_sample_id_sha256=pca.sample_id_sha256,
        effect_knn=16,
        middle_quantile=(0.30, 0.60),
        far_quantile=(0.70, 0.95),
        minimum_middle_count=8,
        minimum_far_count=8,
        backend=args.backend,
        anchor_batch_size=int(args.anchor_batch_size),
        candidate_batch_size=int(args.candidate_batch_size),
    )
    graph.save(args.output)
    print(
        json.dumps(
            {
                "teacher_graph": str(Path(args.output).resolve()),
                "samples": len(sample_ids),
                "descriptor_dim": pca.output_dim,
                "compatible_pools": len(set(pools.tolist())),
            },
            sort_keys=True,
        )
    )


def transform_descriptors(args: argparse.Namespace) -> None:
    """Apply the frozen train PCA to any official split residual store."""

    manifest = _load_manifest(args.manifest)
    sample_ids = [str(value) for value in manifest["sample_id"]]
    store = ResidualStore(args.residual_store, expected_sample_ids=sample_ids)
    pca = FrozenPCA.load(args.pca)
    try:
        import torch

        use_cuda = pca.backend == "torch_randomized_cuda" and torch.cuda.is_available()
    except ImportError:
        use_cuda = False
    descriptors = _transform_residual_store(
        pca,
        store.residuals,
        batch_size=int(args.batch_size),
        label=str(manifest["split"][0]),
        use_cuda=use_cuda,
    )
    if store.start_context is None:
        raise ValueError("Residual store has no start-context descriptor; rerun extraction with the current schema.")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "lawam-stage1-split-descriptors-v1",
        "split": str(manifest["split"][0]),
        "sample_count": len(sample_ids),
        "pca_dim": pca.output_dim,
        "pca_train_sample_id_sha256": pca.sample_id_sha256,
    }
    np.savez_compressed(
        destination,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        descriptors=descriptors,
        start_context=np.asarray(store.start_context, dtype=np.float32),
    )
    print(json.dumps({"descriptors": str(destination.resolve()), **metadata}, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-residuals", help="Extract final post-LN DINO patch residuals.")
    extract.add_argument("--manifest", required=True, type=Path)
    extract.add_argument("--model", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--batch-size", default=16, type=int)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--image-size", default=LOCKED_IMAGE_SIZE, type=int)
    extract.add_argument(
        "--resume",
        action="store_true",
        help="Commit progress after each source shard in a stable work directory.",
    )
    extract.set_defaults(handler=extract_residuals)

    pca = subparsers.add_parser("fit-pca", help="Fit train-only PCA-256 and enforce O16 >= .80.")
    pca.add_argument("--residual-store", required=True, type=Path)
    pca.add_argument("--output", required=True, type=Path)
    pca.add_argument(
        "--pca-backend",
        choices=("torch_randomized_cuda", "sklearn_incremental", "sklearn_randomized", "numpy"),
        default="torch_randomized_cuda",
    )
    pca.add_argument("--index-backend", choices=("torch_cuda", "faiss", "exact"), default="torch_cuda")
    pca.add_argument("--batch-size", default=512, type=int)
    pca.add_argument(
        "--pca-dimensions",
        nargs="+",
        type=int,
        default=(256, 512, 1024),
        help="Ordered PCA dimensions to try while keeping the O16 threshold fixed.",
    )
    pca.add_argument(
        "--allow-exact-test-fallback",
        action="store_true",
        help="Enable NumPy SVD/L2 fallbacks for small synthetic tests; never use for the full train set.",
    )
    pca.set_defaults(handler=fit_pca)

    graph = subparsers.add_parser("build-graph", help="Build compatible-pool mean-Huber teacher graph.")
    graph.add_argument("--manifest", required=True, type=Path)
    graph.add_argument("--residual-store", required=True, type=Path)
    graph.add_argument("--pca", required=True, type=Path)
    graph.add_argument("--output", required=True, type=Path)
    graph.add_argument("--batch-size", default=512, type=int)
    graph.add_argument("--view-column", default="view_id")
    graph.add_argument(
        "--fixed-view",
        default="rgb_start_end",
        help="Current V1 manifest has one RGB view but no view_id column; this explicit constant forms the pool key.",
    )
    graph.add_argument("--horizon-column", default="tau")
    graph.add_argument(
        "--pool-column",
        default="effect_family",
        help="Manifest column defining teacher compatibility pools; use teacher_pool for Bridge.",
    )
    graph.add_argument("--backend", choices=("torch_cuda", "numpy"), default="torch_cuda")
    graph.add_argument("--anchor-batch-size", default=32, type=int)
    graph.add_argument("--candidate-batch-size", default=1024, type=int)
    graph.set_defaults(handler=build_graph)

    transform = subparsers.add_parser(
        "transform-descriptors",
        help="Apply the frozen train PCA and package start-context descriptors for validation/test.",
    )
    transform.add_argument("--manifest", required=True, type=Path)
    transform.add_argument("--residual-store", required=True, type=Path)
    transform.add_argument("--pca", required=True, type=Path)
    transform.add_argument("--output", required=True, type=Path)
    transform.add_argument("--batch-size", default=512, type=int)
    transform.set_defaults(handler=transform_descriptors)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
