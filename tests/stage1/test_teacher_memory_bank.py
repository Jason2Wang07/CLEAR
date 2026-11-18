from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from latent_action_model.stage1.memory_bank import (
    ArtifactValidationError,
    LatentMemoryBank,
    Stage1NeighborProvider,
)
from latent_action_model.stage1.teacher import (
    FrozenPCA,
    TeacherGraph,
    build_teacher_graph,
    compatible_pool_ids,
    episode_keys,
    final_post_ln_patch_tokens,
    fit_train_only_pca,
    fit_validated_pca,
    mean_huber_distance,
    patch_l2_residual,
    patch_mean_context,
)


def synthetic_metadata(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = np.asarray([f"sample-{index:03d}" for index in range(count)])
    families = ["rigid"] * (count - 16) + ["articulated"] * 16
    pools = compatible_pool_ids(families, views=["front"] * count, horizons=["1.6"] * count)
    sources = ["source"] * count
    episodes = list(range(count))
    # Samples 0/1 overlap within one episode; sample 2 is same episode but non-overlapping.
    episodes[1] = episodes[0]
    episodes[2] = episodes[0]
    episode_key = episode_keys(sources, episodes)
    starts = np.zeros(count, dtype=np.int64)
    ends = np.full(count, 10, dtype=np.int64)
    starts[2] = 20
    ends[2] = 30
    return sample_ids, pools, episode_key, starts, ends


class ResidualAndPCATest(unittest.TestCase):
    def test_final_tokens_and_per_patch_residual(self) -> None:
        hidden = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
        patches = final_post_ln_patch_tokens(hidden)
        self.assertEqual(patches.shape, (2, 3, 3))
        np.testing.assert_array_equal(patches, hidden[:, 5:, :])

        start = np.asarray([[[3.0, 0.0], [0.0, 2.0]]], dtype=np.float32)
        end = np.asarray([[[0.0, 4.0], [0.0, 6.0]]], dtype=np.float32)
        residual = patch_l2_residual(start, end)
        # First patch changes x -> y; second patch has identical direction despite scale.
        np.testing.assert_allclose(residual, [[-1.0, 1.0, 0.0, 0.0]], atol=1e-6)
        self.assertNotAlmostEqual(float(np.linalg.norm(residual)), 1.0)  # no sample-wise normalization
        context = patch_mean_context(start)
        np.testing.assert_allclose(np.linalg.norm(context, axis=1), [1.0], atol=1e-6)
        np.testing.assert_allclose(context, [[2**-0.5, 2**-0.5]], atol=1e-6)

    def test_huber_delta_one(self) -> None:
        anchor = np.zeros(3, dtype=np.float32)
        candidate = np.asarray([[0.5, 1.0, 2.0]], dtype=np.float32)
        expected = (0.5 * 0.5**2 + 0.5 * 1.0**2 + (2.0 - 0.5)) / 3.0
        np.testing.assert_allclose(mean_huber_distance(anchor, candidate), [expected], rtol=1e-6)

    def test_train_global_rms_is_scalar_and_persistence_checks_ids(self) -> None:
        rng = np.random.default_rng(7)
        residuals = rng.normal(size=(40, 8)).astype(np.float32)
        sample_ids = [f"id-{index}" for index in range(len(residuals))]
        pca = fit_train_only_pca(
            residuals,
            sample_ids,
            n_components=4,
            backend="numpy",
            allow_numpy_test_fallback=True,
        )
        projected = pca.transform(residuals)
        self.assertAlmostEqual(float(np.sqrt(np.mean(projected * projected))), 1.0, places=5)
        # Scalar scaling keeps PCA dimensions at different variances (no whitening).
        self.assertGreater(float(np.std(np.var(projected, axis=0))), 1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pca.npz"
            pca.save(path)
            loaded = FrozenPCA.load(path, expected_sample_ids=sample_ids)
            np.testing.assert_allclose(loaded.transform(residuals), projected)
            with self.assertRaisesRegex(ArtifactValidationError, "different train sample_id"):
                FrozenPCA.load(path, expected_sample_ids=list(reversed(sample_ids)))

    def test_failed_primary_dimension_upgrades_without_lowering_threshold(self) -> None:
        rng = np.random.default_rng(11)
        residuals = rng.normal(size=(100, 4)).astype(np.float32)
        pca = fit_validated_pca(
            residuals,
            [f"id-{index}" for index in range(len(residuals))],
            candidate_dimensions=(1, 4),
            validation_samples=100,
            overlap_k=5,
            minimum_overlap=0.95,
            pca_backend="numpy",
            index_backend="exact",
            allow_test_fallbacks=True,
        )
        self.assertEqual(pca.output_dim, 4)
        self.assertGreaterEqual(float(pca.overlap_at_16), 0.95)

    def test_candidate_dimension_ladder_must_be_nonempty_and_positive(self) -> None:
        residuals = np.eye(20, dtype=np.float32)
        sample_ids = [f"id-{index}" for index in range(len(residuals))]
        for dimensions in ((), (0, 4)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "non-empty sequence of positive integers"):
                    fit_validated_pca(
                        residuals,
                        sample_ids,
                        candidate_dimensions=dimensions,
                        validation_samples=20,
                        overlap_k=4,
                        pca_backend="numpy",
                        index_backend="exact",
                        allow_test_fallbacks=True,
                    )


class GraphAndMemoryBankTest(unittest.TestCase):
    def setUp(self) -> None:
        self.count = 64
        self.ids, self.pools, self.episodes, self.starts, self.ends = synthetic_metadata(self.count)
        rng = np.random.default_rng(19)
        self.descriptors = rng.normal(size=(self.count, 6)).astype(np.float32)
        # Make the overlapping sample the closest raw descriptor to anchor 0.
        self.descriptors[1] = self.descriptors[0]
        # A non-overlapping window from that episode is nearly as close and remains valid.
        self.descriptors[2] = self.descriptors[0] + np.float32(1.0e-3)
        self.graph = build_teacher_graph(
            sample_ids=self.ids,
            descriptors=self.descriptors,
            pool_ids=self.pools,
            episode_keys=self.episodes,
            start_steps=self.starts,
            end_steps=self.ends,
            global_rms=1.0,
            pca_sample_id_sha256="synthetic",
            effect_knn=4,
            minimum_middle_count=2,
            minimum_far_count=2,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_graph_backend_matches_numpy(self) -> None:
        cuda_graph = build_teacher_graph(
            sample_ids=self.ids,
            descriptors=self.descriptors,
            pool_ids=self.pools,
            episode_keys=self.episodes,
            start_steps=self.starts,
            end_steps=self.ends,
            global_rms=1.0,
            pca_sample_id_sha256="synthetic",
            effect_knn=4,
            minimum_middle_count=2,
            minimum_far_count=2,
            backend="torch_cuda",
            anchor_batch_size=8,
            candidate_batch_size=16,
        )
        np.testing.assert_array_equal(cuda_graph.knn_indices, self.graph.knn_indices)
        np.testing.assert_allclose(cuda_graph.knn_distances, self.graph.knn_distances, atol=1e-6)
        for anchor in range(self.count):
            np.testing.assert_array_equal(cuda_graph.middle_candidates(anchor), self.graph.middle_candidates(anchor))
            np.testing.assert_array_equal(cuda_graph.far_candidates(anchor), self.graph.far_candidates(anchor))

    def test_graph_excludes_overlap_but_allows_nonoverlap_and_resamples_by_epoch(self) -> None:
        self.assertNotIn(0, self.graph.knn_indices[0])
        self.assertNotIn(1, self.graph.knn_indices[0])
        self.assertTrue(np.all(self.pools[self.graph.knn_indices[0]] == self.pools[0]))
        # Candidate 2 shares the episode but not time; it remains eligible in rank bands/top-k universe.
        eligible = set(self.graph.knn_indices[0].tolist())
        eligible.update(self.graph.middle_candidates(0).tolist())
        eligible.update(self.graph.far_candidates(0).tolist())
        self.assertIn(2, eligible)

        first = self.graph.sample_mid_far([0, 3], epoch=2, middle_count=2, far_count=2, seed=5)
        repeat = self.graph.sample_mid_far([0, 3], epoch=2, middle_count=2, far_count=2, seed=5)
        later = self.graph.sample_mid_far([0, 3], epoch=3, middle_count=2, far_count=2, seed=5)
        np.testing.assert_array_equal(first[0], repeat[0])
        np.testing.assert_array_equal(first[1], repeat[1])
        self.assertTrue(np.any(first[0] != later[0]) or np.any(first[1] != later[1]))

    def test_graph_roundtrip_rejects_sample_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.npz"
            self.graph.save(path)
            loaded = TeacherGraph.load(path, expected_sample_ids=self.ids)
            np.testing.assert_array_equal(loaded.knn_indices, self.graph.knn_indices)
            with self.assertRaisesRegex(ArtifactValidationError, "sample_id order"):
                TeacherGraph.load(path, expected_sample_ids=self.ids[::-1])

    def test_bank_queries_live_mu_and_filters_compatible_overlap(self) -> None:
        latents = np.arange(self.count, dtype=np.float32)[:, None]
        latents = np.concatenate((latents, np.zeros_like(latents)), axis=1)
        # Make overlapping index 1 otherwise the nearest to a live anchor query at zero.
        latents[0, 0] = 0.0
        latents[1, 0] = 0.01
        latents[2, 0] = 0.02
        bank = LatentMemoryBank.initialize_release(
            self.ids,
            latents,
            self.pools,
            self.episodes,
            self.starts,
            self.ends,
            backend="exact",
            allow_exact_fallback=True,
        )
        _, neighbours = bank.query([0], np.asarray([[0.0, 0.0]], dtype=np.float32), k=3)
        self.assertNotIn(0, neighbours[0])
        self.assertNotIn(1, neighbours[0])
        self.assertEqual(int(neighbours[0, 0]), 2)
        np.testing.assert_array_equal(bank._excluded_for_anchor(0), np.asarray([0, 1]))
        self.assertTrue(bank.should_full_refresh(5))
        self.assertFalse(bank.should_full_refresh(6))

        before_index_epoch = bank.index_epoch
        bank.update([3], np.asarray([[999.0, 0.0]], dtype=np.float32))
        self.assertTrue(bank.dirty)
        self.assertEqual(bank.index_epoch, before_index_epoch)
        bank.rebuild_index(epoch=1)
        self.assertFalse(bank.dirty)
        self.assertEqual(bank.index_epoch, 1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            bank.save(path)
            loaded = LatentMemoryBank.load(
                path,
                expected_sample_ids=self.ids,
                backend="exact",
                allow_exact_fallback=True,
            )
            np.testing.assert_allclose(loaded.latents, bank.latents)
            self.assertTrue(loaded.should_full_refresh(5))

            smoke_loaded = LatentMemoryBank.load(
                path,
                expected_sample_ids=self.ids,
                backend="exact",
                allow_exact_fallback=True,
                full_refresh_epochs=(),
            )
            self.assertEqual(smoke_loaded.full_refresh_epochs, ())
            self.assertFalse(smoke_loaded.should_full_refresh(5))
            with self.assertRaisesRegex(ArtifactValidationError, "sample_id order"):
                LatentMemoryBank.load(
                    path,
                    expected_sample_ids=self.ids[::-1],
                    backend="exact",
                    allow_exact_fallback=True,
                )

    def test_provider_returns_deduplicated_shared_candidate_set(self) -> None:
        rng = np.random.default_rng(23)
        bank = LatentMemoryBank.initialize_release(
            self.ids,
            rng.normal(size=(self.count, 3)).astype(np.float32),
            self.pools,
            self.episodes,
            self.starts,
            self.ends,
            backend="exact",
            allow_exact_fallback=True,
        )
        provider = Stage1NeighborProvider(
            self.graph,
            bank,
            k_z=3,
            middle_count=2,
            far_count=2,
            seed=9,
        )
        live = rng.normal(size=(2, 3)).astype(np.float32)
        batch = provider.get([0, 4], live, epoch=2)
        self.assertEqual(batch.candidate_indices.shape, (2, 11))
        self.assertEqual(batch.candidate_bank_mu.shape, (2, 11, 3))
        np.testing.assert_allclose(batch.bandwidth, self.graph.knn_distances[[0, 4], -1] * 0.30)
        for row, anchor in enumerate((0, 4)):
            selected = batch.candidate_indices[row][batch.candidate_mask[row]]
            self.assertEqual(len(selected), len(set(selected.tolist())))
            np.testing.assert_allclose(
                batch.teacher_distances[row][batch.candidate_mask[row]],
                self.graph.distances(anchor, selected),
            )
            self.assertNotIn(anchor, selected)

    def test_provider_applies_explicit_teacher_bandwidth_floor(self) -> None:
        rng = np.random.default_rng(31)
        bank = LatentMemoryBank.initialize_release(
            self.ids,
            rng.normal(size=(self.count, 3)).astype(np.float32),
            self.pools,
            self.episodes,
            self.starts,
            self.ends,
            backend="exact",
            allow_exact_fallback=True,
        )
        graph = self.graph
        graph.knn_distances[0, -1] = 0.0
        provider = Stage1NeighborProvider(
            graph,
            bank,
            k_z=3,
            middle_count=2,
            far_count=2,
            teacher_bandwidth_floor=1.0e-6,
        )

        batch = provider.get([0], bank.latents[[0]], epoch=2)

        np.testing.assert_allclose(batch.bandwidth, [1.0e-6])


if __name__ == "__main__":
    unittest.main()
