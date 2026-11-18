import unittest

from latent_action_model.stage1.checkpoint_selection import (
    CheckpointMetrics,
    ReleaseBaseline,
    assess_checkpoint,
    geometry_score,
    select_best_checkpoint,
)


def _release():
    return ReleaseBaseline(
        lwm_loss=1.0,
        state_loss=2.0,
        effective_rank=8.0,
        covariance_trace=10.0,
        pairwise_median_distance=4.0,
    )


def _checkpoint(checkpoint_id="checkpoint", **overrides):
    values = {
        "checkpoint_id": checkpoint_id,
        "lwm_loss": 1.0,
        "state_loss": 2.0,
        "effective_rank": 6.0,
        "covariance_trace": 6.0,
        "pairwise_median_distance": 3.0,
        "active_dimensions": 8,
        "mean_kl_per_dim": 0.02,
        "shuffle_gain": 0.12,
        "overlap_at_16": 0.7,
        "trustworthiness_at_16": 0.8,
        "rho_local": 0.4,
    }
    values.update(overrides)
    return CheckpointMetrics(**values)


class CheckpointGatesTest(unittest.TestCase):
    def test_all_boundary_values_pass(self):
        checkpoint = _checkpoint(
            lwm_loss=1.05,
            state_loss=2.2,
            effective_rank=5.6,
            covariance_trace=5.0,
            pairwise_median_distance=2.8,
            shuffle_gain=0.10,
        )
        assessment = assess_checkpoint(checkpoint, _release())

        self.assertTrue(assessment.eligible)
        self.assertEqual(assessment.failed_gates, ())
        self.assertAlmostEqual(assessment.release_comparison.effective_rank_floor, 5.6)
        self.assertAlmostEqual(assessment.release_comparison.covariance_trace_ratio, 0.5)
        self.assertAlmostEqual(assessment.release_comparison.pairwise_median_distance_ratio, 0.7)

    def test_every_failed_gate_is_reported(self):
        checkpoint = _checkpoint(
            lwm_loss=1.051,
            state_loss=2.201,
            effective_rank=3.9,
            covariance_trace=4.9,
            pairwise_median_distance=2.7,
            active_dimensions=0,
            mean_kl_per_dim=0.0,
            shuffle_gain=0.0,
        )
        assessment = assess_checkpoint(checkpoint, _release())

        self.assertFalse(assessment.eligible)
        self.assertEqual(
            set(assessment.failed_gates),
            {
                "lwm_vs_release",
                "state_vs_release",
                "effective_rank",
                "shuffle_gain",
                "mean_contraction_trace",
                "mean_contraction_distance",
                "posterior_collapse",
            },
        )
        self.assertTrue(assessment.posterior_collapsed)

    def test_effective_rank_uses_absolute_floor_when_release_is_small(self):
        release = ReleaseBaseline(
            lwm_loss=1.0,
            state_loss=1.0,
            effective_rank=3.0,
            covariance_trace=2.0,
            pairwise_median_distance=1.0,
        )
        assessment = assess_checkpoint(_checkpoint(effective_rank=3.99), release)
        self.assertIn("effective_rank", assessment.failed_gates)
        self.assertEqual(assessment.release_comparison.effective_rank_floor, 4.0)

    def test_shuffle_ci_is_recorded_but_not_a_hard_gate(self):
        assessment = assess_checkpoint(
            _checkpoint(shuffle_gain=0.11, shuffle_ci_low=-0.02, shuffle_ci_high=0.2),
            _release(),
        )
        self.assertTrue(assessment.eligible)

    def test_invalid_nan_and_release_ratio_denominators_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            assess_checkpoint(_checkpoint(rho_local=float("nan")), _release())
        bad_release = ReleaseBaseline(
            lwm_loss=1.0,
            state_loss=1.0,
            effective_rank=4.0,
            covariance_trace=0.0,
            pairwise_median_distance=1.0,
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            assess_checkpoint(_checkpoint(), bad_release)


class CheckpointSelectionTest(unittest.TestCase):
    def test_geometry_score_formula(self):
        score = geometry_score(overlap_at_16=0.6, trustworthiness_at_16=0.8, rho_local=0.2)
        self.assertAlmostEqual(score, 0.5 * 0.6 + 0.25 * 0.8 + 0.25 * 0.6)

    def test_highest_geometry_wins_and_lwm_breaks_exact_tie(self):
        lower_geometry = _checkpoint("lower-geometry", overlap_at_16=0.6, lwm_loss=0.7)
        tied_higher_lwm = _checkpoint("tie-higher-lwm", overlap_at_16=0.8, lwm_loss=1.0)
        tied_lower_lwm = _checkpoint("tie-lower-lwm", overlap_at_16=0.8, lwm_loss=0.9)
        selection = select_best_checkpoint(
            [lower_geometry, tied_higher_lwm, tied_lower_lwm],
            _release(),
        )

        self.assertIsNotNone(selection.selected)
        self.assertEqual(selection.selected.metrics.checkpoint_id, "tie-lower-lwm")
        self.assertEqual(
            [item.metrics.checkpoint_id for item in selection.ranked_eligible],
            ["tie-lower-lwm", "tie-higher-lwm", "lower-geometry"],
        )

    def test_ineligible_checkpoint_never_wins_and_none_if_all_fail(self):
        best_but_invalid = _checkpoint("invalid", overlap_at_16=1.0, lwm_loss=1.2)
        valid = _checkpoint("valid", overlap_at_16=0.5)
        selection = select_best_checkpoint([best_but_invalid, valid], _release())
        self.assertEqual(selection.selected.metrics.checkpoint_id, "valid")

        all_failed = select_best_checkpoint([best_but_invalid], _release())
        self.assertIsNone(all_failed.selected)
        self.assertEqual(all_failed.ranked_eligible, ())

    def test_duplicate_checkpoint_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            select_best_checkpoint([_checkpoint("same"), _checkpoint("same")], _release())


if __name__ == "__main__":
    unittest.main()
