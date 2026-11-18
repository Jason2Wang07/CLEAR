import math
import unittest

import torch

from latent_action_model.stage1.losses import (
    linear_neighbor_weight,
    neighborhood_distribution_loss,
    warmup_cosine_multiplier,
)


class Stage1LossTest(unittest.TestCase):
    def test_locked_neighbor_schedule(self):
        steps = 92
        self.assertEqual(linear_neighbor_weight(0, steps_per_epoch=steps), 0.0)
        self.assertEqual(linear_neighbor_weight(steps, steps_per_epoch=steps), 0.0)
        self.assertAlmostEqual(linear_neighbor_weight(2 * steps, steps_per_epoch=steps), 0.0125)
        self.assertAlmostEqual(linear_neighbor_weight(5 * steps, steps_per_epoch=steps), 0.05)

    def test_warmup_cosine_schedule(self):
        self.assertAlmostEqual(
            warmup_cosine_multiplier(0, total_steps=3680, warmup_steps=92),
            1.0 / 92.0,
        )
        self.assertAlmostEqual(
            warmup_cosine_multiplier(92, total_steps=3680, warmup_steps=92),
            1.0,
        )
        self.assertAlmostEqual(
            warmup_cosine_multiplier(3680, total_steps=3680, warmup_steps=92),
            0.1,
        )

    def test_neighbor_loss_is_finite_and_updates_anchor_only(self):
        anchor = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
        candidates = torch.tensor(
            [[[0.1, 0.0], [2.0, 2.0], [9.0, 9.0]], [[1.1, 1.0], [0.0, 0.0], [9.0, 9.0]]],
            requires_grad=True,
        )
        teacher = torch.tensor([[0.1, 1.0, 0.0], [0.1, 1.0, 0.0]])
        mask = torch.tensor([[True, True, False], [True, True, False]])
        out = neighborhood_distribution_loss(
            anchor,
            candidates,
            teacher,
            torch.tensor([1.0, 1.0]),
            student_gamma=1.0,
            candidate_mask=mask,
            expected_latent_dim=2,
        )
        self.assertTrue(math.isfinite(float(out.loss)))
        out.loss.backward()
        self.assertIsNotNone(anchor.grad)
        self.assertIsNone(candidates.grad)
        self.assertGreaterEqual(float(out.teacher_perplexity), 1.0)


if __name__ == "__main__":
    unittest.main()
