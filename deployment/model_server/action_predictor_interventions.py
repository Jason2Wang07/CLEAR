"""Inference-only interventions on the dense action predictor's inputs."""

import logging


def configure_action_predictor_future_input(policy, mode: str) -> None:
    if mode not in {"predicted", "current"}:
        raise ValueError(f"Unknown action predictor future input: {mode!r}")
    if hasattr(policy, "_action_predictor_future_input"):
        raise RuntimeError("Action predictor intervention was already configured")
    if mode == "predicted":
        policy._action_predictor_future_input = mode
        return

    backend = getattr(policy, "policy_backend", None)
    predictor = getattr(backend, "paired_latent_predictor", None)
    if (
        getattr(backend, "stage2_arm", None) != "p2"
        or getattr(predictor, "action_distribution", None) != "dense_transition_gaussian_v3"
    ):
        raise ValueError("Future-input intervention requires a P2 dense action predictor")
    head = predictor.action_head
    if policy.training or head.training:
        raise ValueError("Future-input intervention is inference-only; call eval() first")

    calls = 0

    def replace_future(module, args, kwargs):
        nonlocal calls
        if policy.training or module.training:
            raise RuntimeError("Future-input intervention cannot run during training")
        current = kwargs["current_visual"]
        future = kwargs["future_visual"]
        if current.shape != future.shape:
            raise ValueError("Current and future visual shapes must match")
        # Replace only this head's argument, never the shared future tensor.
        kwargs = dict(kwargs)
        kwargs["future_visual"] = current
        calls += 1
        if calls == 1:
            logging.info(
                "[INTERVENTION_APPLIED] action_predictor future=current; "
                "raw_delta=0; world_and_flow_future_unchanged; shape=%s",
                tuple(current.shape),
            )
        return args, kwargs

    policy._action_predictor_future_hook = head.register_forward_pre_hook(
        replace_future, with_kwargs=True
    )
    policy._action_predictor_future_input = mode
    logging.info("[INTERVENTION_CONFIGURED] action_predictor_future_input=%s", mode)
