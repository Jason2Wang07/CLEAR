def __getattr__(name: str):
    if name == "LatentWorldPolicyConfigBuilder":
        from .config_builder import LatentWorldPolicyConfigBuilder

        return LatentWorldPolicyConfigBuilder
    if name == "LatentWorldPolicyInferBatchBuilder":
        from .batch_builder import LatentWorldPolicyInferBatchBuilder

        return LatentWorldPolicyInferBatchBuilder
    if name in {"LatentWorldPolicyComponents", "build_policy_components"}:
        from .runtime import LatentWorldPolicyComponents, build_policy_components

        return {
            "LatentWorldPolicyComponents": LatentWorldPolicyComponents,
            "build_policy_components": build_policy_components,
        }[name]
    if name in {
        "LatentWorldPolicyInferBatch",
        "LatentWorldPolicyInferExample",
        "LatentWorldPolicyTrainBatch",
        "LatentWorldPolicyTrainRawSample",
    }:
        from . import types

        return getattr(types, name)
    if name == "LatentWorldPolicyVLMAdapter":
        from .vlm_adapter import LatentWorldPolicyVLMAdapter

        return LatentWorldPolicyVLMAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LatentWorldPolicyInferBatch",
    "LatentWorldPolicyInferBatchBuilder",
    "LatentWorldPolicyInferExample",
    "LatentWorldPolicyComponents",
    "LatentWorldPolicyConfigBuilder",
    "LatentWorldPolicyTrainBatch",
    "LatentWorldPolicyTrainRawSample",
    "LatentWorldPolicyVLMAdapter",
    "build_policy_components",
]
