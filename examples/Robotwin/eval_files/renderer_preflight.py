#!/usr/bin/env python3
"""Fail fast unless the current container can construct a SAPIEN renderer."""

from __future__ import annotations

import gc
import os


def main() -> int:
    try:
        import sapien
    except Exception as exc:
        raise RuntimeError(
            "ROBOTWIN_RUNTIME_PREFLIGHT_FAILED: importing sapien failed"
        ) from exc

    try:
        renderer = sapien.SapienRenderer()
        del renderer
        gc.collect()
    except Exception as exc:
        raise RuntimeError(
            "ROBOTWIN_RENDERER_PREFLIGHT_FAILED: "
            f"SapienRenderer construction failed with VK_ICD_FILENAMES="
            f"{os.getenv('VK_ICD_FILENAMES', '')!r}"
        ) from exc

    print(
        "ROBOTWIN_RENDERER_PREFLIGHT_OK "
        f"cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', '') or 'all'} "
        f"vk_icd={os.getenv('VK_ICD_FILENAMES', '')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
