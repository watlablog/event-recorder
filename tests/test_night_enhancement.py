from __future__ import annotations

import numpy as np

from event_recorder.config import NightEnhancementConfig
from event_recorder.night_enhancement import NightFrameEnhancer


def test_night_enhancement_disabled_returns_original_frame():
    frame = np.full((2, 2, 3), 20, dtype=np.uint8)
    enhancer = NightFrameEnhancer(
        NightEnhancementConfig(
            enabled=False,
            contrast=1.35,
            brightness=12.0,
            gamma=1.20,
        )
    )

    enhanced = enhancer.apply(frame)

    assert enhancer._gamma_lut is None
    assert enhanced is frame
    assert np.array_equal(enhanced, frame)


def test_night_enhancement_preserves_shape_and_dtype_when_enabled():
    frame = np.full((3, 4, 3), 20, dtype=np.uint8)
    enhancer = NightFrameEnhancer(
        NightEnhancementConfig(
            enabled=True,
            contrast=1.5,
            brightness=10.0,
            gamma=1.0,
        )
    )

    enhanced = enhancer.apply(frame)

    assert enhanced.shape == frame.shape
    assert enhanced.dtype == frame.dtype


def test_night_enhancement_applies_contrast_brightness_and_gamma():
    frame = np.full((1, 1, 3), 20, dtype=np.uint8)
    enhancer = NightFrameEnhancer(
        NightEnhancementConfig(
            enabled=True,
            contrast=2.0,
            brightness=10.0,
            gamma=1.0,
        )
    )
    gamma_enhancer = NightFrameEnhancer(
        NightEnhancementConfig(
            enabled=True,
            contrast=2.0,
            brightness=10.0,
            gamma=1.4,
        )
    )

    enhanced = enhancer.apply(frame)
    gamma_enhanced = gamma_enhancer.apply(frame)

    assert int(enhanced[0, 0, 0]) == 50
    assert int(gamma_enhanced[0, 0, 0]) > 50


def test_gamma_lut_is_created_only_when_gamma_changes_output():
    no_gamma = NightFrameEnhancer(
        NightEnhancementConfig(
            enabled=True,
            contrast=1.0,
            brightness=0.0,
            gamma=1.0,
        )
    )
    with_gamma = NightFrameEnhancer(
        NightEnhancementConfig(
            enabled=True,
            contrast=1.0,
            brightness=0.0,
            gamma=1.2,
        )
    )

    assert no_gamma._gamma_lut is None
    assert with_gamma._gamma_lut is not None
