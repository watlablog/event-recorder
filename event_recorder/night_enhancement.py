from __future__ import annotations

from dataclasses import replace

import numpy as np

from event_recorder.config import NightEnhancementConfig
from event_recorder.models import FramePacket


class NightFrameEnhancer:
    def __init__(self, config: NightEnhancementConfig) -> None:
        self.config = config
        self._gamma_lut = (
            _build_gamma_lut(config.gamma)
            if config.enabled and _uses_gamma(config.gamma)
            else None
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return frame

        try:
            import cv2
        except ImportError:
            return frame

        enhanced = cv2.convertScaleAbs(
            frame,
            alpha=self.config.contrast,
            beta=self.config.brightness,
        )
        if self._gamma_lut is not None:
            enhanced = cv2.LUT(enhanced, self._gamma_lut)
        return enhanced

    def apply_packet(self, packet: FramePacket) -> FramePacket:
        frame = self.apply(packet.frame)
        if frame is packet.frame:
            return packet
        return replace(packet, frame=frame)


def _build_gamma_lut(gamma: float) -> np.ndarray:
    inverse_gamma = 1.0 / gamma
    return np.array(
        [
            min(255, max(0, int(((value / 255.0) ** inverse_gamma) * 255.0)))
            for value in range(256)
        ],
        dtype=np.uint8,
    )


def _uses_gamma(gamma: float) -> bool:
    return abs(gamma - 1.0) > 1e-6
