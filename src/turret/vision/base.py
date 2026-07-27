from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from turret.vision.types import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> tuple[Detection | None, np.ndarray]:
        """Return (best target or None, debug image).

        The debug image is whatever's useful for `--show-mask`: a binary
        mask for HSV, or a candidate-box overlay for ML backends. Either a
        (H,W) or (H,W,3) array renders fine via cv2.imshow.
        """
