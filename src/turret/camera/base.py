from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Camera(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Next BGR frame, or None if unavailable."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """(width, height) of delivered frames."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
