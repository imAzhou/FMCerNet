import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class WSIReader(ABC):
    """Abstract interface for vendor-specific whole-slide image readers."""

    def __init__(self, source_path: str) -> None:
        self.source_path = source_path

    @property
    @abstractmethod
    def level_dimensions(self) -> Sequence[tuple[int, int]]:
        """Return ``(width, height)`` for every pyramid level."""

    @property
    @abstractmethod
    def level_downsamples(self) -> Sequence[float]:
        """Return the downsample factor for every pyramid level."""

    @abstractmethod
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Return an RGB uint8 array with shape ``(height, width, 3)``."""

    @abstractmethod
    def close(self) -> None:
        """Release resources owned by the reader."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def open_wsi(source_path: str, reader_spec: str | None = None) -> WSIReader:
    """Instantiate the user-provided WSI reader.

    The reader class is selected with ``FMCERNET_WSI_READER`` or an explicit
    ``reader_spec``. The value must use ``package.module:ReaderClass`` syntax.
    """

    spec = reader_spec or os.environ.get("FMCERNET_WSI_READER")
    if spec is None:
        raise RuntimeError(
            "No WSI reader is configured. Set FMCERNET_WSI_READER to "
            "'package.module:ReaderClass'."
        )
    if spec.count(":") != 1:
        raise ValueError(
            "WSI reader specification must use 'package.module:ReaderClass' syntax."
        )

    module_name, class_name = spec.split(":")
    module = importlib.import_module(module_name)
    reader_class = getattr(module, class_name)
    if not isinstance(reader_class, type) or not issubclass(reader_class, WSIReader):
        raise TypeError(f"{spec} must identify a WSIReader subclass.")

    reader = reader_class(source_path)
    if not isinstance(reader, WSIReader):
        raise TypeError(f"{spec} did not construct a WSIReader instance.")
    return reader
