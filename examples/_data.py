"""Shared image-loading helper for the examples/ scripts that use real data.

Not part of the invexapi package - a plain sibling module the example scripts
import from each other, kept out of invexapi/ since loading/normalizing a test
image is an examples-only concern, not a library concern.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_image(relative_path: str) -> torch.Tensor:
    """Load an image under examples/data/ as a float tensor normalized to [0, 1].

    ``relative_path`` is relative to examples/data/, e.g. ``"images/noisy_image.tif"``.
    Normalization divides by the dtype's max value (e.g. 65535 for uint16) so the
    result is comparable to the [0, 1]-scaled synthetic signals the other examples
    use; if the file is already floating point, it's returned as-is.
    """
    path = DATA_DIR / relative_path
    raw = np.array(Image.open(path))

    if np.issubdtype(raw.dtype, np.integer):
        factor = float(np.iinfo(raw.dtype).max)
        raw = raw.astype(np.float32) / factor

    return torch.from_numpy(raw.astype(np.float32))
