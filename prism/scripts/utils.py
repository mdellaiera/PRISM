import numpy as np


def input_normalization(img: np.array) -> np.array:
    """
    Normalize the input image to the range [0, 1] and reshape to (H, W, C) if needed.

    Args:
        img (np.array): Input image.

    Returns:
        np.array: Normalized image.
    """
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 2:
        img = img[..., np.newaxis]
    return img
