import numpy as np
from skimage import color
from prism.scripts.utils import input_normalization


def color_transfer(src: np.array, dst: np.array) -> np.array:
    """
    Transfer the color from the source image to the destination image.
    From the article "Color Transfer between Images" by Reinhard et al.
    https://ieeexplore.ieee.org/document/946629

    Args:
        src (np.array): Source image of size HxWx3 (RGB).
        dst (np.array): Destination image of size HxWx3 (RGB).
    Returns:
        np.array: Color transferred image of size HxWx3 (RGB).
    """
    src = input_normalization(src)
    dst = input_normalization(dst)
    src_lab = color.rgb2lab(src)
    dst_lab = color.rgb2lab(dst)
    for c in range(3):
        src_mean, src_std = np.mean(src_lab[..., c]), np.std(src_lab[..., c])
        dst_mean, dst_std = np.mean(dst_lab[..., c]), np.std(dst_lab[..., c])
        dst_lab[..., c] = (dst_lab[..., c] - dst_mean) / dst_std * src_std + src_mean
    dst_rgb = color.lab2rgb(dst_lab)
    dst_rgb = np.clip(dst_rgb, 0, 1)
    return dst_rgb
