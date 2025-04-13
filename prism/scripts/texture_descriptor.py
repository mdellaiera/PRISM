import numpy as np
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import color
from tqdm import tqdm
from multiprocessing import Pool

"""
From the article "Color Transfer and Colorization based on Textural Properties" by Arbelot et al.
https://hal.science/hal-01246615/file/RR-8834.pdf
"""

def compute_feature(img: np.ndarray) -> np.ndarray:
    filter_first_order = np.array([0.5, 0, -0.5]).reshape(3, 1)
    filter_second_order = np.array([1, -2, 1]).reshape(3, 1)

    feature_vector = np.dstack([
        conv2(img, filter_first_order, mode='same'),
        conv2(img, filter_first_order.T, mode='same'),
        conv2(img, filter_second_order, mode='same'),
        conv2(img, filter_second_order.T, mode='same'),
        conv2(conv2(img, filter_first_order, mode='same'), filter_first_order.T, mode='same'),
        img,
    ])

    for i in range(feature_vector.shape[-1]):
        feature_vector[:, :, i] /= feature_vector[:, :, i].std()

    return feature_vector


def get_gaussian_filter(radius: int) -> np.ndarray:
    sigma = radius / 3
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x * x / (2 * sigma * sigma))
    kernel = kernel / sum(kernel)
    return kernel.reshape(-1, 1)


def compute_average(tensor: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.dstack([
        conv2(conv2(tensor[:, :, i], kernel, mode='same'), kernel.T, mode='same') for i in range(tensor.shape[2])
    ])


def compute_covariance(tensor: np.ndarray, tensor_averaged: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    covariance_matrix = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[2]))

    for i in range(tensor.shape[2]):
        for j in range(i, tensor.shape[2]):
            t_i = tensor[:, :, i]
            t_j =  tensor[:, :, j]
            
            mean_i = tensor_averaged[:, :, i]
            mean_j = tensor_averaged[:, :, j]

            covariance_matrix[:, :, i, j] = conv2(conv2(t_i * t_j, kernel, 'same'), kernel.T, 'same') - mean_i * mean_j
            if i != j:
                covariance_matrix[:, : ,j, i] = covariance_matrix[:, :, i, j]  # by symmetry

    return covariance_matrix


def cholesky_decomposition(tensor: np.ndarray) -> np.ndarray:
    L = np.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            L[i, j] = np.linalg.cholesky(tensor[i, j])

    return L


def add_averaged(tensor: np.ndarray, tensor_averaged: np.ndarray) -> np.ndarray:
    descriptor = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[2] + 1))

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            descriptor[i, j] = np.hstack([tensor[i, j], tensor_averaged[i, j].reshape(-1, 1)])

    return descriptor


def compute_texture_descriptor(img: np.ndarray, radius: int) -> np.ndarray:
    feature = compute_feature(img)
    kernel = get_gaussian_filter(radius)
    feature_averaged = compute_average(feature, kernel)
    covariance = compute_covariance(feature, feature_averaged, kernel)
    cholesky = cholesky_decomposition(covariance)
    # descriptor = add_averaged(cholesky, feature_averaged)

    return cholesky


def pixel_similarity(pixel: np.ndarray, descriptor: np.ndarray, sigma: float = 1.0) -> float:
    descriptor = descriptor.reshape(descriptor.shape[0], descriptor.shape[1], -1)
    distance = np.linalg.norm(descriptor - pixel.reshape(-1), ord=2, axis=2)
    return np.exp(-distance / (2 * sigma * sigma))


# def colorization(ref_rgb: np.ndarray, tgt_gray: np.ndarray, desc_ref: np.ndarray, desc_tgt: np.ndarray, sigma: float) -> np.ndarray:
#     X, Y = np.meshgrid(np.arange(desc_ref.shape[1]), np.arange(desc_ref.shape[0]))
#     pixels = np.stack([X.flatten(), Y.flatten()], axis=1)

#     ref_lab = color.rgb2lab(ref_rgb)
#     tgt_lab = np.zeros_like(ref_lab)
#     tgt_lab[:, :, 0] = tgt_gray * 100  # luminance channel
#     for p in tqdm(pixels, total=pixels.shape[0]):
#         sim = pixel_similarity(desc_tgt[p[1], p[0]].reshape(-1), desc_ref, sigma)
#         sim /= sim.sum()
#         tgt_lab[p[1], p[0], 1] = np.sum(sim * ref_lab[:, :, 1])  # chromatic 'a' channel
#         tgt_lab[p[1], p[0], 2] = np.sum(sim * ref_lab[:, :, 2])  # chromatic 'b' channel

#     return color.lab2rgb(tgt_lab)


def colorization(ref_rgb: np.ndarray, tgt_gray: np.ndarray, desc_ref: np.ndarray, desc_tgt: np.ndarray, sigma: float, num_process: int=10) -> np.ndarray:
    X, Y = np.meshgrid(np.arange(desc_ref.shape[1]), np.arange(desc_ref.shape[0]))
    pixels = np.stack([X.flatten(), Y.flatten()], axis=1)

    ref_lab = color.rgb2lab(ref_rgb)
    tgt_lab = np.zeros_like(ref_lab)
    tgt_lab[:, :, 0] = tgt_gray * 100  # luminance channel

    # Prepare arguments for multiprocessing
    args = [(p, desc_tgt, desc_ref, sigma, ref_lab) for p in pixels]

    # Use multiprocessing pool to process pixels in parallel
    with Pool(num_process) as pool:
        results = list(tqdm(pool.imap(process_pixel, args), total=len(pixels)))

    # Assign chromatic channels back to the target Lab image
    for y, x, chroma_a, chroma_b in results:
        tgt_lab[y, x, 1] = chroma_a
        tgt_lab[y, x, 2] = chroma_b

    return color.lab2rgb(tgt_lab)


def process_pixel(args):
    p, desc_tgt, desc_ref, sigma, ref_lab = args
    sim = pixel_similarity(desc_tgt[p[1], p[0]].reshape(-1), desc_ref, sigma)
    sim /= sim.sum()
    chroma_a = np.sum(sim * ref_lab[:, :, 1])  # chromatic 'a' channel
    chroma_b = np.sum(sim * ref_lab[:, :, 2])  # chromatic 'b' channel
    return p[1], p[0], chroma_a, chroma_b


def check_symmetric(x: np.ndarray, rtol: float=1e-05, atol: float=1e-08) -> bool:
    return np.allclose(x, x.transpose(0, 1, 3, 2), rtol=rtol, atol=atol)


def is_pos_def(x: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(x) > 0)


def plot_descriptor(descriptor: np.ndarray, path: str=None) -> None:
    plt.figure(figsize=(10, 10))
    for i in range(6):
        for j in range(7):
            plt.subplot(6, 7, i * 7 + j + 1)
            plt.imshow(descriptor[:, :, i, j], cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300)


def plot_similarity_map(ref: np.ndarray, tgt: np.ndarray, desc_ref: np.ndarray, desc_tgt: np.ndarray, sigma: float, num_patches: int, path: str=None) -> None:
    markersize = 10
    marker = 'o'
    ncols = num_patches + 2  # add ref and tgt images on both sides
    nrows = num_patches

    pixels = get_random_pixel_coordinates_from_patches(ref.shape, num_patches)
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())
    similarities = []
    for p in pixels:
        similarities.append(pixel_similarity(desc_tgt[p[1], p[0]], desc_ref, sigma))

    plt.figure(figsize=(ncols * 5, nrows * 5))

    for i in range(num_patches):  # add ref image
        j = i * ncols + 1
        plt.subplot(nrows, ncols, j)
        plt.imshow(ref, cmap='gray')
        # for p, c in zip(pixels, colors):
        #     plt.plot(p[1], p[0], marker, color=c, markersize=markersize)
        plt.axis('off')

    for i in range(num_patches):  # add tgt image
        j = i * ncols + ncols
        plt.subplot(nrows, ncols, j)
        plt.imshow(tgt, cmap='gray')
        for p, c in zip(pixels, colors):
            plt.plot(p[1], p[0], marker, color=c, markersize=markersize)
        plt.axis('off')

    j = 1
    for i, (p, s, c) in enumerate(zip(pixels, similarities, colors)):
        if (j - 1) % ncols == 0:
            j += 1
        if j % ncols == 0:
            j += 2
        plt.subplot(nrows, ncols, j)
        plt.imshow(s, cmap='gray')
        plt.plot(p[1], p[0], marker, color=c, markersize=markersize)
        plt.axis('off')
        j += 1

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300)


def get_random_pixel_coordinates_from_patches(image_shape: float, num_patches: int) -> np.ndarray:
    height, width = image_shape[:2]
    patch_height = height // num_patches
    patch_width = width // num_patches

    coordinates = []

    for i in range(num_patches):
        for j in range(num_patches):
            # Define the patch boundaries
            y_start = i * patch_height
            y_end = (i + 1) * patch_height if i < num_patches - 1 else height
            x_start = j * patch_width
            x_end = (j + 1) * patch_width if j < num_patches - 1 else width

            # Randomly select a pixel within the patch
            y_rand = np.random.randint(y_start, y_end)
            x_rand = np.random.randint(x_start, x_end)

            coordinates.append((y_rand, x_rand))

    return np.array(coordinates)
