"""Image degradation operators used during fusion training."""

import random

import cv2
import numpy as np
import scipy
import scipy.stats as ss
import torch
from PIL import ImageEnhance
from scipy import ndimage
from scipy.linalg import orth

from utils_image import pil2single, single2pil, single2uint, uint2single


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype("float32")
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharpened = img + weight * residual
    sharpened = np.clip(sharpened, 0, 1)
    return soft_mask * sharpened + (1 - soft_mask) * img


def analytic_kernel(k):
    k_size = k.shape[0]
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    for row in range(k_size):
        for col in range(k_size):
            big_k[2 * row : 2 * row + k_size, 2 * col : 2 * col + k_size] += (
                k[row, col] * k
            )
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    return cropped_big_k / cropped_big_k.sum()


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    v = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
        np.array([1.0, 0.0]),
    )
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    return gm_blur_kernel(mean=[0, 0], cov=sigma, size=ksize)


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    kernel = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            kernel[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)
    return kernel / np.sum(kernel)


def blur(x, k):
    n, c = x.shape[:2]
    p1, p2 = (k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2
    x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode="replicate")
    k = k.repeat(1, c, 1, 1)
    k = k.view(-1, 1, k.shape[2], k.shape[3])
    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=0, groups=n * c)
    return x.view(n, c, x.shape[2], x.shape[3])


def gen_kernel(
    k_size=np.array([15, 15]),
    scale_factor=np.array([4, 4]),
    min_var=0.6,
    max_var=10.0,
    noise_level=0,
):
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    lambd = np.diag([lambda_1, lambda_2])
    q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sigma = q @ lambd @ q.T
    inv_sigma = np.linalg.inv(sigma)[None, None, :, :]

    mu = k_size // 2 - 0.5 * (scale_factor - 1)
    mu = mu[None, None, :, None]

    x_grid, y_grid = np.meshgrid(range(k_size[0]), range(k_size[1]))
    z_grid = np.stack([x_grid, y_grid], 2)[:, :, :, None]

    z_centered = z_grid - mu
    z_transposed = z_centered.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(z_transposed @ inv_sigma @ z_centered)) * (
        1 + noise
    )
    return raw_kernel / np.sum(raw_kernel)


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    size = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    x, y = np.meshgrid(
        np.arange(-size[1], size[1] + 1),
        np.arange(-size[0], size[0] + 1),
    )
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel[kernel < scipy.finfo(float).eps * kernel.max()] = 0
    sumh = kernel.sum()
    if sumh != 0:
        kernel = kernel / sumh
    return kernel


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha, 1])])
    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)
    return np.array([[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]])


def fspecial(filter_type, *args, **kwargs):
    if filter_type == "gaussian":
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == "laplacian":
        return fspecial_laplacian(*args, **kwargs)
    raise ValueError(f"Unsupported filter type: {filter_type}")


def add_blur(img, sf=4):
    wd2 = 2.0 + sf
    wd = 1.0 + 0.5 * sf

    if random.random() < 0.5:
        l1 = wd2 * random.uniform(0.5, 1.5)
        l2 = wd2 * random.uniform(0.5, 1.5)
        kernel = anisotropic_Gaussian(
            ksize=2 * random.randint(2, 5) + 3,
            theta=random.random() * np.pi,
            l1=l1,
            l2=l2,
        )
    else:
        kernel = fspecial("gaussian", 2 * random.randint(2, 5) + 3, wd * random.random())

    return ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode="mirror")


def add_Gaussian_noise(img, noise_level1=10, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:
        img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(
            np.float32
        )
    else:
        noise_bound = noise_level2 / 255.0
        diagonal = np.diag(np.random.rand(3))
        basis = orth(np.random.rand(3, 3))
        covariance = np.dot(np.dot(np.transpose(basis), diagonal), basis)
        img += np.random.multivariate_normal(
            [0, 0, 0],
            np.abs(noise_bound**2 * covariance),
            img.shape[:2],
        ).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_Gaussian_none(img, noise_level2=25):
    noise_bound = noise_level2 / 255.0
    diagonal = np.diag(np.random.rand(3))
    basis = orth(np.random.rand(3, 3))
    covariance = np.dot(np.dot(np.transpose(basis), diagonal), basis)
    img += np.random.multivariate_normal(
        [0, 0, 0],
        np.abs(noise_bound**2 * covariance),
        img.shape[:2],
    ).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_Gaussian_WAG(img, noise_level):
    img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_Gaussian_GAWG(img, noise_level):
    img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(
        np.float32
    )
    return np.clip(img, 0.0, 1.0)


def add_JPEG_noise(img, quality_factor):
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    _, encoded_img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encoded_img, 1)
    return cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)


def adjust_contrast(img, contrast_rate):
    img = single2pil(img)
    img = ImageEnhance.Contrast(img).enhance(contrast_rate)
    img = pil2single(img)
    return np.clip(img, 0.0, 1.0)


def adjust_brightness(img, brightness_rate, brightness_prob):
    img = single2pil(img)
    hsv_image = img.convert("HSV")
    h, s, v = hsv_image.split()
    v = v.point(lambda p: p * brightness_rate)
    hsv_image = Image.merge("HSV", (h, s, v))
    img = hsv_image.convert("RGB")
    img = pil2single(img)
    return np.clip(img, 0.0, 1.0)
