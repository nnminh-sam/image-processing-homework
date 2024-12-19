import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib

matplotlib.use('TkAgg')


def read_image(file_path, size=(256, 256)):
    return np.fromfile(file_path, dtype=np.uint8).reshape(size)


def display_images(images, titles):
    plt.figure(figsize=(12, 6))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def compute_mse(original, noisy):
    return np.mean((original - noisy) ** 2)


def compute_isnr(original, noisy, filtered):
    numerator = np.sum((original - noisy) ** 2)
    denominator = np.sum((original - filtered) ** 2)
    return 10 * np.log10(numerator / denominator)


def apply_ideal_lpf(image, cutoff):
    M, N = image.shape
    U, V = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2), indexing='ij')
    H = np.sqrt(U**2 + V**2) <= cutoff
    dft_image = fftshift(fft2(image))
    filtered_dft = H * dft_image
    filtered_image = np.abs(ifft2(ifftshift(filtered_dft)))
    return filtered_image


def apply_gaussian_lpf(image, cutoff):
    M, N = image.shape
    U, V = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2), indexing='ij')
    sigma = 0.19 * M / cutoff
    H = np.exp(-2 * np.pi**2 * sigma**2 * (U**2 + V**2) / M**2)
    dft_image = fftshift(fft2(image))
    filtered_dft = H * dft_image
    filtered_image = np.abs(ifft2(ifftshift(filtered_dft)))
    return filtered_image


original = read_image("girl2bin.sec")
noise_high = read_image("girl2Noise32Hibin.sec")
noise_broadband = read_image("girl2Noise32bin.sec")

display_images([original, noise_high, noise_broadband], ["Original", "High Noise", "Broadband Noise"])

mse_high = compute_mse(original, noise_high)
mse_broadband = compute_mse(original, noise_broadband)
print(f"MSE (High Noise): {mse_high}")
print(f"MSE (Broadband Noise): {mse_broadband}")

cutoff = 64
filtered_ideal_high = apply_ideal_lpf(noise_high, cutoff)
filtered_ideal_broadband = apply_ideal_lpf(noise_broadband, cutoff)

display_images([filtered_ideal_high, filtered_ideal_broadband], ["Ideal LPF - High Noise", "Ideal LPF - Broadband Noise"])

isnr_high = compute_isnr(original, noise_high, filtered_ideal_high)
isnr_broadband = compute_isnr(original, noise_broadband, filtered_ideal_broadband)
print(f"ISNR (High Noise): {isnr_high} dB")
print(f"ISNR (Broadband Noise): {isnr_broadband} dB")

filtered_gaussian_high = apply_gaussian_lpf(noise_high, cutoff)
filtered_gaussian_broadband = apply_gaussian_lpf(noise_broadband, cutoff)

display_images([filtered_gaussian_high, filtered_gaussian_broadband], ["Gaussian LPF - High Noise", "Gaussian LPF - Broadband Noise"])

cutoff = 77.5
filtered_gaussian_high_adjusted = apply_gaussian_lpf(noise_high, cutoff)
filtered_gaussian_broadband_adjusted = apply_gaussian_lpf(noise_broadband, cutoff)

display_images([filtered_gaussian_high_adjusted, filtered_gaussian_broadband_adjusted],
               ["Gaussian LPF (77.5) - High Noise", "Gaussian LPF (77.5) - Broadband Noise"])
