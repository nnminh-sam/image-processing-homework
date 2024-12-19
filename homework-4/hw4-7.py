import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def load_image(filename, shape=(256, 256)):
    with open(filename, 'rb') as f:
        img = np.frombuffer(f.read(), dtype=np.uint8)
    return img.reshape(shape)


def compute_dft(image):
    dft = np.fft.fft2(image)
    return np.fft.fftshift(dft)


def reconstruct_image(magnitude, phase):
    complex_dft = magnitude * np.exp(1j * phase)
    return np.real(np.fft.ifft2(np.fft.ifftshift(complex_dft)))


def normalize_to_8bit(array):
    array = array.copy()
    array -= np.min(array)
    array = array / np.max(array)
    return (array * 255).astype(np.uint8)


def display_images(images, titles, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


filename = 'camerabin.sec'
image = load_image(filename)

dft = compute_dft(image)
magnitude = np.abs(dft)
phase = np.angle(dft)

magnitude_j1 = magnitude
phase_j1 = np.zeros_like(phase)
J1 = reconstruct_image(magnitude_j1, phase_j1)

magnitude_j2 = np.ones_like(magnitude)
phase_j2 = phase
J2 = reconstruct_image(magnitude_j2, phase_j2)

J1_log = np.log(1 + np.abs(J1))

J1_normalized = normalize_to_8bit(J1_log.copy())
J2_normalized = normalize_to_8bit(J2.copy())

display_images(
    [normalize_to_8bit(image), J1_normalized, J2_normalized],
    ["Original Image", "J1 (Log-transformed)", "J2 (Phase Contribution)"]
)