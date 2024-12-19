import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def load_image(filename, shape=(256, 256)):
    with open(filename, 'rb') as f:
        img = np.frombuffer(f.read(), dtype=np.uint8)
    return img.reshape(shape)


def compute_dft_components(image):
    dft = np.fft.fft2(image)
    dft_centered = np.fft.fftshift(dft)

    real_part = np.real(dft_centered)
    imag_part = np.imag(dft_centered)
    magnitude_spectrum = np.log(1 + np.abs(dft_centered))
    phase_spectrum = np.angle(dft_centered)

    return real_part, imag_part, magnitude_spectrum, phase_spectrum


def normalize_to_8bit(array):
    array -= np.min(array)
    array = array / np.max(array)
    return (array * 255).astype(np.uint8)


def display_results(image, real, imag, magnitude, phase, title="Image"):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"{title} - Original")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(normalize_to_8bit(real), cmap='gray')
    plt.title(f"{title} - Real Part")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(normalize_to_8bit(imag), cmap='gray')
    plt.title(f"{title} - Imaginary Part")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(normalize_to_8bit(magnitude), cmap='gray')
    plt.title(f"{title} - Log-Magnitude Spectrum")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(normalize_to_8bit(phase), cmap='gray')
    plt.title(f"{title} - Phase Spectrum")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


filenames = ["camerabin.sec", "salesmanbin.sec", "headbin.sec", "eyeRbin.sec"]
for file in filenames:
    try:
        image = load_image(file)
        real, imag, magnitude, phase = compute_dft_components(image)
        display_results(image, real, imag, magnitude, phase, title=file.split('.')[0])
    except FileNotFoundError:
        print(f"File '{file}' not found. Please ensure it is in the correct directory.")