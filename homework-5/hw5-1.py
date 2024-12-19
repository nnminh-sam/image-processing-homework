import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import matplotlib

matplotlib.use('TkAgg')


def load_image(filename, size=(256, 256)):
    with open(filename, 'rb') as f:
        image = np.fromfile(f, dtype=np.uint8, count=size[0]*size[1])
    return image.reshape(size)


def pad_image(image, pad_width):
    return np.pad(image, pad_width, mode='constant', constant_values=0)


def normalize_to_8bit(image):
    image = image - np.min(image)
    image = (image / np.max(image) * 255).astype(np.uint8)
    return image


input_image = load_image("salesmanbin.sec")

filter_size = 7
average_filter = np.ones((filter_size, filter_size)) / (filter_size ** 2)

padded_image = pad_image(input_image, pad_width=3)

output_image_a = convolve2d(padded_image, average_filter, mode='valid')

output_image_a = normalize_to_8bit(output_image_a)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image_a, cmap='gray')
plt.title("Output Image - Part (a)")
plt.axis('off')

plt.show()

# Part B

from numpy.fft import fft2, ifft2, fftshift


def dft_filtering(input_image, filter_size=7):
    padded_size = 384
    padded_image = np.zeros((padded_size, padded_size))
    padded_image[:256, :256] = input_image

    H = np.zeros((padded_size, padded_size))
    center = padded_size // 2
    H[center - filter_size//2:center + filter_size//2 + 1,
      center - filter_size//2:center + filter_size//2 + 1] = 1 / (filter_size ** 2)

    dft_image = fft2(padded_image)
    dft_H = fft2(H)

    dft_output = dft_image * dft_H

    output_padded = np.real(ifft2(dft_output))
    output_image_b = output_padded[:256, :256]

    output_image_b = normalize_to_8bit(output_image_b)

    return padded_image, H, dft_image, dft_H, dft_output, output_image_b


padded_image, H, dft_image, dft_H, dft_output, output_image_b = dft_filtering(input_image)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Input Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(padded_image, cmap='gray')
plt.title("Zero-padded Input Image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(normalize_to_8bit(H), cmap='gray')
plt.title("Impulse Response H")
plt.axis('off')

plt.subplot(2, 3, 4)
log_magnitude_input = np.log(1 + np.abs(fftshift(dft_image)))
plt.imshow(normalize_to_8bit(log_magnitude_input), cmap='gray')
plt.title("Log-Magnitude Spectrum - Input")
plt.axis('off')

plt.subplot(2, 3, 5)
log_magnitude_H = np.log(1 + np.abs(fftshift(dft_H)))
plt.imshow(normalize_to_8bit(log_magnitude_H), cmap='gray')
plt.title("Log-Magnitude Spectrum - H")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(output_image_b, cmap='gray')
plt.title("Output Image - Part (b)")
plt.axis('off')

plt.show()

max_difference = np.max(np.abs(output_image_a - output_image_b))
print(f"(b): Max difference from part (a): {max_difference}")

# Part c

from numpy.fft import fft2, ifft2, fftshift
import numpy as np
import matplotlib.pyplot as plt


def create_zero_phase_impulse_response(size, filter_size=7):
    h = np.zeros((size, size))
    center = size // 2
    start = center - filter_size // 2
    end = center + filter_size // 2 + 1
    h[start:end, start:end] = 1 / (filter_size ** 2)

    h2 = fftshift(h)
    return h2


def zero_pad_image(image, pad_size):
    padded_image = np.zeros((pad_size, pad_size))
    original_size = image.shape[0]
    start = (pad_size - original_size) // 2
    padded_image[start:start + original_size, start:start + original_size] = image
    return padded_image


def dft_zero_phase_filtering(input_image, h2, padded_size=512):
    input_padded = zero_pad_image(input_image, padded_size)
    h2_padded = zero_pad_image(h2, padded_size)

    dft_input = fft2(input_padded)
    dft_h2 = fft2(h2_padded)

    dft_output = dft_input * dft_h2

    output_padded = np.real(ifft2(dft_output))

    start = (padded_size - input_image.shape[0]) // 2
    output_image_c = output_padded[start:start + 256, start:start + 256]

    output_image_c = normalize_to_8bit(output_image_c)

    return input_padded, h2_padded, output_image_c


h2 = create_zero_phase_impulse_response(size=256)
input_padded, h2_padded, output_image_c = dft_zero_phase_filtering(input_image, h2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Original Input Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(normalize_to_8bit(h2), cmap='gray')
plt.title("Zero-Phase Impulse Response (h2)")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(normalize_to_8bit(h2_padded), cmap='gray')
plt.title("Zero-Padded Zero-Phase Impulse Response (h2ZP)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(output_image_c, cmap='gray')
plt.title("Output Image - Part (c)")
plt.axis('off')

plt.tight_layout()
plt.show()

max_difference_c = np.max(np.abs(output_image_a - output_image_c))
print(f"(c): Max difference from part (a): {max_difference_c}")
