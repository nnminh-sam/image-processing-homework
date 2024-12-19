import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

u1 = v1 = 1.5
pi = np.pi

cos_value = lambda m, n: np.cos((2 * pi / 8) * (u1 * m + v1 * n))

m, n = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
I5 = cos_value(m, n)

I5_normalized = 255 * (I5 - np.min(I5)) / (np.max(I5) - np.min(I5))
I5_normalized = I5_normalized.astype(np.uint8)

I5_dft = np.fft.fft2(I5)
I5_dft_centered = np.fft.fftshift(I5_dft)

real_part = np.real(I5_dft_centered)
imag_part = np.imag(I5_dft_centered)

print("Real part of centered DFT (I5):")
print(np.array2string(real_part, formatter={'float_kind': lambda x: f"{x:6.2f}"}))

print("\nImaginary part of centered DFT (I5):")
print(np.array2string(imag_part, formatter={'float_kind': lambda x: f"{x:6.2f}"}))

plt.imshow(I5_normalized, cmap='gray', vmin=0, vmax=255)
plt.title("I5 as 8-bit Grayscale Image")
plt.colorbar(label="Intensity")
plt.show()
