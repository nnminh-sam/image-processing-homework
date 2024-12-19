import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

u0 = v0 = 2
pi = np.pi
sin_value = lambda m, n: np.sin((2 * pi / 8) * (u0 * m + v0 * n))

m, n = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
I4 = sin_value(m, n)

I4_normalized = 255 * (I4 - np.min(I4)) / (np.max(I4) - np.min(I4))
I4_normalized = I4_normalized.astype(np.uint8)

I4_dft = np.fft.fft2(I4)
I4_dft_centered = np.fft.fftshift(I4_dft)

real_part = np.real(I4_dft_centered)
imag_part = np.imag(I4_dft_centered)

print("Real part of centered DFT (I4):")
print(np.array2string(real_part, formatter={'float_kind': lambda x: f"{x:6.2f}"}))

print("\nImaginary part of centered DFT (I4):")
print(np.array2string(imag_part, formatter={'float_kind': lambda x: f"{x:6.2f}"}))

plt.imshow(I4_normalized, cmap='gray', vmin=0, vmax=255)
plt.title("I4 as 8-bit Grayscale Image")
plt.colorbar(label="Intensity")
plt.show()
