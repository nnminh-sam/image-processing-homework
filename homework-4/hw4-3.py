import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

N = 8
u0, v0 = 2, 2

m = np.arange(N)
n = np.arange(N)
COLS, ROWS = np.meshgrid(m, n)  # Create 2D grid

I3 = np.cos((2 * np.pi / N) * (u0 * COLS + v0 * ROWS))

I3_gray = 255 * (I3 - np.min(I3)) / (np.max(I3) - np.min(I3))
I3_gray = I3_gray.astype(np.uint8)  # Convert to unsigned 8-bit integers

plt.imshow(I3_gray, cmap='gray', vmin=0, vmax=255)
plt.title('I3 as Grayscale Image')
plt.colorbar(label='Pixel Value')
plt.show()

I3_dft = np.fft.fft2(I3)
I3_dft_centered = np.fft.fftshift(I3_dft)

real_dft_I3 = np.round(np.real(I3_dft_centered), 4)
imag_dft_I3 = np.round(np.imag(I3_dft_centered), 4)

print("Real Part of Centered DFT(I3):")
for row in real_dft_I3:
    print(" ".join(f"{val:8.4f}" for val in row))


print("\nImaginary Part of Centered DFT(I3):")
for row in imag_dft_I3:
    print(" ".join(f"{val:8.4f}" for val in row))
