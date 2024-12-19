import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

N = 8
u0, v0 = 2, 2

m = np.arange(N)
n = np.arange(N)
COLS, ROWS = np.meshgrid(m, n)

I2 = 0.5 * np.exp(-1j * (2 * np.pi / N) * (u0 * COLS + v0 * ROWS))

real_I2 = np.real(I2)
imag_I2 = np.imag(I2)


def normalize_to_gray(matrix):
    return 255 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))


real_I2_gray = normalize_to_gray(real_I2)
imag_I2_gray = normalize_to_gray(imag_I2)

plt.imshow(real_I2_gray, cmap='gray')
plt.title('Real Part of I2')
plt.colorbar()
plt.savefig('Real-Part-of-I2-hw4-2.png')
plt.show()

plt.imshow(imag_I2_gray, cmap='gray')
plt.title('Imaginary Part of I2')
plt.colorbar()
plt.savefig('Imaginary-Part-of-I2-hw4-2.png')
plt.show()

I2_dft = np.fft.fft2(I2)
I2_dft_centered = np.fft.fftshift(I2_dft)

real_dft_I2 = np.round(np.real(I2_dft_centered), 4)
imag_dft_I2 = np.round(np.imag(I2_dft_centered), 4)

print("Real Part of Centered DFT(I2):")
for row in real_dft_I2:
    print(" ".join(f"{val:8.4f}" for val in row))

print("\nImaginary Part of Centered DFT(I2):")
for row in imag_dft_I2:
    print(" ".join(f"{val:8.4f}" for val in row))
