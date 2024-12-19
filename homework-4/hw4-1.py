import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

m = np.arange(8)
n = np.arange(8)
COLS, ROWS = np.meshgrid(m, n)

u0, v0 = 2, 2
I1 = 0.5 * np.exp(1j * 2 * np.pi / 8 * (u0 * COLS + v0 * ROWS))

real_I1 = np.real(I1)
imag_I1 = np.imag(I1)

real_norm = 255 * (real_I1 - np.min(real_I1)) / (np.max(real_I1) - np.min(real_I1))
imag_norm = 255 * (imag_I1 - np.min(imag_I1)) / (np.max(imag_I1) - np.min(imag_I1))

plt.imshow(real_norm, cmap='gray')
plt.title('Real Part of I1')
plt.savefig('real-part-of-i1-hw4-1.png')
plt.show()

plt.imshow(imag_norm, cmap='gray')
plt.title('Imaginary Part of I1')
plt.savefig('imaginary-part-of-i1-hw4-1.png')
plt.show()

I1_dft = np.fft.fft2(I1)
I1_dft_centered = np.fft.fftshift(I1_dft)

real_dft = np.round(np.real(I1_dft_centered), 4)
imag_dft = np.round(np.imag(I1_dft_centered), 4)

print("Real Part of DFT(I1):")
print(real_dft)

print("Imaginary Part of DFT(I1):")
print(imag_dft)
