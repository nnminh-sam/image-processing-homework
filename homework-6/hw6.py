import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, grey_opening, grey_closing

matplotlib.use('TkAgg')

IMG_SIZE = 256
STRUCTURING_ELEMENT_SIZE = 3


def read_image(filename):
    with open(filename, 'rb') as f:
        img = np.fromfile(f, dtype=np.uint8).reshape((IMG_SIZE, IMG_SIZE))
    return img


def apply_median_filter(image):
    return median_filter(image, size=STRUCTURING_ELEMENT_SIZE, mode='constant', cval=0)


def apply_morphological_opening(image):
    return grey_opening(image, size=(STRUCTURING_ELEMENT_SIZE, STRUCTURING_ELEMENT_SIZE))


def apply_morphological_closing(image):
    return grey_closing(image, size=(STRUCTURING_ELEMENT_SIZE, STRUCTURING_ELEMENT_SIZE))


def save_and_display_results(original, median_result, opening_result, closing_result, title_prefix):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title(f"{title_prefix}: Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f"{title_prefix}: Median Filter")
    plt.imshow(median_result, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title(f"{title_prefix}: Opening")
    plt.imshow(opening_result, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title(f"{title_prefix}: Closing")
    plt.imshow(closing_result, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{title_prefix}-hw6.png")
    plt.show()


def main():
    image1 = read_image('camera9bin.sec')
    image2 = read_image('camera99bin.sec')

    median1 = apply_median_filter(image1)
    opening1 = apply_morphological_opening(image1)
    closing1 = apply_morphological_closing(image1)

    median2 = apply_median_filter(image2)
    opening2 = apply_morphological_opening(image2)
    closing2 = apply_morphological_closing(image2)

    save_and_display_results(image1, median1, opening1, closing1, title_prefix="Image 1")

    save_and_display_results(image2, median2, opening2, closing2, title_prefix="Image 2")


if __name__ == "__main__":
    main()