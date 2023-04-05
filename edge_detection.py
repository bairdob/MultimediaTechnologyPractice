import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_file() -> np.ndarray:
    """
    Читаем файл по пути и возвращаем массив картинки
    """
    file_path = input("Enter the path to the file: ")
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File not found")
    else:
        return cv2.imread(file_path)


def process_image(src_img: str,
                  blur_kernel_size: Tuple[int, int] = (5, 5),
                  blur_sigma: float = 1.4) -> None:
    """
    Применяем фильтр Собеля и показываем результат
    """
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # Reduce noise in the image using Gaussian blur
    img_gray_blur = cv2.GaussianBlur(img_gray, blur_kernel_size, blur_sigma)

    # Calculate the gradients using Sobel filters
    gx = cv2.Sobel(np.float32(img_gray_blur), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img_gray_blur), cv2.CV_64F, 0, 1, 3)

    # Compute the magnitude of the gradient
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Display the resulting image
    plt.imshow(np.uint8(mag))
    plt.show()


if __name__ == "__main__":
    image = read_file()
    process_image(image)
