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


def process_image(src_img: str, k: int = 8, max_iter: int = 10) -> None:
    """
    Применяем алгоритм k-средних для картинки
    """
    # Reshape the image to a 2D array of pixels
    Z = src_img.reshape((-1, 3))

    # Convert the pixel values to np.float32
    Z = np.float32(Z)

    # Define the convergence criteria for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)

    # Apply k-means clustering to the pixel values
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the center values back to uint8
    center = np.uint8(center)

    # Apply the quantization to the pixel values and reshape back to the original image shape
    res = center[label.flatten()]
    res2 = res.reshape((src_img.shape))

    # Display the resulting image
    plt.imshow(res2)
    plt.show()


if __name__ == "__main__":
    image = read_file()
    process_image(image)
