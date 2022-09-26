import cv2
import numpy as np


def cluster(image, k=4, with_labels_centers=False, only_labels_centers=False):
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    if only_labels_centers:
        return labels, centers
    segmented_data = centers[labels.flatten()]  # Mapping labels to center points (RGB Value)
    res = segmented_data.reshape(image.shape)
    if with_labels_centers:
        return res, labels, centers
    return res
