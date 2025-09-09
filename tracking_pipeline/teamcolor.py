import cv2
import numpy as np

def get_grass_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return hsv, mask

def extract_kit_color(player_img, grass_hsv_val):
    if player_img is None or player_img.size == 0:
        return np.array([0, 0, 0])
    hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
    lower = np.array([grass_hsv_val - 10, 40, 40])
    upper = np.array([grass_hsv_val + 10, 255, 255])
    grass_mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(grass_mask)
    h, w = mask.shape
    mask[int(0.5 * h):, :] = 0
    if np.count_nonzero(mask) == 0:
        return np.array([0, 0, 0])
    mean_bgr = cv2.mean(player_img, mask=mask)[:3]
    return np.array(mean_bgr)

class KitsHolder:
    """Minimal holder to mimic the previous inline class API."""
    def __init__(self, centers):
        self.cluster_centers_ = np.array(centers)

    def predict(self, X):
        arr = np.array(X)
        d = np.linalg.norm(arr[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(d, axis=1)
