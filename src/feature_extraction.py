from typing import List
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

IMG_SIZE = (128, 128)

P = 16
R = 2
METHOD = "uniform"
NUM_BINS = P + 2


def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    face_eq = cv2.equalizeHist(gray_face)
    face_resized = cv2.resize(face_eq, IMG_SIZE)
    return face_resized


def gaussian_smooth(img: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def directional_derivatives(img: np.ndarray) -> List[np.ndarray]:
    img = img.astype(np.float32)

    kx = np.array([[0, 0, 0],
                   [1, -2, 1],
                   [0, 0, 0]], dtype=np.float32)

    ky = np.array([[0, 1, 0],
                   [0, -2, 0],
                   [0, 1, 0]], dtype=np.float32)

    kdiag1 = np.array([[1, 0, 0],
                       [0, -2, 0],
                       [0, 0, 1]], dtype=np.float32)

    kdiag2 = np.array([[0, 0, 1],
                       [0, -2, 0],
                       [1, 0, 0]], dtype=np.float32)

    d_h = cv2.filter2D(img, -1, kx)
    d_v = cv2.filter2D(img, -1, ky)
    d_d1 = cv2.filter2D(img, -1, kdiag1)
    d_d2 = cv2.filter2D(img, -1, kdiag2)

    return [d_h, d_v, d_d1, d_d2]


def lbp_hist(img: np.ndarray) -> np.ndarray:
    img_uint8 = np.uint8(np.clip(img, 0, 255))
    lbp = local_binary_pattern(img_uint8, P, R, METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=NUM_BINS,
        range=(0, NUM_BINS),
        density=True,
    )
    return hist


def extract_hddl_lbp_features(gray_face: np.ndarray) -> np.ndarray:
    face = preprocess_face(gray_face)
    face_smooth = gaussian_smooth(face)

    h, w = face_smooth.shape

    upper_face = face_smooth[0:int(0.5 * h), :]
    lower_face = face_smooth[int(0.5 * h):h, :]

    upper_derivs = directional_derivatives(upper_face)
    upper_hists = [lbp_hist(d) for d in upper_derivs]
    upper_features = np.hstack(upper_hists)

    lower_derivs = directional_derivatives(lower_face)
    lower_hists = [lbp_hist(d) for d in lower_derivs]
    lower_features = np.hstack(lower_hists)

    feature_vec = np.hstack([upper_features, lower_features]).astype(np.float32)
    return feature_vec
