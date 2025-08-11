"""
Pure OpenCV/Numpy helpers shared by servo nodes.
No ROS imports here.

Usage:
    from ear_servo.tracking import (
        compute_center, glare_mask_hsv, inpaint_if,
        good_features_to_track, lk_track,
        estimate_affine_ransac, compose_scaled_affine,
        iou, center_distance, pixel_error_from_box,
    )
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple

# ------------------ small utilities ------------------

def compute_center(box: Tuple[int,int,int,int]) -> np.ndarray:
    x, y, w, h = box
    return np.array([x + w/2.0, y + h/2.0], dtype=np.float32)

def pixel_error_from_box(box: Tuple[int,int,int,int], W: int, H: int) -> Tuple[float,float]:
    """Return (ex, ey) = target_center - image_center (pixels)."""
    x, y, w, h = box
    cx_img, cy_img = W/2.0, H/2.0
    cx_tgt, cy_tgt = x + w/2.0, y + h/2.0
    return float(cx_tgt - cx_img), float(cy_tgt - cy_img)

def iou(boxA: Optional[Tuple[float,float,float,float]],
        boxB: Optional[Tuple[float,float,float,float]]) -> Optional[float]:
    if (boxA is None) or (boxB is None):
        return None
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter = max(0.0, xB-xA) * max(0.0, yB-yA)
    if inter <= 0:
        return 0.0
    areaA = max(0.0, boxA[2]) * max(0.0, boxA[3])
    areaB = max(0.0, boxB[2]) * max(0.0, boxB[3])
    denom = areaA + areaB - inter
    return float(inter/denom) if denom > 0 else 0.0

def center_distance(boxA, boxB) -> Optional[float]:
    if (boxA is None) or (boxB is None):
        return None
    cA = (boxA[0]+boxA[2]/2.0, boxA[1]+boxA[3]/2.0)
    cB = (boxB[0]+boxB[2]/2.0, boxB[1]+boxB[3]/2.0)
    return float(np.hypot(cA[0]-cB[0], cA[1]-cB[1]))

# ------------------ glare & preprocessing ------------------

def glare_mask_hsv(bgr: np.ndarray,
                   v_hi: int = 240,
                   s_lo: int = 40,
                   v_hi2: int = 220) -> np.ndarray:
    """
    Detect specular highlights by HSV rules:
    - V very high OR (S very low AND V high) → glare.
    Returns uint8 mask {0,255}.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    m1 = (V >= v_hi)
    m2 = (S <= s_lo) & (V >= v_hi2)
    mask = (m1 | m2).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 1)
    mask = cv2.dilate(mask, k, 1)
    return mask

def inpaint_if(mask: Optional[np.ndarray], gray: np.ndarray) -> np.ndarray:
    if mask is None or (mask.max() == 0):
        return gray
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

# ------------------ features & tracking ------------------

def good_features_to_track(gray: np.ndarray,
                           glare_mask: Optional[np.ndarray],
                           max_corners: int = 150,
                           quality: float = 0.01,
                           min_distance: int = 7,
                           block_size: int = 7) -> Optional[np.ndarray]:
    """
    Shi–Tomasi corners avoiding glare. Returns (N,1,2) float32 or None.
    """
    mask = None
    if glare_mask is not None:
        mask = cv2.bitwise_not(glare_mask)
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_distance, blockSize=block_size,
        mask=mask, useHarrisDetector=False
    )
    return pts

def lk_track(prev_gray: np.ndarray, gray: np.ndarray, prev_pts: np.ndarray,
             win: int = 21, levels: int = 3):
    """
    Pyramidal LK from prev_gray to gray. Returns (good_old Nx2, good_new Nx2).
    """
    lk_params = dict(winSize=(win, win), maxLevel=levels,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
    if p1 is None:
        return None, None
    good_new = p1[st==1].reshape(-1,2)
    good_old = prev_pts[st==1].reshape(-1,2)
    if good_new.shape[0] < 2:
        return None, None
    return good_old, good_new

def estimate_affine_ransac(good_old: np.ndarray, good_new: np.ndarray,
                           ransac_thr: float = 3.0,
                           confidence: float = 0.99,
                           min_inliers: int = 6):
    """
    Estimate 2x3 affine with RANSAC. Returns (M_2x3 or None, inlier_count).
    """
    M, inliers = cv2.estimateAffinePartial2D(
        good_old, good_new, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thr, maxIters=2000, confidence=confidence
    )
    inl = int(inliers.sum()) if inliers is not None else 0
    if M is None or inl < min_inliers:
        return None, inl
    return M.astype(np.float32), inl

def scale_affine_2x3(M_work: np.ndarray, scale: float) -> np.ndarray:
    """Convert work-scale affine to full-res by conjugation with S."""
    A = np.vstack([M_work, [0,0,1]]).astype(np.float32)
    S  = np.array([[scale,0,0],[0,scale,0],[0,0,1]], np.float32)
    Si = np.array([[1/scale,0,0],[0,1/scale,0],[0,0,1]], np.float32)
    return (Si @ A @ S)[:2,:]

def compose_scaled_affine(M_work: np.ndarray, scale: float,
                          last_M_full: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Scale M_work to full-res and (optionally) left-compose with last_M_full.
    Returns new 2x3 cumulative.
    """
    M_full = scale_affine_2x3(M_work, scale)
    if last_M_full is None:
        return M_full
    H_step = np.vstack([M_full, [0,0,1]])
    H_cum  = np.vstack([last_M_full, [0,0,1]])
    H_new  = H_step @ H_cum
    return H_new[:2,:]
