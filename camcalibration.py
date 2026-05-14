"""
Camera Calibration Tool - Barrel Distortion Removal
====================================================
Usage:
  python camera_calibration.py --rtsp "rtsp://user:pass@ip:port/stream"
  python camera_calibration.py --rtsp "rtsp://user:pass@ip:port/stream" --cols 9 --rows 6

Controls (when camera window is focused):
  S  - Capture current frame for calibration
  Q  - Quit
"""

import cv2
import numpy as np
import threading
import time
import sys
import os
import argparse
from datetime import datetime
from collections import deque

# ─── Configuration ────────────────────────────────────────────────────────────
DISPLAY_WIDTH  = 1920          # Max display width  (fits HD screen)
DISPLAY_HEIGHT = 1080           # Max display height
TARGET_POSITIONS = [           # (label, hint_x_norm, hint_y_norm)
    ("CENTER",        0.50, 0.50),
    ("TOP-LEFT",      0.20, 0.20),
    ("TOP-RIGHT",     0.80, 0.20),
    ("BOTTOM-LEFT",   0.20, 0.80),
    ("BOTTOM-RIGHT",  0.80, 0.80),
    ("LEFT-CENTER",   0.15, 0.50),
    ("RIGHT-CENTER",  0.85, 0.50),
    ("TOP-CENTER",    0.50, 0.15),
    ("BOTTOM-CENTER", 0.50, 0.85),
    ("SLIGHT-TILT",   0.50, 0.50),   # user tilts camera ~15°
    ("TILT-LEFT",     0.30, 0.35),
    ("TILT-RIGHT",    0.70, 0.65),
]
MIN_GOOD_IMAGES  = 12          # minimum accepted frames before computing
MAX_REPROJ_ERROR = 0.3         # target reprojection error
BLUR_THRESHOLD   = 80.0        # Laplacian variance; below = too blurry
CHECKER_TIMEOUT  = 8.0         # seconds to wait for chessboard detection
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Barrel distortion calibration via RTSP")
    p.add_argument("--rtsp", help="RTSP stream URL",default="rtsp://admin:123456789!@192.168.0.150:554/ch1/sub")
    p.add_argument("--cols", type=int, default=9,
                   help="Inner corners per row (default 9)")
    p.add_argument("--rows", type=int, default=6,
                   help="Inner corners per column (default 6)")
    p.add_argument("--out", default="barrel_calibration.npz",
                   help="Output .npz file (default: barrel_calibration.npz)")
    return p.parse_args()


# ─── Overlay helpers ──────────────────────────────────────────────────────────

def resize_for_display(frame):
    h, w = frame.shape[:2]
    scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h, 1.0)
    
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return frame


def put_overlay(frame, lines, color=(255, 255, 255), bg=(0, 0, 0)):
    """Draw multi-line message in top-left with background box.  Erases previous by redrawing."""
    print(lines)
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = 0.6
    thickness  = 1
    pad        = 8
    line_h     = 24
    max_w      = max(cv2.getTextSize(l, font, scale, thickness)[0][0] for l in lines)
    box_h      = line_h * len(lines) + pad * 2
    cv2.rectangle(frame, (0, 0), (max_w + pad * 2, box_h), bg, -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (pad, pad + line_h * (i + 1) - 4),
                    font, scale, color, thickness, cv2.LINE_AA)


def draw_target_zone(frame, nx, ny):
    """Draw a faint target rectangle in the center as aiming guide."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    # Approximate chessboard footprint at display size
    bw = int(w * 0.55)
    bh = int(h * 0.55)
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)


def direction_hint(target_label):
    """Return arrow / instruction string for where to move camera."""
    hints = {
        "CENTER":        "Move camera to CENTER of chessboard",
        "TOP-LEFT":      "Move camera toward TOP-LEFT corner",
        "TOP-RIGHT":     "Move camera toward TOP-RIGHT corner",
        "BOTTOM-LEFT":   "Move camera toward BOTTOM-LEFT corner",
        "BOTTOM-RIGHT":  "Move camera toward BOTTOM-RIGHT corner",
        "LEFT-CENTER":   "Slide camera to the LEFT",
        "RIGHT-CENTER":  "Slide camera to the RIGHT",
        "TOP-CENTER":    "Move camera UPWARD",
        "BOTTOM-CENTER": "Move camera DOWNWARD",
        "SLIGHT-TILT":   "Tilt camera ~15 degrees",
        "TILT-LEFT":     "Tilt camera LEFT and move slightly left",
        "TILT-RIGHT":    "Tilt camera RIGHT and move slightly right",
    }
    return hints.get(target_label, f"Move to {target_label}")


# ─── Blur detection ───────────────────────────────────────────────────────────

def is_blurry(gray, threshold=BLUR_THRESHOLD):
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold, lap_var


# ─── Chessboard detection (runs in worker thread) ─────────────────────────────

class DetectionResult:
    def __init__(self):
        self.lock    = threading.Lock()
        self.status  = "idle"   # idle | running | found | notfound | blurry
        self.corners = None
        self.message = ""


def detect_worker(frame_bgr, board_size, result: DetectionResult):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurry, var = is_blurry(gray)
    if blurry:
        with result.lock:
            result.status  = "blurry"
            result.message = f"Image too blurry (var={var:.1f}). Hold camera still."
        return

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        with result.lock:
            result.status  = "found"
            result.corners = corners
            result.message = "Chessboard detected!"
    else:
        with result.lock:
            result.status  = "notfound"
            result.message = "Chessboard NOT found. Reposition and press S again."


# ─── Calibration computation ──────────────────────────────────────────────────

def compute_calibration(obj_points, img_points, img_size, radial_only=True):
    """
    Compute camera matrix + only barrel (radial k1,k2,k3) distortion.
    radial_only=True fixes p1=p2=0 to avoid tangential fitting artefacts.
    """
    flags = cv2.CALIB_FIX_TANGENT_DIST   # p1 = p2 = 0
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None, flags=flags)
    return ret, K, dist, rvecs, tvecs


def per_image_error(obj_points, img_points, K, dist, rvecs, tvecs):
    errors = []
    for i, (op, ip) in enumerate(zip(obj_points, img_points)):
        projected, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(ip, projected, cv2.NORM_L2) / len(projected)
        errors.append(err)
    return errors


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    board_size = (args.cols, args.rows)   # inner corners
    nx, ny     = board_size

    # Prepare 3-D object points for one chessboard view
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    obj_points   = []   # 3-D points across accepted images
    img_points   = []   # 2-D points across accepted images
    accepted_raw = []   # raw full-res frames that were accepted
    img_size     = None

    # ── Open RTSP stream ──
    print(f"[INFO] Connecting to {args.rtsp} …")
    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open RTSP stream. Check URL / network.")

    WIN = "Camera Calibration (S=capture  Q=quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    pos_idx     = 0                     # which target position we're aiming for
    det_result  = DetectionResult()
    det_thread  = None
    state       = "guide"               # guide | waiting | done
    status_msg  = [""]                  # mutable list so lambda can write it
    last_key_t  = 0

    while True:
        ret, frame_full = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        if img_size is None:
            img_size = (frame_full.shape[1], frame_full.shape[0])

        display = resize_for_display(frame_full.copy())

        # ── State machine ──────────────────────────────────────────────────
        if state == "guide":
            if pos_idx >= len(TARGET_POSITIONS):
                state = "compute"
            else:
                label, hx, hy = TARGET_POSITIONS[pos_idx]
                hint           = direction_hint(label)
                n_ok           = len(obj_points)
                draw_target_zone(display, nx, ny)
                put_overlay(display, [
                    f"Step {pos_idx+1}/{len(TARGET_POSITIONS)}  ({n_ok} captured)",
                    hint,
                    "Press  S  when ready  |  Q to quit",
                ], color=(0, 255, 0))

        elif state == "waiting":
            with det_result.lock:
                s = det_result.status
                m = det_result.message

            if s == "running":
                put_overlay(display, ["Analyzing … please wait"], color=(255, 200, 0))

            elif s in ("found", "notfound", "blurry"):
                if s == "found":
                    with det_result.lock:
                        corners = det_result.corners
                    # Accept it
                    obj_points.append(objp)
                    img_points.append(corners)
                    accepted_raw.append(frame_full.copy())
                    pos_idx += 1
                    put_overlay(display, [
                        "Good capture! Moving to next position …"
                    ], color=(0, 255, 0))
                    cv2.imshow(WIN, display)
                    cv2.waitKey(1200)
                    det_result.status = "idle"
                    state = "guide"
                else:
                    put_overlay(display, [
                        m,
                        "Press  S  to try again  |  Q to quit",
                    ], color=(0, 80, 255))
                    det_result.status = "idle"
                    state = "guide"   # go back and let user re-press S

        elif state == "compute":
            put_overlay(display, ["Computing calibration … please wait"],
                        color=(255, 200, 0))
            cv2.imshow(WIN, display)
            cv2.waitKey(1)
            state = "calibrating"   # will run computation next iteration

        elif state == "calibrating":
            # ── Run calibration ──
            rms, K, dist, rvecs, tvecs = compute_calibration(
                obj_points, img_points, img_size)
            errors = per_image_error(obj_points, img_points, K, dist, rvecs, tvecs)
            mean_err = float(np.mean(errors))

            if mean_err <= MAX_REPROJ_ERROR:
                # Save and show undistorted sample
                np.savez(args.out, K=K, dist=dist, img_size=img_size,
                         rms=rms, mean_error=mean_err)
                sample = cv2.undistort(accepted_raw[-1], K, dist)
                sample_d = resize_for_display(sample)
                put_overlay(sample_d, [
                    f"SUCCESS!  mean error = {mean_err:.4f}",
                    f"Saved to: {args.out}",
                    "Press any key to exit.",
                ], color=(0, 255, 0))
                cv2.imshow(WIN, sample_d)
                cv2.waitKey(0)
                break

            else:
                # Find worst image and ask to retake it
                worst_idx = int(np.argmax(errors))
                worst_err = errors[worst_idx]
                label     = TARGET_POSITIONS[worst_idx % len(TARGET_POSITIONS)][0]
                put_overlay(display, [
                    f"Mean error {mean_err:.4f} > {MAX_REPROJ_ERROR}",
                    f"Worst image #{worst_idx+1} ({label}) error={worst_err:.4f}",
                    "Removing it. Press S to retake that position.",
                ], color=(0, 80, 255))
                cv2.imshow(WIN, display)
                cv2.waitKey(2500)

                # Remove worst image
                del obj_points[worst_idx]
                del img_points[worst_idx]
                del accepted_raw[worst_idx]

                # Queue that position to be retaken
                pos_idx = worst_idx % len(TARGET_POSITIONS)
                state   = "guide"

        # ── Show frame ────────────────────────────────────────────────────
        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ──────────────────────────────────────────────────
        if key == ord('q') or key == ord('Q'):
            print("[INFO] User quit.")
            break

        if key == ord('s') or key == ord('S'):
            if state == "guide" and (det_thread is None or not det_thread.is_alive()):
                # Snapshot the current full-res frame
                snap = frame_full.copy()
                det_result.status = "running"
                state  = "waiting"
                det_thread = threading.Thread(
                    target=detect_worker,
                    args=(snap, board_size, det_result),
                    daemon=True)
                det_thread.start()

        # ── Calibrate early if enough images and user presses Enter ───────
        if key == 13 and len(obj_points) >= MIN_GOOD_IMAGES and state == "guide":
            state = "compute"

    cap.release()
    cv2.destroyAllWindows()

    # ── Print summary ──────────────────────────────────────────────────────
    if os.path.exists(args.out):
        data = np.load(args.out)
        print("\n╔══════════════════════════════════════════╗")
        print("║        Calibration Result Summary        ║")
        print("╠══════════════════════════════════════════╣")
        print(f"║  Output file   : {args.out:<24}║")
        print(f"║  RMS error     : {float(data['rms']):<24.6f}║")
        print(f"║  Mean repr err : {float(data['mean_error']):<24.6f}║")
        print(f"║  Image size    : {str(tuple(data['img_size'])):<24}║")
        print("║  Camera matrix (K):                      ║")
        for row in data['K']:
            print(f"║    {row}  ║")
        print(f"║  Distortion (k1 k2 p1 p2 k3):           ║")
        print(f"║    {data['dist'].ravel()}  ║")
        print("╚══════════════════════════════════════════╝")
        print("\nTo undistort any frame later:\n")
        print("  import cv2, numpy as np")
        print(f"  d = np.load('{args.out}')")
        print("  K, dist = d['K'], d['dist']")
        print("  undistorted = cv2.undistort(frame, K, dist)")


if __name__ == "__main__":
    main()