from scipy.ndimage import label
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
import time
from skimage.morphology import skeletonize

with np.load("lowrescalib.npz") as data:
    mtx, dist = data['K'], data['dist']

def showimg(img):
    (h, w) = len(img),len(img[0])

    # 2. Define new width and calculate new height
    new_width = 1720
    ratio = new_width / float(w)
    new_height = int(h * ratio)

    # 3. Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imshow("image",resized_img)


def test_thresholding_methods(img):
    # ==========================================
    # Method 2: Adaptive (Local) Thresholding
    # ==========================================
    # Parameters to tune for your specific wire:
    block_size = 15  # Size of a pixel neighborhood (must be an odd number: 3, 5, 7, 11, 15...)
    C = 5            # Constant subtracted from the mean. Increase this to remove more background noise.
    
    adaptive_result = cv2.adaptiveThreshold(
        img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Uses a weighted sum of neighborhood pixels
        cv2.THRESH_BINARY_INV,          # Inverts so dark wire is white
        block_size, 
        C
    )
    showimg(adaptive_result)
    cv2.waitKey(0)
    
def analyze_wire_segment(segment):
    global mtx
    global dist
    camera_matrix=mtx
    dist_coeffs=dist

    if segment is None or len(segment) < 2:
        return None

    # --- Undistort points ---
    undistorted = cv2.undistortPoints(
        segment.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        P=camera_matrix
    ).reshape(-1, 2)

    # --- Fit line ---
    [vx, vy, x0, y0] = cv2.fitLine(
        undistorted.reshape(-1, 1, 2).astype(np.float32),
        cv2.DIST_L2, 0, 0.01, 0.01
    )

    # --- Angle from horizontal ---
    angle_deg = np.degrees(np.arctan2(vy, vx))[0]
    if angle_deg < 0:
        angle_deg += 180

    # --- Length: distance from first to last point ---
    p_start = undistorted[0]
    p_end   = undistorted[-1]
    length_px = np.linalg.norm(p_end - p_start)

    return angle_deg, length_px,p_start, p_end, vx, vy, x0, y0, undistorted



def find_horizontal_segment(points_xy, max_y_deviation=5, min_segment_length=50):
    """
    Find longest segment where:
    - X is increasing (already sorted)
    - Y doesn't deviate more than max_y_deviation from a rolling average
    """
    if len(points_xy) == 0:
        return None

    xs = points_xy[:, 0]
    ys = points_xy[:, 1]

    best_start = 0
    best_end = 0
    current_start = 0

    for i in range(1, len(points_xy)):
        # Get Y reference from current segment start
        segment_ys = ys[current_start:i]
        y_ref = np.median(segment_ys)  # use median for robustness

        # If Y deviates too much, start a new segment
        if abs(ys[i] - y_ref) > max_y_deviation:
            # Check if current segment is the best so far
            if (i - 1 - current_start) > (best_end - best_start):
                best_start = current_start
                best_end = i - 1
            current_start = i  # start fresh from this point

    # Final check for last segment
    if (len(points_xy) - 1 - current_start) > (best_end - best_start):
        best_start = current_start
        best_end = len(points_xy) - 1

    segment = points_xy[best_start:best_end + 1]

    if len(segment) < min_segment_length:
        return None

    return segment



    
def score_wire_candidate(mask: np.ndarray,
                          img_w: int,
                          img_h: int) -> dict:

    skeleton = skeletonize(mask // 255).astype(np.uint8) * 255
    points = np.column_stack(np.where(skeleton > 0))
    points_xy = points[:, ::-1].astype(np.float32)  # (x, y)

    # Sort by X (left to right)
    points_xy = points_xy[np.argsort(points_xy[:, 0])]
    segment = find_horizontal_segment(points_xy, max_y_deviation=8, min_segment_length=30)
    
    result = analyze_wire_segment(segment)
    if result is None:
        return None
    
    angle_deg, length_px,p_start, p_end, vx, vy, x0, y0, undistorted=result


    angle_score = max(0.0, 1.0 - abs(angle_deg) / 45.0)
    coverage = length_px / img_w
    coverage_score = min(coverage, 1.0)
    score = (angle_score    * 0.50 +
             coverage_score * 0.50)

    return {
        "score":         score,
        "angle_deg":     angle_deg,
        "coverage":      coverage,
        "y_center":      float(y0[0]),
        "points":        segment,
    }

    # old one
    # ys, xs = np.where(mask)
    # if len(xs) < 10:
    #     return None

    # points = np.column_stack([xs, ys]).astype(np.float32)
    # # ── Fit a line ────────────────────────────────────────────────────────────
    # vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # vx, vy, cx, cy = float(vx[0]), float(vy[0]), float(cx[0]), float(cy[0])
    # angle_deg = float(np.degrees(np.arctan2(vy, vx)))
    # # Normalise to -90..90
    # angle_deg = (angle_deg + 90) % 180 - 90

    # # ── Horizontal angle penalty ──────────────────────────────────────────────
    # # score = 1 when angle = 0°, drops to 0 at ±45°
    # angle_score = max(0.0, 1.0 - abs(angle_deg) / 45.0)

    # # ── Width coverage ────────────────────────────────────────────────────────
    # x_span = int(xs.max()) - int(xs.min())
    # coverage = x_span / img_w
    # coverage_score = min(coverage / 0.8, 1.0)   # saturates at 80 % width

    # # ── Meijering response quality ────────────────────────────────────────────
    

    # # ── Thinness: low height spread relative to point count ───────────────────
    # y_spread = int(ys.max()) - int(ys.min())
    # thin_score = max(0.0, 1.0 - y_spread / (img_h * 0.05))

    # # ── Composite confidence ──────────────────────────────────────────────────
    # score = (angle_score    * 0.35 +
    #          coverage_score * 0.35 +
    #          1     * 0.15 +
    #          thin_score     * 0.15)

    # return {
    #     "score":         score,
    #     "angle_deg":     angle_deg,
    #     "coverage":      coverage,
    #     "y_center":      float(cy),
    #     "points":        points,
    # }


def findbestwire(binary,minwidthpercent):
    img_h, img_w = binary.shape[:2]
    um_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    widths  = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas   = stats[1:, cv2.CC_STAT_AREA]

    # Wires are thin and long → high aspect ratio, low area-to-bbox ratio
    aspect_ratio = widths / (heights + 1e-5)   # wire: >> 1 (horizontal) or << 1 (vertical)
    fill_ratio   = areas / (widths * heights + 1e-5)  # wire: low fill (sparse bbox)

    # Table edges are VERY long and span near full image width
    img_width = binary.shape[1]

    wire_mask_flags = (
        (widths >= img_width * minwidthpercent) &        # not spanning full image (table edge)
        (aspect_ratio > 3) &                # elongated shape
        (fill_ratio < 0.4)                  # sparse (thin wire, not solid blob)
    )

    best = None
    mylabels=(np.where(wire_mask_flags))
    for lab in mylabels:
        mymask = np.isin(labels, lab+1).astype(np.uint8) * 255
        candidate = score_wire_candidate(mymask, img_w, img_h)
        if candidate is None:
            continue
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    
    return best

    # wire_labels = np.where(wire_mask_flags)[0] + 1
    # wire_mask = np.isin(labels, wire_labels).astype(np.uint8) * 255
    # return wire_mask


def _draw_wire_points(img, undist_points, thickness=2):
    """
    Paint the actual undistorted wire pixel positions onto the image in red.
    Pixel-accurate — no fitted line involved.

    undist_points : (N, 2) float array of (x, y) in undistorted image space.
    thickness     : dilation radius so the wire stays visible (1=single pixel).
    """
    if undist_points is None or len(undist_points) == 0:
        return

    img_h, img_w = img.shape[:2]
    xs = np.clip(undist_points[:, 0].astype(np.int32), 0, img_w - 1)
    ys = np.clip(undist_points[:, 1].astype(np.int32), 0, img_h - 1)

    if thickness <= 1:
        img[ys, xs] = (0, 0, 255)
    else:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[ys, xs] = 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
        img[mask > 0] = (0, 0, 255)

def undistort_points(points_xy: np.ndarray,
                     camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistort a set of 2-D points using the camera intrinsics.
    points_xy : (N, 2) float array  [x, y]
    Returns   : (N, 2) float array of corrected points
    """
    if len(points_xy) == 0:
        return points_xy
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    corrected = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    return corrected.reshape(-1, 2)


def _draw_wire(img, vx, vy, cx, cy, img_w, confidence):
    """Draw a red line across the full image width."""
    if abs(vx) < 1e-6:
        return
    t_left  = (0  - cx) / vx
    t_right = (img_w - cx) / vx
    pt1 = (0,     int(cy + t_left  * vy))
    pt2 = (img_w, int(cy + t_right * vy))

    cv2.line(img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    
def detect_wire(frame,callbackimg):
    global mtx
    global dist
    img_h, img_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    thresh_sauvola = threshold_sauvola(gray, window_size=15)
    binary_sauvola = gray < thresh_sauvola
    output = (binary_sauvola * 255).astype(np.uint8)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    binary = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    
    callbackimg(binary)

    best=findbestwire(binary,minwidthpercent=0.3)
    
    if best is None or best["score"] < 0.45:
        # No wire — return original with overlay
        return None

    # ── 6. Undistort the detected wire *points* ───────────────────────────────
    raw_pts = best["points"]                       # (N, 2) in distorted image
    corrected_pts = undistort_points(raw_pts, mtx, dist)

    # ── 7. Fit a clean line through undistorted points ────────────────────────
    # vx, vy, cx, cy = cv2.fitLine(
    #     corrected_pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    # vx, vy, cx, cy = float(vx[0]), float(vy[0]), float(cx[0]), float(cy[0])

    # Compute undistorted full image for output (visualisation only)
    result = cv2.undistort(frame, mtx, dist)
    _draw_wire_points(result,corrected_pts)
    # ── 8. Draw the wire in red ───────────────────────────────────────────────
    
    # _draw_wire(result, vx, vy, cx, cy, img_w, best["score"])

    # info = {
    #     "score":         best["score"],
    #     "angle_deg":     best["angle_deg"],
    #     "coverage":      best["coverage"],
    #     "y_center":      best["y_center"],
    #     "line_params":   (float(vx), float(vy), float(cx), float(cy)),
    # }
    # marked, confidence, angle, y_len, coverage

    return result, best["score"],best["angle_deg"],best["y_center"],best["coverage"]


if __name__ == "__main__":
    img=cv2.imread("tripimage/image5.png")
    def something(data):
        pass
    prev=time.perf_counter()
    output=detect_wire(img,something)
    if output is None:
        print("Nothing found")
        exit()
    result, score,angle, y_center, coverage=output
    print("Time:"+str(time.perf_counter()-prev))
    print(score)
    showimg(result)
    cv2.waitKey(0)
