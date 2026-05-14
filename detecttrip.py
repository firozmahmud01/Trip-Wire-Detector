from scipy.ndimage import label
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
import time
import math
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.integrate import quad



# with np.load("highrescalib.npz") as data:
#     mtx, dist = data['K'], data['dist']

# map1=None
# map2=None

mtx=None
dist=None










def showimg(img,title="image"):
    (h, w) = len(img),len(img[0])

    # 2. Define new width and calculate new height
    new_width = 1720
    ratio = new_width / float(w)
    new_height = int(h * ratio)

    # 3. Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imshow(title,resized_img)



    
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




def get_midpoint(p1, p2):
    # p1 and p2 are tuples or lists like (x, y)
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2
    return (mx, my)


ransac=RANSACRegressor(residual_threshold=5.0)
def analyze_wire(binary_image):
    global ransac
    # 1. Extract coordinates of all white pixels (the wire + noise)
    y_coords, x_coords = np.where(binary_image > 0)
    
    if len(x_coords) < 10:
        return None

    X = x_coords.reshape(-1, 1)

    # 2. RANSAC + Polynomial Fit (Degree 2)
    # This ignores noise/attachments and fits the most dominant curve/line
    model = make_pipeline(PolynomialFeatures(degree=2), ransac)
    model.fit(X, y_coords)

    # 3. Extract the math coefficients
    # Formula: y = ax^2 + bx + c
    ransac = model.named_steps['ransacregressor']
    poly = model.named_steps['polynomialfeatures']
    
    # Coefficients are usually [1, x, x^2] in poly features, so:
    # coef_ will be [0, b, a]
    b = ransac.estimator_.coef_[1]
    a = ransac.estimator_.coef_[2]
    c = ransac.estimator_.intercept_

    # 4. Identify Matching Points (Inliers)
    # This separates the wire pixels from the table/noise pixels
    inlier_mask = ransac.inlier_mask_
    wire_points_x = x_coords[inlier_mask]
    wire_points_y = y_coords[inlier_mask]
    
    # 5. Calculate Length (Arc Length Integration)

    def integrand(x):
        return np.sqrt(1 + (2 * a * x + b)**2)

    x_min, x_max = wire_points_x.min(), wire_points_x.max()
    wire_length, _ = quad(integrand, x_min, x_max)

    # 6. Calculate Angle
    # We calculate the chord angle (start point to end point) for general orientation
    # This works perfectly for tilted 45 degree lines or straight lines.
    y_start = a * x_min**2 + b * x_min + c
    y_end = a * x_max**2 + b * x_max + c
    
    angle_rad = np.arctan2(y_end - y_start, x_max - x_min)
    angle_deg = np.degrees(angle_rad)

    return angle_deg,wire_length, (a, b, c),wire_points_x,wire_points_y
    

def score_wire_candidate(mask: np.ndarray,
                          img_w: int,
                          img_h: int) -> dict:

    data=analyze_wire(mask)
    if data is None:
        return None
    angle,length,abc,pointx,pointy=data

    if (angle>45 and angle<135) or (angle>225 and angle<315):
        return None


    # x1, y1, x2, y2 = line[0]
    # angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    
    # Filter for nearly horizontal lines (e.g., within 5 degrees of 0)
    # if angle < 45 or (angle > 135 and angle<225) or angle>315:
        
    angle_score = max(0.0, 1.0 - abs(angle) / 45.0)
    coverage = length / (img_w*0.8)
    coverage_score = min(coverage, 1.0)
    score = (angle_score    * 0.30 +
            coverage_score * 0.70)

    return{
        "score":         score,
        "angle_deg":     angle,
        "coverage":      coverage_score,
        "y_center":      pointy[0],
        "pointx":        pointx,
        "pointy":        pointy
    }


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


def _draw_wire_points(img, pointx,pointy, thickness=2):
    """
    Paint the actual undistorted wire pixel positions onto the image in red.
    Pixel-accurate — no fitted line involved.

    undist_points : (N, 2) float array of (x, y) in undistorted image space.
    thickness     : dilation radius so the wire stays visible (1=single pixel).
    """
    

    img_h, img_w = img.shape[:2]
    xs = pointx                 #np.clip(undist_points[:, 0].astype(np.int32), 0, img_w - 1)
    ys = pointy                 #np.clip(undist_points[:, 1].astype(np.int32), 0, img_h - 1)

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




def detect_wire(frame,callbackimg):
    global mtx
    global dist
    # global map1
    # global map2
    img_h, img_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    thresh_sauvola = threshold_sauvola(gray, window_size=15)
    binary_sauvola = gray < thresh_sauvola
    output = (binary_sauvola * 255).astype(np.uint8)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    binary = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)

    # if map1 is None or map2 is None:
    #     R = np.eye(3, dtype=np.float32)

    #     # 2. Compute optimal new camera matrix
    #     # alpha=0 crops, alpha=1 keeps all original pixels
    #     newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    #         mtx, dist, (img_w, img_h), alpha=0, newImgSize=(img_w, img_h)
    #     )
    #     map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, R, newCameraMatrix, (img_w,img_h), cv2.CV_32FC1)
    

    # undistorted_img = cv2.remap(binary, map1, map2, interpolation=cv2.INTER_NEAREST)
    callbackimg(binary)

    best=findbestwire(binary,minwidthpercent=0.3)
    
    if best is None:
        return None

    
    # result=frame.copy()

    # result = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_NEAREST)
    # _draw_wire_points(result,best['pointx'],best['pointy'])

    return best["pointx"],best["pointy"], best["score"],best["angle_deg"],best["y_center"],best["coverage"]

    # return result, best["score"],best["angle_deg"],best["y_center"],best["coverage"]


if __name__ == "__main__":
    img=cv2.imread("tripimage/image1.png")
    def something(data):
        showimg(data,"binary")
    prev=time.perf_counter()
    output=detect_wire(img,something)
    print("Time:"+str(time.perf_counter()-prev))
    if output is None:
        print("Nothing found")
        exit()
    result, score,angle, y_center, coverage=output
    print(score)
    
    showimg(result)
    cv2.waitKey(0)
