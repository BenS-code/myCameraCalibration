import cv2
import numpy as np
import glob
import os

class CameraCalibrator:
    def __init__(self, image_dir, pattern_size, pattern_type='circle'):
        self.image_dir = image_dir
        self.pattern_size = pattern_size  # (columns, rows)
        self.pattern_type = pattern_type
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.image_shape = None

    def _get_object_points(self):
        objp = np.zeros((self.pattern_size[0]*self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        return objp

    def _preprocess(self, img):
        # Invert the binary image
        inverted = cv2.bitwise_not(img)  # or: inverted = 255 - binary
        return inverted

    def _robust_find_corners(self, gray):
        processed = self._preprocess(gray)
        if self.pattern_type == 'circle':
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
            ret, centers = cv2.findCirclesGrid(processed, self.pattern_size, flags=flags)
            if not ret:
                flags = cv2.CALIB_CB_ASYMMETRIC_GRID
                ret, centers = cv2.findCirclesGrid(processed, self.pattern_size, flags=flags)
            return ret, centers
        else:
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                     cv2.CALIB_CB_NORMALIZE_IMAGE |
                     cv2.CALIB_CB_FAST_CHECK)
            ret, corners = cv2.findChessboardCorners(processed, self.pattern_size, flags=flags)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(processed, corners, (5, 5), (-1, -1), criteria)
            return ret, corners

    def find_image_points(self, visualize=False):
        objp = self._get_object_points()
        images = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")) +
                        glob.glob(os.path.join(self.image_dir, "*.png")))
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.image_shape is None:
                self.image_shape = gray.shape[::-1]
            ret, centers = self._robust_find_corners(gray)
            if ret:
                self.objpoints.append(objp)
                self.imgpoints.append(centers)
                if visualize:
                    vis = img.copy()
                    if self.pattern_type == 'circle':
                        cv2.drawChessboardCorners(vis, self.pattern_size, centers, ret)
                    else:
                        cv2.drawChessboardCorners(vis, self.pattern_size, centers, ret)

                    desired_width = 1600
                    desired_height = 900
                    h, w = vis.shape[:2]
                    scale = min(desired_width / w, desired_height / h)
                    if scale < 1:
                        vis_resized = cv2.resize(vis, (int(w * scale), int(h * scale)))
                    else:
                        vis_resized = vis
                    window_name = f'Detected Corners: {fname}'
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, desired_width, desired_height)
                    cv2.imshow(window_name, vis_resized)
                    cv2.waitKey(100)
                    cv2.destroyAllWindows()
            else:
                print(f"Pattern not found in {fname}")

    def calibrate(self, known_intrinsic=None):
        if not self.objpoints or not self.imgpoints:
            raise ValueError("No image points or object points found.")
        
        # If a known intrinsic matrix is provided, use it and fix intrinsics during calibration
        if known_intrinsic is not None:
            flags = cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_INTRINSIC
            camera_matrix = np.array(known_intrinsic, dtype=np.float64)
        else:
            flags = 0
            camera_matrix = None

        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.image_shape,
            camera_matrix,
            None,
            flags=flags
        )
        print(f"Calibration RMS error: {ret}")
        print(f"Camera Matrix:\n{self.camera_matrix}")
        print(f"Distortion Coefficients:\n{self.dist_coeffs}")

class CameraReprojector:
    def __init__(self, cam1, cam2):
        self.cam1 = cam1
        self.cam2 = cam2

    def reproject_points_cam1tocam2(self, idx=0, show=True):
        objpoints1 = self.cam1.objpoints[idx]
        imgpoints2 = self.cam2.imgpoints[idx]
        rvec2, tvec2 = self.cam2.rvecs[idx], self.cam2.tvecs[idx]
        imgpoints2_proj, _ = cv2.projectPoints(
            objpoints1, rvec2, tvec2, self.cam2.camera_matrix, self.cam2.dist_coeffs
        )
        cam2_images = sorted(glob.glob(os.path.join(self.cam2.image_dir, "*.jpg")) +
                             glob.glob(os.path.join(self.cam2.image_dir, "*.png")))
        img2 = cv2.imread(cam2_images[idx])
        for proj_pt, orig_pt  in zip(imgpoints2_proj, imgpoints2):
            proj_pt = tuple(np.round(proj_pt[0]).astype(int))
            orig_pt = tuple(np.round(orig_pt[0]).astype(int))
            cv2.circle(img2, proj_pt, 5, (0, 0, 255), -1)
            cv2.circle(img2, orig_pt, 3, (0, 255, 0), -1)
        if show:
            desired_width = 1600
            desired_height = 900
            h, w = img2.shape[:2]
            scale = min(desired_width / w, desired_height / h)
            if scale < 1:
                img1_resized = cv2.resize(img2, (int(w * scale), int(h * scale)))
            else:
                img1_resized = img2
            window_name = "Reprojected Points on Camera 2"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, desired_width, desired_height)
            cv2.imshow(window_name, img1_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return imgpoints2_proj
    
    def reproject_points_cam2tocam1(self, idx=0, show=True):
        objpoints2 = self.cam2.objpoints[idx]
        imgpoints1 = self.cam1.imgpoints[idx]
        rvec1, tvec1 = self.cam1.rvecs[idx], self.cam1.tvecs[idx]
        imgpoints1_proj, _ = cv2.projectPoints(
            objpoints2, rvec1, tvec1, self.cam1.camera_matrix, self.cam1.dist_coeffs
        )
        cam1_images = sorted(glob.glob(os.path.join(self.cam1.image_dir, "*.jpg")) +
                             glob.glob(os.path.join(self.cam1.image_dir, "*.png")))
        img1 = cv2.imread(cam1_images[idx])
        for proj_pt, orig_pt  in zip(imgpoints1_proj, imgpoints1):
            proj_pt = tuple(np.round(proj_pt[0]).astype(int))
            orig_pt = tuple(np.round(orig_pt[0]).astype(int))
            cv2.circle(img1, proj_pt, 5, (0, 0, 255), -1)
            cv2.circle(img1, orig_pt, 3, (0, 255, 0), -1)
        if show:
            desired_width = 1600
            desired_height = 900
            h, w = img1.shape[:2]
            scale = min(desired_width / w, desired_height / h)
            if scale < 1:
                img1_resized = cv2.resize(img1, (int(w * scale), int(h * scale)))
            else:
                img1_resized = img1
            window_name = "Reprojected Points on Camera 1"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, desired_width, desired_height)
            cv2.imshow(window_name, img1_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return imgpoints1_proj

def compute_relative_extrinsics(rvec1, tvec1, rvec2, tvec2):
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = R2 @ R1.T
    t_rel = tvec2 - R_rel @ tvec1
    rvec_rel, _ = cv2.Rodrigues(R_rel)
    return rvec_rel, t_rel

def average_rvecs(rvecs):
        # Convert all rvecs to rotation matrices
        R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
        # Average the rotation matrices using Singular Value Decomposition (SVD)
        R_avg = np.mean(R_matrices, axis=0)
        # Re-orthogonalize using SVD
        U, _, Vt = np.linalg.svd(R_avg)
        R_avg = U @ Vt
        # Convert back to rvec
        rvec_avg, _ = cv2.Rodrigues(R_avg)
        return rvec_avg

def average_tvecs(tvecs):
    return np.mean(np.array(tvecs), axis=0)

def rvec_to_yaw_pitch_roll(rvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles from rotation matrix
    # Using the ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    
    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0], sy)
        roll = np.arctan2(R[2,1], R[2,2])
    else:
        # Gimbal lock case
        yaw = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        roll = 0

    # Convert radians to degrees for readability
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    roll_deg = np.degrees(roll)

    return yaw_deg, pitch_deg, roll_deg

if __name__ == "__main__":
    # Adjust pattern_size to your grid (columns, rows)
    pattern_size = (5, 4)  # circular grid

    focal_length_1 = 50e-3
    pixel_size_1 = 9.3e-6
    resolution_x_1 = 3840
    resolution_y_1 = 2160

    focal_length_2 = 50e-3
    pixel_size_2 = 6.044e-6 # 17e-6
    resolution_x_2 = 3840 # 1028
    resolution_y_2 = 2160 # 768

    # Calculate focal lengths in pixels
    fx1 = focal_length_1 / pixel_size_1
    fy1 = focal_length_1 / pixel_size_1
    cx1 = resolution_x_1 / 2
    cy1 = resolution_y_1 / 2

    fx2 = focal_length_2 / pixel_size_2
    fy2 = focal_length_2 / pixel_size_2
    cx2 = resolution_x_2 / 2
    cy2 = resolution_y_2 / 2

    intrinsics1 = np.array([
        [fx1,    0, cx1],
        [   0, fy1, cy1],
        [   0,    0,   1]
    ])

    intrinsics2 = np.array([
        [fx2,    0, cx2],
        [   0, fy2, cy2],
        [   0,    0,   1]
    ])

    # Calibrate first camera
    cam1 = CameraCalibrator("CameraData/SyncedCollimatorImages/VIS/", pattern_size, pattern_type='circle')
    cam1.find_image_points(visualize=True)
    cam1.calibrate(intrinsics1)

    # Calibrate second camera
    cam2 = CameraCalibrator("CameraData/SyncedCollimatorImages/TIR4K/", pattern_size, pattern_type='circle')
    cam2.find_image_points(visualize=True)
    cam2.calibrate(intrinsics2)

    # Average all rvecs and tvecs for each camera
    rvec1_avg = average_rvecs(cam1.rvecs)
    tvec1_avg = average_tvecs(cam1.tvecs)
    rvec2_avg = average_rvecs(cam2.rvecs)
    tvec2_avg = average_tvecs(cam2.tvecs)

    # Compute relative extrinsics using the averaged values
    rvec_rel, t_rel = compute_relative_extrinsics(rvec1_avg, tvec1_avg, rvec2_avg, tvec2_avg)
    print("Relative Rotation Vector (rvec):\n", rvec_rel)
    yaw_deg, pitch_deg, roll_deg = rvec_to_yaw_pitch_roll(rvec_rel)
    print(f"Yaw: {yaw_deg:.2f}°, Pitch: {pitch_deg:.2f}°, Roll: {roll_deg:.2f}°")

    print("Relative Translation Vector (tvec):\n", t_rel)

    # Reproject points from cam1 to cam2 and visualize
    reprojector = CameraReprojector(cam1, cam2)
    for i in range(0,len(cam1.imgpoints)):
        reprojector.reproject_points_cam1tocam2(idx=i, show=True)
        reprojector.reproject_points_cam2tocam1(idx=i, show=True)
