import cv2
import numpy as np
import glob
import os
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class CameraCalibrator:
    def __init__(self, image_dir, pattern_size, pattern_spacing, pattern_type='circle'):
        self.image_dir = image_dir
        self.pattern_size = pattern_size  # (columns, rows)
        self.pattern_spacing = pattern_spacing
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
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2) * self.pattern_spacing
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
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH |
                    cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST)
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
                img2_resized = cv2.resize(img2, (int(w * scale), int(h * scale)))
            else:
                img2_resized = img2
            window_name = "Reprojected Points on Camera 2"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, desired_width, desired_height)
            cv2.imshow(window_name, img2_resized)
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

    def reproject_points_cam1tocam2_known_extrinsics(self, idx, points_3d_cam1, R_1to2, t_1to2, K2, show=True, arun=False):
        """
        Reprojects 3D points from camera 1's coordinate system to camera 2's image plane,
        using the extrinsics from cam1 to cam2.
        
        Args:
            points_3d_cam1: (N, 3) array of 3D points in camera 1 coordinates
            R_1to2: (3, 3) rotation matrix from cam1 to cam2
            t_1to2: (3,) translation vector from cam1 to cam2
            K2: (3, 3) intrinsic matrix of camera 2
        
        Returns:
            points_2d_cam2: (N, 2) array of 2D points in camera 2 image coordinates
        """
        # Transform points from cam1 to cam2 coordinates
        points_3d_cam2 = (R_1to2 @ points_3d_cam1.T) + t_1to2.reshape(3,1)  # shape (3, N)
        # Project to cam2 image plane
        points_2d_hom = K2 @ points_3d_cam2  # shape (3, N)
        points_2d = (points_2d_hom[:2, :] / points_2d_hom[2, :]).T  # shape (N, 2)
        if arun:
            points_2d = points_2d + np.array([fx2 * t_arun[0], fy2 * t_arun[1]])
        cam2_images = sorted(glob.glob(os.path.join(self.cam2.image_dir, "*.jpg")) +
                             glob.glob(os.path.join(self.cam2.image_dir, "*.png")))
        img2 = cv2.imread(cam2_images[idx])
        for proj_pt  in points_2d:
            proj_pt = tuple(np.round(proj_pt).astype(int))
            cv2.circle(img2, proj_pt, 5, (0, 0, 255), -1)
        if show:
            desired_width = 1600
            desired_height = 900
            h, w = img2.shape[:2]
            scale = min(desired_width / w, desired_height / h)
            if scale < 1:
                img2_resized = cv2.resize(img2, (int(w * scale), int(h * scale)))
            else:
                img2_resized = img2
            window_name = "Reprojected Points on Camera 2"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, desired_width, desired_height)
            cv2.imshow(window_name, img2_resized)
            cv2.waitKey(800)
            cv2.destroyAllWindows()
        return points_2d

    
    def reproject_points_cam2tocam1_known_extrinsics(self, idx, points_3d_cam2, R_2to1, t_2to1, K1, show=True, arun=False):
        """
        Reprojects 3D points from camera 2's coordinate system to camera 1's image plane,
        using the extrinsics from cam2 to cam1.
        
        Args:
            points_3d_cam2: (N, 3) array of 3D points in camera 2 coordinates
            R_2to1: (3, 3) rotation matrix from cam2 to cam1
            t_2to1: (3,) translation vector from cam2 to cam1
            K1: (3, 3) intrinsic matrix of camera 1
        
        Returns:
            points_2d_cam1: (N, 2) array of 2D points in camera 1 image coordinates
        """
        # Transform points from cam2 to cam1 coordinates
        points_3d_cam1 = (R_2to1 @ points_3d_cam2.T) + t_2to1.reshape(3,1)  # shape (3, N)

        # Project to cam1 image plane
        points_2d_hom = K1 @ points_3d_cam1  # shape (3, N)
        points_2d = (points_2d_hom[:2, :] / points_2d_hom[2, :]).T  # shape (N, 2)
        if arun:
            points_2d =  points_2d - (np.array([fx1 * t_arun[0], fy1 * t_arun[1]]))
        cam1_images = sorted(glob.glob(os.path.join(self.cam1.image_dir, "*.jpg")) +
                             glob.glob(os.path.join(self.cam1.image_dir, "*.png")))
        img1 = cv2.imread(cam1_images[idx])
        for proj_pt  in points_2d:
            proj_pt = tuple(np.round(proj_pt).astype(int))
            cv2.circle(img1, proj_pt, 5, (0, 0, 255), -1)
        if show:
            desired_width = 1600
            desired_height = 900
            h, w = img1.shape[:2]
            scale = min(desired_width / w, desired_height / h)
            if scale < 1:
                img2_resized = cv2.resize(img1, (int(w * scale), int(h * scale)))
            else:
                img2_resized = img1
            window_name = "Reprojected Points on Camera 2"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, desired_width, desired_height)
            cv2.imshow(window_name, img2_resized)
            cv2.waitKey(800)
            cv2.destroyAllWindows()
        return points_2d

def compute_relative_extrinsics(rvec1, tvec1, rvec2, tvec2):
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = R2 @ R1.T
    t_rel = tvec2 - R_rel @ tvec1
    rvec_rel, _ = cv2.Rodrigues(R_rel)
    return rvec_rel, R_rel, t_rel

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

def stereo_calibrate(
    objpoints,          # List of 3D points in world coordinates (e.g., chessboard corners)
    imgpoints1,         # List of corresponding 2D points in camera 1 images
    imgpoints2,         # List of corresponding 2D points in camera 2 images
    cameraMatrix1,      # Intrinsic matrix for camera 1
    distCoeffs1,        # Distortion coefficients for camera 1
    cameraMatrix2,      # Intrinsic matrix for camera 2
    distCoeffs2,        # Distortion coefficients for camera 2
    imageSize           # (width, height) of calibration images
):
    """
    Perform stereo calibration to obtain relative extrinsics between two cameras.

    Returns:
        retval: RMS re-projection error
        R: Rotation matrix from cam1 to cam2
        T: Translation vector from cam1 to cam2
        E: Essential matrix
        F: Fundamental matrix
    """
    flags = cv2.CALIB_FIX_INTRINSIC # Use this if intrinsics are known/fixed

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, \
    R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        cameraMatrix1,
        distCoeffs1,
        cameraMatrix2,
        distCoeffs2,
        imageSize,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-8),
        flags=flags
    )

    return retval, R, T, E, F

def normalize_points(points, K):
    # Ensure points is a 2D NumPy array
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)  # Convert single point to (1, D)
    
    K_inv = np.linalg.inv(K)
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    norm_points = (K_inv @ points_h.T).T
    return norm_points[:, :2] / norm_points[:, 2][:, None]

def unnormalize_points(norm_points, K):
    """
    Convert normalized coordinates to pixel coordinates using the camera matrix K.
    
    Args:
        norm_points: (N, 2) array of normalized coordinates [x', y']
        K: (3, 3) camera intrinsic matrix
    
    Returns:
        points: (N, 2) array of pixel coordinates [u, v]
    """
    # Ensure input is a 2D array
    norm_points = np.asarray(norm_points)
    if norm_points.ndim == 1:
        norm_points = norm_points.reshape(1, -1)
    
    # Add homogeneous coordinate (1)
    norm_points_h = np.hstack([norm_points, np.ones((norm_points.shape[0], 1))])
    
    # Apply K to map to pixel coordinates (homogeneous)
    points_h = (K @ norm_points_h.T).T  # Shape: (N, 3)
    
    # Convert to 2D by dividing by the third coordinate
    points = points_h[:, :2] / points_h[:, 2][:, None]
    return points

def arun_method(X1, X2):
    """
    X1: Nx3 array of 3D points in camera 1 coordinates
    X2: Nx3 array of corresponding 3D points in camera 2 coordinates
    Returns:
        R: 3x3 rotation matrix
        T: 3x1 translation vector
    """
    assert X1.shape == X2.shape

    # Compute centroids
    centroid_X1 = np.mean(X1, axis=0)
    centroid_X2 = np.mean(X2, axis=0)

    # Center the points
    Q1 = X1 - centroid_X1
    Q2 = X2 - centroid_X2

    # Cross-covariance matrix
    H = Q1.T @ Q2

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Translation
    T = centroid_X2 - R @ centroid_X1

    return R, T

def arun_similarity_transform(X1, X2):
    """
    Estimate scale, rotation, and translation to align X1 to X2.
    X1: (N, 3) array of 3D points in the first camera's frame
    X2: (N, 3) array of corresponding 3D points in the second camera's frame

    Returns:
        s: scale factor (float)
        R: (3, 3) rotation matrix
        T: (3,) translation vector
    """
    assert X1.shape == X2.shape
    N = X1.shape[0]

    # 1. Compute centroids
    centroid_X1 = np.mean(X1, axis=0)
    centroid_X2 = np.mean(X2, axis=0)

    # 2. Center the points
    Q1 = X1 - centroid_X1
    Q2 = X2 - centroid_X2

    # 3. Compute cross-covariance matrix
    H = Q1.T @ Q2

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 5. Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 6. Compute scale
    var1 = np.sum(Q1 ** 2)
    s = np.sum(S) / var1

    # 7. Compute translation
    T = centroid_X2 - s * R @ centroid_X1

    return s, R, T

def extract_angles_from_arun(R_arun, t_arun):
    # Extract rotation angle from 2D rotation matrix
    roll = np.arctan2(R_arun[1, 0], R_arun[0, 0])
    # Custom yaw and pitch from translation vector
    yaw = np.arcsin(np.clip(t_arun[0], -1.0, 1.0))
    pitch = np.arcsin(np.clip(t_arun[1], -1.0, 1.0))
    return roll, yaw, pitch

def visualize_alignment(X1, X2, s, R, T):
    # Transform X1 to X2 frame
    X1_aligned = (s * (R @ X1.T)).T + T

    plt.figure()
    plt.scatter(X1[:, 0], X1[:, 1], c='b', label='X1 (original)', alpha=0.5)
    plt.scatter(X1_aligned[:, 0], X1_aligned[:, 1], c='g', marker='^', label='X1 (aligned)')
    plt.scatter(X2[:, 0], X2[:, 1], c='r', label='X2 (target)', alpha=0.5)
    plt.legend()
    plt.title('2D Point Alignment')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

def euler_to_rotation_matrix(yaw, pitch, roll):
    """
    Convert yaw, pitch, roll (in radians) to a 3x3 rotation matrix using ZYX convention.
    """
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    Ry = np.array([
        [cp, 0, sp],
        [ 0, 1,  0],
        [-sp, 0, cp]
    ])
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    # Note: The order is Rz * Ry * Rx (ZYX)
    R = Rz @ Ry @ Rx
    return R

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
    pattern_spacing = 9e-3

    focal_length_1 = 50e-3
    pixel_size_1 = 9.3e-6
    resolution_x_1 = 3840
    resolution_y_1 = 2160

    focal_length_2 = 50e-3
    pixel_size_2 = 6.3e-6 # 17e-6
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
    cam1 = CameraCalibrator("CameraData/SyncedCollimatorImages/VIS/", pattern_size, pattern_spacing, pattern_type='circle')
    cam1.find_image_points(visualize=False)
    cam1.calibrate(intrinsics1)

    # Calibrate second camera
    cam2 = CameraCalibrator("CameraData/SyncedCollimatorImages/TIR4K/", pattern_size, pattern_spacing, pattern_type='circle')
    cam2.find_image_points(visualize=False)
    cam2.calibrate(intrinsics2)

    ################################################################################
    # # Average all rvecs and tvecs for each camera
    # rvec1_avg = average_rvecs(cam1.rvecs)
    # tvec1_avg = average_tvecs(cam1.tvecs)
    # rvec2_avg = average_rvecs(cam2.rvecs)
    # tvec2_avg = average_tvecs(cam2.tvecs)

    # # Compute relative extrinsics using the averaged values
    # rvec_rel, R_rel, t_rel = compute_relative_extrinsics(rvec1_avg, tvec1_avg, rvec2_avg, tvec2_avg)
    #####################################################################################################

    # retval, R_rel, t_rel, E, F = stereo_calibrate(cam1.objpoints, cam1.imgpoints, cam2.imgpoints,
    #                                       cam1.camera_matrix, cam1.dist_coeffs, cam2.camera_matrix,cam2.dist_coeffs, (3840, 2160))

    ###########################################################################
    cam1_first_points = np.array([arr[0, 0] for arr in cam1.imgpoints])
    cam2_first_points = np.array([arr[0, 0] for arr in cam2.imgpoints])

    cam1_norm_point = normalize_points(cam1_first_points, cam1.camera_matrix)
    cam2_norm_point = normalize_points(cam2_first_points, cam2.camera_matrix)

    scale_arun, R_arun, t_arun = arun_similarity_transform(cam1_norm_point, cam2_norm_point)

    # roll, yaw, pitch = extract_angles_from_arun(R_arun, t_arun)
        
    R_rel = np.eye(3)
    R_rel[:2, :2] = R_arun
        
    # t_arun_unnormalized = unnormalize_points(t_arun, cam1.camera_matrix)
    
    # R_rel = euler_to_rotation_matrix(yaw, pitch, roll)
    # t_rel = np.append(t_arun[0]*fx1, t_arun[1]*fy1)
    # t_rel = np.append(t_rel, 0)
    t_rel = np.array([0,0,0])

    visualize_alignment(cam1_norm_point, cam2_norm_point, scale_arun, R_arun, t_arun)
    #######################################################################################

    rvec_rel, _ = cv2.Rodrigues(R_rel)

    # print("Stereo RMS error:", retval)
    print("Rotation matrix:\n", R_rel)
    print("Translation vector:\n", t_rel)

    print("Relative Rotation Vector (rvec):\n", R_rel)
    yaw_deg, pitch_deg, roll_deg = rvec_to_yaw_pitch_roll(rvec_rel)
    print(f"Yaw: {yaw_deg:.2f}°, Pitch: {pitch_deg:.2f}°, Roll: {roll_deg:.2f}°")

    print("Relative Translation Vector (tvec):\n", t_rel)

    R12 = R_rel
    R21 = np.linalg.inv(R12)

    T12 = t_rel
    T21 = -T12

    # Reproject points from cam1 to cam2 and visualize
    reprojector = CameraReprojector(cam1, cam2)
    number_of_images = len(cam1.imgpoints)
    for i in range(0, number_of_images):
        # reproject using opencv
        # reprojector.reproject_points_cam1tocam2(idx=i, show=True)
        # reprojector.reproject_points_cam2tocam1(idx=i, show=True)

        # reproject using matrix multiplications
        # Convert rvec to rotation matrix
        R_obj2cam1, _ = cv2.Rodrigues(cam1.rvecs[i])
        t_obj2cam1 = cam1.tvecs[i].reshape(3, 1)
        # Transform object points to camera 1 coordinates
        points_3d_cam1 = (R_obj2cam1 @ (cam1.objpoints[i]).T + t_obj2cam1).T # shape (N, 3)
        
        reprojector.reproject_points_cam1tocam2_known_extrinsics(i, points_3d_cam1, R12, T12, cam2.camera_matrix, arun=True)

        # Convert rvec to rotation matrix
        R_obj2cam2, _ = cv2.Rodrigues(cam2.rvecs[i])
        t_obj2cam2 = cam2.tvecs[i].reshape(3, 1)
        # Transform object points to camera 1 coordinates
        points_3d_cam2 = (R_obj2cam2 @ (cam2.objpoints[i]).T + t_obj2cam2).T  # shape (N, 3)
        
        reprojector.reproject_points_cam2tocam1_known_extrinsics(i, points_3d_cam2, R21, T21, cam1.camera_matrix, arun=True)
