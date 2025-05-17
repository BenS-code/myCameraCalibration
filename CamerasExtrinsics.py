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
        # 2. Apply binary threshold
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # 3. Invert the binary image
        inverted = cv2.bitwise_not(binary)  # or: inverted = 255 - binary
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

    def calibrate(self):
        if not self.objpoints or not self.imgpoints:
            raise ValueError("No image points or object points found.")
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_shape, None, None)
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

if __name__ == "__main__":
    # Adjust pattern_size to your grid (columns, rows)
    pattern_size = (5, 4)  # circular grid

    # Calibrate first camera
    cam1 = CameraCalibrator("CameraData/SyncedCollimatorImages/VIS/", pattern_size, pattern_type='circle')
    cam1.find_image_points(visualize=True)
    cam1.calibrate()

    # Calibrate second camera
    cam2 = CameraCalibrator("CameraData/SyncedCollimatorImages/TIR/", pattern_size, pattern_type='circle')
    cam2.find_image_points(visualize=True)
    cam2.calibrate()

    # Use the first view for each camera for extrinsic estimation
    rvec1, tvec1 = cam1.rvecs[0], cam1.tvecs[0]
    rvec2, tvec2 = cam2.rvecs[0], cam2.tvecs[0]

    rvec_rel, t_rel = compute_relative_extrinsics(rvec1, tvec1, rvec2, tvec2)
    print("Relative Rotation Vector (rvec):\n", rvec_rel)
    print("Relative Translation Vector (tvec):\n", t_rel)

    # Reproject points from cam1 to cam2 and visualize
    reprojector = CameraReprojector(cam1, cam2)
    for i in range(0,2):
        reprojector.reproject_points_cam1tocam2(idx=i, show=True)
        reprojector.reproject_points_cam2tocam1(idx=i, show=True)
