import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CameraCalibrator:
    def __init__(self, rows, cols, square_size, pixel_size_x, pixel_size_y, scale_factor):
        # Initialize calibrator with chessboard dimensions and square size
        self.rows = rows
        self.cols = cols
        self.square_size = square_size  # Size of a square in the chessboard (in your preferred unit)
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.scale_factor = scale_factor
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = self.create_objp()
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.img_list = [] # list of images of which corners were detected
        self.mtx = None  # Camera matrix
        self.scaled_mtx = None
        self.dist = None  # Distortion coefficients
        self.rvecs = None  # Rotation vectors
        self.tvecs = None  # Translation vectors
        self.principal_point_mm = []
        self.focal_length_x_mm = []
        self.focal_length_y_mm = []


    def create_objp(self):
        # Create object points for the chessboard
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2) * self.square_size
        return objp
    
    def resize_image(self, image):
        h, w = image.shape[:2]
        aspect_ratio = w / h

        new_width = int(self.scale_factor * w)
        new_height = int(new_width / aspect_ratio)

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA) 
    
    def imshow(self, img, win_resize=True):
        if win_resize:
            cv2.namedWindow('img',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('img', 1200,800)
        cv2.imshow('img', img)
        cv2.waitKey(100)


    def find_corners(self, image_path):
        # Find chessboard corners in the given image
        img = cv2.imread(image_path)
        resized_image = self.resize_image(img)
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + \
            cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_EXHAUSTIVE
        ret, corners = cv2.findChessboardCornersSB(gray, (self.cols, self.rows), flags)
        if ret:
            print('Corners found in: ', image_path)
            self.objpoints.append(self.objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            self.img_list.append(image_path)
            self.imgpoints.append(corners2) # Store corners for each image
            cv2.drawChessboardCorners(resized_image, (self.cols, self.rows), corners2, ret)
            self.imshow(resized_image)
        else: 
            print("Corners were not found in: ", image_path)
            self.imshow(resized_image)
        return gray.shape[::-1]

    def calibrate(self, image_size):
        # Perform camera calibration
        print('\nCalibrating camera instrinsics and extrinsics parameters...')
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None)
        
        self.scaled_mtx = self.mtx.copy()
        self.scaled_mtx[0,0] /= self.scale_factor 
        self.scaled_mtx[1,1] /= self.scale_factor 
        self.scaled_mtx[0,2] /= self.scale_factor 
        self.scaled_mtx[1,2] /= self.scale_factor 

        self.focal_length_x_mm = self.scaled_mtx[0,0] * self.pixel_size_x * 1000
        self.focal_length_y_mm = self.scaled_mtx[1,1] * self.pixel_size_y * 1000
        self.principal_point_mm = (self.scaled_mtx[0,2] * self.pixel_size_x * 1000, self.scaled_mtx[1,2] * self.pixel_size_y * 1000)

    def calculate_error(self):
        # Calculate the reprojection error
        print('\nCalculating reprojection error...')
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        return mean_error / len(self.objpoints)

    def undistort_image(self, img_path, output_path):
        # Undistort an image using the calibration results
        img = cv2.imread(img_path)
        resized_image = self.resize_image(img)
        h, w = resized_image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv2.undistort(resized_image, self.mtx, self.dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(output_path, dst)

    def draw_axis(self, img, corners, imgpts):
        # Draw 3D axis on the image
        corner = tuple(map(int, corners[0].ravel()))
        img = cv2.line(img, corner, tuple(map(int, imgpts[0].ravel())), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0,0,255), 5)
        return img

    def display_axis(self, image_path, index):
        # Display 3D axis on the image
        img = cv2.imread(image_path)
        resized_image = self.resize_image(img)
        # Use stored corners
        corners2 = self.imgpoints[index]
        
        # Use pre-calculated rvecs and tvecs
        rvec = self.rvecs[index]
        tvec = self.tvecs[index]
        
        # Project 3D points to image plane
        axis_length = self.square_size * 3  # Length of the axis arrows
        axis = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,-axis_length]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.mtx, self.dist)
        
        img = self.draw_axis(resized_image, corners2, imgpts)
        self.imshow(resized_image)

    def plot_camera_and_image_planes(self, image_size):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Camera center
        ax.scatter(0, 0, 0, color='r', s=10, label='Camera Center')

        # Camera axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.5)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.5)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.5)

        # Image plane
        width, height = self.principal_point_mm[0] * 2, self.principal_point_mm[1] * 2
        image_corners = np.array([
            [-width/2, -height/2, 1],
            [width/2, -height/2, 1],
            [width/2, height/2, 1],
            [-width/2, height/2, 1],
            [-width/2, -height/2, 1]
        ])
        ax.plot(image_corners[:, 0], image_corners[:, 1], image_corners[:, 2], 'k-', label='Image Plane')

        # Create chessboard corners
        board_size_x, board_size_y = (self.cols + 1) * self.square_size, (self.rows + 1) * self.square_size
        board_corners = np.array([
            [-self.square_size, -self.square_size, 0],
            [board_size_x - self.square_size, -self.square_size, 0],
            [board_size_x - self.square_size, board_size_y - self.square_size, 0],
            [-self.square_size, board_size_y - self.square_size, 0],
            [-self.square_size, -self.square_size, 0]
        ])

        print(f"\nBoard size: ({board_size_x, board_size_y}) mm")

        # Plot chessboard planes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.rvecs)))
        for rvec, tvec, color, label in zip(self.rvecs, self.tvecs, colors, self.img_list):
            label = label[label.rfind("/")+1:label.rfind(".")]
            R, _ = cv2.Rodrigues(rvec)     
            
            # Transform corners to camera coordinate system
            transformed_corners = np.dot(R, self.objp.T).T + tvec.T
            transformed_board_corners = np.dot(R, board_corners.T).T + tvec.T

            # Plot the transformed corners
            ax.scatter(transformed_corners[:, 0], transformed_corners[:, 1], transformed_corners[:, 2], 
                    s=1, color=color, alpha=0.5)
            ax.text(transformed_corners[self.cols-1][0], transformed_corners[self.cols-1][1], transformed_corners[self.cols-1][2],
                    label,
                    fontsize=8,
                    color=color)

            # Plot the transformed chessboard corners
            ax.plot(transformed_board_corners[:, 0], transformed_board_corners[:, 1], transformed_board_corners[:, 2], 
                    color=color, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Camera and Chessboard Planes')
        ax.set_aspect('equal')
        plt.show()
    

def main():
    # Main function to run the camera calibration process
    images_root_path = "CameraData/"
    images_path = images_root_path + "focal18mm_square20mm/"
    file_type = "*.JPG"
    # file_type = "*.jpg"
    # file_type = "*.png" 
    # file_type = "*.tiff"
    results_path = "results/"

    # Check if directories exist, if not create them
    for path in [images_path, results_path, results_path + images_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            # print(f"Directory already exists: {path}")
            pass

    rows = 7 # number of corners in a arow
    cols = 10 # number of corners in a column
    square_size = 20.0  # Size of a square in millimeters
    scale_factor = 1.0 # scale the image
    pixel_size_x = 3.72  # pixel size in the x-direction in microns
    pixel_size_y = 3.72  # pixel size in the y-direction in microns
    
    calibrator = CameraCalibrator(rows, cols, square_size, pixel_size_x * 10**-6, pixel_size_y * 10**-6, scale_factor)
    
    images = glob.glob(images_path + file_type)
    images = images[:] # limit number of images in dataset
    image_size = None
    print('Finding chessboard corners...')
    for fname in images:
        image_size = calibrator.find_corners(fname)
    
    cv2.destroyAllWindows()

    print(f"\nCorners were found in {len(calibrator.img_list)} out of {len(images)} images")
    
    calibrator.calibrate(image_size)

    print(f"\nCamera intrinsics: \n{calibrator.scaled_mtx}")

    print(f"\nFocal length X = {np.round(calibrator.focal_length_x_mm, 4)} mm")
    print(f"Focal length Y = {np.round(calibrator.focal_length_y_mm, 4)} mm")
    print(f"Principal point = {np.round(calibrator.principal_point_mm, 4)} mm")

    total_error = calibrator.calculate_error()
    print(f"Total error: {total_error}")

    print(f"\nCamera extrinsics:")
    for i in range(len(calibrator.tvecs)):
        print(f"File name: {calibrator.img_list[i]}")
        print(f"Translation: {calibrator.tvecs[i].flatten()}")
        print(f"Rotation: {calibrator.rvecs[i].flatten()}\n")

    # Undistort images
    print('\nUndistorting images...')
    for fname in calibrator.img_list:
        calibrator.undistort_image(fname, results_path + fname)
    

    # Display axis on all images
    print('\nDisplaying axes...')
    for i, fname in enumerate(calibrator.img_list):
        calibrator.display_axis(fname, i)

    cv2.destroyAllWindows()

    # Plot 3D visualization of camera and image planes
    print('Plotting 3D visualization...')
    calibrator.plot_camera_and_image_planes(image_size)

if __name__ == "__main__":
    # Clear the console and run the main function
    os.system('cls' if os.name == 'nt' else 'clear')
    print("------------------- Camera Calibration -------------------")
    main()

