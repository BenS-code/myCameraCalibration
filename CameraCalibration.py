import os
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CameraCalibrator:
    def __init__(self, rows, cols, square_size):
        # Initialize calibrator with chessboard dimensions and square size
        self.rows = rows
        self.cols = cols
        self.square_size = square_size  # Size of a square in the chessboard (in your preferred unit)
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = self.create_objp()
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.corners_list = []  # Store corners for each image
        self.mtx = None  # Camera matrix
        self.dist = None  # Distortion coefficients
        self.rvecs = None  # Rotation vectors
        self.tvecs = None  # Translation vectors

    def create_objp(self):
        # Create object points for the chessboard
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2) * self.square_size
        return objp

    def find_corners(self, image_path):
        # Find chessboard corners in the given image
        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (self.cols, self.rows), None)
        if ret:
            self.objpoints.append(self.objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            self.imgpoints.append(corners2)
            self.corners_list.append(corners2)  # Store corners for each image
            cv.drawChessboardCorners(img, (self.cols, self.rows), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(100)
        return gray.shape[::-1]

    def calibrate(self, image_size):
        # Perform camera calibration
        print('Calibrating...')
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None)

    def undistort_image(self, img_path, output_path):
        # Undistort an image using the calibration results
        img = cv.imread(img_path)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite(output_path, dst)

    def calculate_error(self):
        # Calculate the reprojection error
        print('Calculating reprojection error...')
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        return mean_error / len(self.objpoints)

    def draw_axis(self, img, corners, imgpts):
        # Draw 3D axis on the image
        corner = tuple(map(int, corners[0].ravel()))
        img = cv.line(img, corner, tuple(map(int, imgpts[0].ravel())), (255,0,0), 5)
        img = cv.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0,255,0), 5)
        img = cv.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0,0,255), 5)
        return img

    def display_axis(self, image_path, index):
        # Display 3D axis on the image
        img = cv.imread(image_path)
        
        # Use stored corners
        corners2 = self.corners_list[index]
        
        # Use pre-calculated rvecs and tvecs
        rvec = self.rvecs[index]
        tvec = self.tvecs[index]
        
        # Project 3D points to image plane
        axis_length = self.square_size * 3  # Length of the axis arrows
        axis = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,-axis_length]]).reshape(-1,3)
        imgpts, _ = cv.projectPoints(axis, rvec, tvec, self.mtx, self.dist)
        
        img = self.draw_axis(img, corners2, imgpts)
        cv.imshow('img', img)
        cv.waitKey(100)

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
        width, height = image_size
        image_corners = np.array([
            [-width/2, -height/2, 1],
            [width/2, -height/2, 1],
            [width/2, height/2, 1],
            [-width/2, height/2, 1],
            [-width/2, -height/2, 1]
        ])
        ax.plot(image_corners[:, 0], image_corners[:, 1], image_corners[:, 2], 'k-', label='Image Plane')

        # Plot chessboard planes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.rvecs)))
        for rvec, tvec, color in zip(self.rvecs, self.tvecs, colors):
            R, _ = cv.Rodrigues(rvec)
            R = R.T  # Transpose for numpy multiplication
            t = -R @ tvec.reshape(3, 1)
            
            # Create chessboard corners
            board_size = (self.cols - 1) * self.square_size
            board_corners = np.array([
                [0, 0, 0],
                [board_size, 0, 0],
                [board_size, board_size, 0],
                [0, board_size, 0],
                [0, 0, 0]
            ])
            
            # Transform corners to camera coordinate system
            transformed_corners = (R @ board_corners.T + t).T
            
            # Plot the transformed chessboard
            ax.plot(transformed_corners[:, 0], transformed_corners[:, 1], transformed_corners[:, 2], 
                    color=color, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Camera and Chessboard Planes')
        plt.show()


def main():
    # Main function to run the camera calibration process
    images_path = "cam1_images/"
    results_path = "results/"

    # Check if directories exist, if not create them
    for path in [images_path, results_path, results_path + images_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            # print(f"Directory already exists: {path}")
            pass

    rows = 6
    cols = 9
    square_size = 25.0  # Size of a square in millimeters (adjust as needed)
    
    calibrator = CameraCalibrator(rows, cols, square_size)
    
    images = glob.glob(images_path + '*.jpg')
    image_size = None
    print('Finding chessboard corners...')
    for fname in images:
        image_size = calibrator.find_corners(fname)
    
    cv.destroyAllWindows()
    
    calibrator.calibrate(image_size)

    print(f"Camera intrinsics: \n{calibrator.mtx}")

    print('Undistorting images...')
    for fname in images:
        calibrator.undistort_image(fname, results_path + fname)
    
    total_error = calibrator.calculate_error()
    print(f"Total error: {total_error}")

    # Display axis on all images
    print('Displaying axes...')
    for i, fname in enumerate(images):
        calibrator.display_axis(fname, i)

    cv.destroyAllWindows()

    # Plot 3D visualization of camera and image planes
    print('Plotting 3D visualization...')
    calibrator.plot_camera_and_image_planes(image_size)

if __name__ == "__main__":
    # Clear the console and run the main function
    os.system('cls' if os.name == 'nt' else 'clear')
    print("------------------- Camera Calibration -------------------")
    main()
