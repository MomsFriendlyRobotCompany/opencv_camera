import numpy as np
import cv2
# from matplotlib import pyplot as plt
# from opencv_camera import StereoCamera
from colorama import Fore
import cv2.aruco as aruco
from glob import glob
# import pickle

def blurry(image, threshold=100.0):
    """
    Given an image and threshold, returns if image is blurry and its value
    
    Args:
      image: opencv grayscale image
      threshold: blur threshold value, below this value an image is considered
                 to be blurry
    Return:
      blurry: True/False
      value: numeric value of blurriness
      
    Reference: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    val = cv2.Laplacian(image, cv2.CV_64F).var()
    val = int(val)
    if val < threshold:
        return True, val
    return False, val


def get_images(path):
    """
    Given a path, it reads all images. This uses glob to 
    grab file names and excepts wild cards *
    Ex. getImages('./images/*.jpg')
    """
    imgsL = []
    imgsR = []
    files = glob(path)
    files.sort()  # put in order

    print("Found {} images at {}".format(len(tuple(files)), path))
    # print('-'*40)

    for i, f in enumerate(files):
        img = cv2.imread(f, 0)
        if img is None:
            print('>> Could not read: {}'.format(f))
            continue
        
        h, w = img.shape[:2]

        imgsL.append(img[:,:w//2])
        imgsR.append(img[:,w//2:])
    
    return imgsL, imgsR


class Calibrate:
    """Calibrate an individual camera"""
    def __init__(self, dictionary, board):
        self.dictionary = dictionary
        self.board = board
        
    def calculateReprojectionError(self, imgpoints, objpoints, rvecs, tvecs, mtx, dist):
        """
        imgpts: features found in image, (num_imgs, 2)
        objpts: calibration known features in 3d, (num_imgs, 3)
        """
        imgpoints = [c.reshape(-1,2) for c in imgpoints]
        mean_error = 0
        for i in range(len(objpoints)):
#             print('img',imgpoints[i].shape)
#             print('obj', objpoints[i].shape)
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints2 = imgpoints2.reshape(-1,2)
            
            # if not all markers were found, then the norm below will fail
            if len(imgpoints[i]) != len(imgpoints2):
                continue
                
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        # print( "total error: {}".format(mean_error/len(objpoints)) )
        return mean_error/len(objpoints)
        
    def calibrate(self, imgs, K=None):
        calcorners = []  # 2d points in image
        calids = []  # ids found in image
        h,w = imgs[0].shape[:2]
        
        # so we know a little bit about the camera, so
        # start off the algorithm with a simple guess
        f = max(h,w)  # focal length is a function of image size in pixels
        K = np.array([
            [f,0,w/2],
            [0,f,h/2],
            [0.,0.,1.]
        ])

        for im in imgs:
            # make grayscale if it is not already
            if len(im.shape) > 2:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            else:
                gray = im.copy()
                
            corners, ids, rejectedImgPts = aruco.detectMarkers(gray, self.dictionary)
            
            # if ids were found, then
            if ids is not None and len(ids) > 0:
                ret, chcorners, chids = aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board)
                
                calcorners.append(chcorners)
                calids.append(chids)


        flags = 0
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS  # make an inital guess at cameraMatrix (K)
#         flags |= cv2.CALIB_FIX_PRINCIPAL_POINT  # value? makes it worse
        rms, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            calcorners, calids, self.board, (w,h), K, None, flags=flags)
        
        cam_params = {
            'marker_type': 'aruco',
            'cameraMatrix': cameraMatrix,
            'distCoeffs': distCoeffs,
            'image_size': imgs[0].shape[:2],
#             'marker_size': (x,y),
#             'marker_scale:': sqr
            'rms': rms
        }
#         objpts = [self.board.chessboardCorners.copy() for c in calcorners]
        h, w = self.board.chessboardCorners.shape
        objpts = [self.board.chessboardCorners.reshape((h,1,3)) for c in calcorners]
#         imgpts = [c.reshape(-1,2) for c in calcorners]
        imgpts = calcorners
        
        # print('obj', len(objpts))
        # print('imgpts', len(imgpts))
        
        reproError = self.calculateReprojectionError(imgpts, objpts, rvecs, tvecs, cameraMatrix, distCoeffs)
        
        return (rms, cameraMatrix, distCoeffs, rvecs, tvecs, objpts, imgpts)