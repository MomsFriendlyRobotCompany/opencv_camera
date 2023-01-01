from opencv_camera import *
import opencv_camera
import cv2
import numpy as np
from pathlib import Path
import pytest
import os


def rm(fname):
    """Removes (deletes) a file or list of files"""
    if fname is None:
        # print(f"{Fore.RED}*** No file to remove ***{Fore.RESET}")
        return
    if not isinstance(fname, list):
        fname = [fname]
    for f in fname:
        try:
            os.remove(f)
            # print(f"{Fore.RED}- {f}{Fore.RESET}")
        except FileNotFoundError:
            # folder was already deleted or doesn't exist ... it's ok
            pass


def get():
    imgs = []

    p = Path(__file__).parent.absolute() / "cal_images"
    pp = p.glob("*.jpg")
    cal = [str(x) for x in pp]

    for i in cal:
        im = cv2.imread(i)
        imgs.append(im)

    print(f">> Found {len(imgs)} images")
    return imgs

# def test_info():
#     print("")
#     print(f">> opencv_camera version: {opencv_camera.__version__}")
#     print(f">> cv2 version: {cv2.__version__}")
#     print(f">> slurm version: {slurm.__version__}")
#
#     assert True


"""
https://kushalvyas.github.io/calib.html
Opencv cv2.calibrateCamera() function Camera Matrix:
[
    [532.79536563, 0, 342.4582516],
    [0, 532.91928339, 233.90060514],
    [0, 0, 1]
]
"""
def test_checkerboard_calibrate():
    print("")
    imgs = get()

    board = ChessboardFinder((9,6), 1)

    cal = CameraCalibration()
    data = cal.calibrate(imgs, board)
    assert (data['rms'] - 0.5882563398961391) < 1e-6

    print("camera matrix\n",data["K"],"\n")
    print("distortion coeff:",data["d"],"\n")
    # print("rms error:",data["rms"],"\n")

    # fb = FlipBook(cal.save_cal_imgs)
    # fb.run()
    #
    # cimgs=[]
    # d = UnDistort(data["distCoeffs"], data["cameraMatrix"])
    # for i in imgs:
    #     im = d.undistort(i)
    #     cimgs.append(im)
    #
    # fb = FlipBook(cimgs)
    # fb.run()

def test_save():
    print("")
    imgs = get()
    h,w = imgs[0].shape[:2]
    mpeg = SaveVideo()
    mpeg.open("single.mp4",w,h,fps=1)
    for i in imgs:
        mpeg.write(i)
    mpeg.close()

    p = Path("single.mp4")
    assert p.stat().st_size > 0 and p.exists()

    mpeg.write_list(imgs,fps=1, fname="batch.mp4")
    p = Path("batch.mp4")
    assert p.stat().st_size > 0 and p.exists()

    rm("batch.mp4")
    rm("single.mp4")

    assert True

def colorspace(a, b):
    p = Path(__file__).parent.absolute() / "cal_images/left01.jpg"
    im = cv2.imread( str(p) )
    g = a(im)
    im2 = b(g)
    assert np.array_equal(im, im2)
    assert np.array_equal(im.shape, im2.shape)

def test_colorspace():
    colorspace(bgr2hsv, hsv2bgr)
    colorspace(bgr2rgb, rgb2bgr)

def compressor(fmt, color):
    p = Path(__file__).parent.absolute() / "cal_images/left01.jpg"
    im = cv2.imread( str(p) )
    if not color:
        im = bgr2gray(im)
    print(">> Compressor:", fmt, color, im.shape)

    c = Compressor()
    c.format = fmt
    cimg = c.compress(im)
    uimg = c.uncompress(cimg,im.shape)

    if fmt in ["png", ".png"]:
        assert np.array_equal(im, uimg)
    assert np.array_equal(im.shape, uimg.shape)

def test_compressor():
    for fmt in ["png", "jpg", ".png", ".jpg"]:
        for color in [True, False]:
            compressor(fmt, color)

def test_fail_compressor():
    with pytest.raises(ValueError):
        compressor("jpeg", False)
