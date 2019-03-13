import imutils
import cv2
from panorama import Stitcher

imageA = cv2.imread("./img/first.jpg")
imageB = cv2.imread("./img/second.jpg")
imageC = cv2.imread("./img/third.jpg")
imageD = cv2.imread("./img/fourth.jpg")
imageE = cv2.imread("./img/fifth.jpg")

imageA = imutils.resize(imageA, width=200)
imageB = imutils.resize(imageB, width=200)
imageC = imutils.resize(imageC, width=200)
imageD = imutils.resize(imageD, width=200)
imageE = imutils.resize(imageE, width=200)

images = [imageA, imageB, imageC, imageD, imageE]

Stitcher(images, showMatches=False)

