import numpy as np
import cv2
from skimage import io
import glob


for file in glob.glob('*.jpeg'):
    img = io.imread(file)
    cropped = img[100:img.shape[0]-100,:]
    img_hls = cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS)

    mask = cv2.inRange(img_hls, (0, 0, 0), (255, 245, 255))
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    moments = cv2.moments(closing, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']
    if dArea > 100:
        x = int(dM10 / dArea)
        y = int(dM01 / dArea) + 100
        cv2.circle(img, (x, y), 1, (0, 0, 0), 2)
        cv2.imshow('Original', img)
        cv2.waitKey(0)
