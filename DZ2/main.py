import cv2
import numpy as np
import matplotlib.pyplot as plt

# capture frames from a camera with device index=0
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized
while (1):

    # reads frame from a camera
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Camera', frame)

    img_gauss_49 = cv2.GaussianBlur(frame, ksize=(49, 49), sigmaX=0, sigmaY=0)
    result_subtraction = cv2.subtract(img_gauss_49, frame)
    cv2.imshow('Result', result_subtraction)

    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('Canny', edges)

    # Wait for 25ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera from video capture
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()