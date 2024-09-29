import cv2
import numpy as np

background = cv2.imread('bookshelf.jpg')
vc = cv2.VideoCapture('video.mp4')
ret, frame = vc.read()
background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

while ret:
    ret, frame = vc.read()
    if frame is None:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 100, 50])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    mask_inv_blurred = cv2.GaussianBlur(mask_inv, (1, 1), 0)

    mask_inv_blurred = cv2.bilateralFilter(mask_inv_blurred, 9, 60, 90)

    foreground = cv2.bitwise_and(frame, frame, mask=mask_inv_blurred)

    background_portion = cv2.bitwise_and(background, background, mask=mask)

    combined = cv2.add(foreground, background_portion)

    cv2.imshow('result', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
