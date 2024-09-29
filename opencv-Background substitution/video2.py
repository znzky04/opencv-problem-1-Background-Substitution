import cv2
import numpy as np

def blur_edge(mask, d=31):
    h, w = mask.shape
    img_edge = cv2.Canny(mask, 100, 200)
    kernel = np.ones((d, d), np.uint8)
    img_dilate = cv2.dilate(img_edge, kernel, iterations=1)
    img_blur = cv2.blur(img_dilate, (d, d))
    img_blur /= 255.0
    return img_blur

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

    # Create mask and inverse mask for the green screen
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    filtered = cv2.bilateralFilter(src=mask_inv, d=9, sigmaColor=64, sigmaSpace=120)

    mask_inv_blurred = cv2.GaussianBlur(mask_inv, (1, 1), 0)

    kernel_size = 9
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask_inv_blurred, kernel, iterations=3)  # 膨胀操作
    mask_eroded = cv2.erode(mask_dilated, kernel, iterations=3)  # 腐蚀操作

    alpha_mask = cv2.addWeighted(mask_inv_blurred.astype(np.float32), 0.2, mask_eroded.astype(np.float32), 0.8, 0.0)

    # Use the alpha mask to extract the foreground (subject)
    foreground = cv2.bitwise_and(frame, frame, mask=filtered)

    # Apply morphological operations to reduce the green fringes
    kernel = np.ones((3, 3), np.uint8)
    alpha_mask = cv2.dilate(alpha_mask, kernel, iterations=1)  # This may cause the subject to slightly "inflate"
    alpha_mask = cv2.erode(alpha_mask, kernel, iterations=1)   # This can help reduce the "inflation"

    # Use the alpha mask to extract the foreground (subject)
    foreground = cv2.bitwise_and(frame, frame, mask=alpha_mask.astype(np.uint8))

    # Use the mask to extract the matching part of the background image
    background_portion = cv2.bitwise_and(background, background, mask=mask)

    # Merge foreground and background portions
    combined = cv2.add(foreground, background_portion)

    cv2.imshow('result', combined)

    # Exit loop by pressing 'q' or when the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close any open windows
vc.release()
cv2.destroyAllWindows()