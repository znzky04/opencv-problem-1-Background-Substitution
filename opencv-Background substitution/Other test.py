import cv2
import matplotlib.pyplot as plt
img = cv2.imread('kana.jpg')

cv2.imshow("Image", img)

cv2.resizeWindow("Image", 1000, 1000)
cv2.waitKey(0)
cv2.destroyAllWindows()

resized_image = cv2.resize(img, (300, 200))
cropped_image = img[50:150, 50:250]

plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title('Resized Image')
plt.axis('off')
plt.show()

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')
plt.axis('off')
plt.show()

