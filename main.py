import cv2
import numpy as np

original_image = cv2.imread('sample.png')
cv2.imshow("Original Image", original_image)

grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grey Image", grey_image)

bw_image = cv2.adaptiveThreshold(
    ~grey_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
cv2.imshow("BW Image", bw_image)


horizontal = bw_image
vertical = bw_image
rows, cols = horizontal.shape

horizontal_size = int(cols / 20)
horizontal_structure = cv2.getStructuringElement(
    cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontal_structure, (-1, -1))
horizontal = cv2.dilate(horizontal, horizontal_structure, (-1, -1))
cv2.imshow("Horizontal", horizontal)

res = cv2.bitwise_xor(horizontal, bw_image)
cv2.imshow("result", ~res)

vertical_size = int(rows / 10)
vertical_structure = cv2.getStructuringElement(
    cv2.MORPH_RECT, (1, vertical_size))
vertical = cv2.erode(vertical, vertical_structure, (-1, -1))
vertical = cv2.dilate(vertical, vertical_structure, (-1, -1))
cv2.imshow("Vertical", vertical)

res = cv2.bitwise_xor(vertical, res)
cv2.imshow("result", res)
res=~res
kernel = np.ones((2,2))
res = cv2.dilate(res, kernel)
cv2.imshow("result_dilate", res)

res = cv2.blur(res, (2,2))
cv2.imshow("result_blur", res)

cv2.waitKey(0)
