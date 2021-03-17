import cv2
import numpy as np
import os

# a = np.arange(1, 26).reshape(5, 5)
# kernel = np.ones((3,3),np.float32)/9
# print(cv2.blur(a, (3, 3)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
a = cv2.imread("../news_out/news-0000000.jpg")
cv2.imshow("a", a)
cv2.waitKey(0)
b = cv2.blur(a, (3, 3))
cv2.imshow("b", b)
cv2.waitKey(0)
cv2.destoryAllWindow()