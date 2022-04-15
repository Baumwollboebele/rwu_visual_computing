import cv2
from cv2 import imshow
from cv2 import imwrite
import os
import numpy as np
from imutils.perspective import four_point_transform
print(cv2.__version__)

img_path = os.path.join("src","images","doc2.jpg")
concat_path = os.path.join("src","images","solution","concat.jpg")
solution_path = os.path.join("src","images","solution","solution.jpg")
green = (0, 255, 0)

height = 800
width = int(np.floor(height*(1/np.sqrt(2))))

img = cv2.imread(img_path)
img = cv2.resize(img, (width, height))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
edge = cv2.Canny(blur,75,200)

contours, _ = cv2.findContours(edge,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)

    if len(approx) == 4:
        doc_cnts = approx
        break

warped = four_point_transform(img, doc_cnts.reshape(4, 2))
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = cv2.resize(warped, (width, height))

(thresh, blackAndWhiteImage) = cv2.threshold(warped, 165, 255, cv2.THRESH_BINARY)

img_h = cv2.hconcat([gray, edge, warped, blackAndWhiteImage])
imwrite(concat_path,img_h)

cv2.imshow("Original",img)
cv2.imshow("Canny Edges",edge)
cv2.imshow("Contours of the document", cv2.drawContours(img, [doc_cnts], -1, green, 3))
cv2.imshow("Scanned",warped )
cv2.imshow("Binary", blackAndWhiteImage)

cv2.imwrite(solution_path,cv2.resize(warped, (width, height)))
cv2.waitKey(0)
cv2.destroyAllWindows()