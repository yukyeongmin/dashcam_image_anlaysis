import cv2
import numpy as np

trapezoid = np.array([[ 870,  526],
       [1006,  638],
       [ 541,  526],
       [ 334,  638]])
trapezoid_ideal = np.array([[ 872,  526],
       [1022,  638],
       [ 572,  526],
       [ 422,  638]])
H, status = cv2.findHomography(trapezoid, trapezoid_ideal, cv2.RANSAC, 5.0)

imageA=cv2.imread('edgedashcam1.png')
before = []
for x in range(imageA.shape[1]):
    for y in range(imageA.shape[0]):
        point = [x, y, 1]
        before.append(point)
before = np.array(before).transpose()

after=np.matmul(H,before)
after = after / after[2, :]
after = after[:2, :]
after = np.round(after, 0).astype(np.int)

height, width, _ = imageA.shape
result = np.zeros((height, width * 2, 3), dtype = np.uint8)
for pt1, pt2 in zip(before[:2, :].transpose(), after.transpose()):
    if pt2[1] >= height:
        continue
 
    if np.sum(pt2 < 0) >= 1:
        continue
    
    result[pt2[1], pt2[0]] = imageA[pt1[1], pt1[0]]

cv2.imshow('origin', imageA)
cv2.imshow('result', result)  
cv2.waitKey()