import numpy as np
import cv2
from tqdm import tqdm

img1 = cv2.imread('SIFT1_img.jpg')
grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
sift1 = cv2.SIFT_create()
kp1, des1 = sift1.detectAndCompute(grey1, None)

img1 = cv2.drawKeypoints(img1, kp1, img1)
#cv2.imwrite('SIFT1_img_keypoints.jpg', img1)


img2 = cv2.imread('SIFT2_img.jpg')
grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift2 = cv2.SIFT_create()
kp2, des2 = sift2.detectAndCompute(grey2, None)
img2 = cv2.drawKeypoints(img2, kp2, img2)
#cv2.imwrite('SIFT2_img_keypoints.jpg', img2)

featureMap = np.empty((len(kp1), 2), dtype=type(kp1[0]))

for loop1, i in enumerate(tqdm(des1)):
    min = np.Infinity
    best_index = None
    best_dist = None

    for loop2, j in enumerate(des2):

        current_len = np.sqrt(np.sum((i-j)**2))

        if current_len < min:
            min = current_len
            best_index = loop2
            best_dist = min
    featureMap[loop1][1] = cv2.DMatch(loop1, best_index, best_dist)
    featureMap[loop1][0] = best_dist

featureMap = featureMap[featureMap[:, 0].argsort()]
flength = len(featureMap)/10
featureMap = featureMap[:int(flength)]

img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                       featureMap.T[1], outImg=img1)
cv2.imwrite('SIFT2_img_keypoints_match.jpg', img3)
