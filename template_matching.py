import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('invoice-example-FI.png',0)
# Initiate ORB detector
orb = cv.ORB_create(nfeatures=1000)
# find the keypoints with ORB
kp = orb.detect(img,None)

print len(kp)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img_orb = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv.imshow('img',img_orb)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

img2 = cv.imread('invoice-example-SE.png',0)
kp2 = orb.detect(img,None)

print len(kp2)
# compute the descriptors with ORB
kp2, des2 = orb.compute(img2, kp2)
# draw only keypoints location,not size and orientation
img2_orb = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)




index_params= dict(algorithm = 6,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>20:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),10)
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img,kp1,img2,kp2,good,None,**draw_params)

cv.imshow('img',img3)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
