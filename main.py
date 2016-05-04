

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('trunk.jpg',0)          # queryImage
img2 = cv2.imread('tree2.jpg',0) # trainImage


"""
drawMatches source: http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
"""

def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


"""
Kmeans clustering
"""
# Z = img1.reshape((-1,3))
# # convert to np.float32
# Z = np.float32(Z)
#
# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 2
# ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img1.shape))
# gray = img1
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
#
# cv2.imwrite('clustered.png',res2)


"""
Image Segmentation
"""
# gray = cv2.imread('treetrunk.jpg',0)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
#
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
#
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
#
# cv2.imwrite('sure_bg.png',sure_bg)
# cv2.imwrite('sure_fg.png',sure_fg)



# """
# find contours
# """
#
# im = cv2.imread('clustered.png')
# imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im,contours,-1,(0,255,0),0)
# cv2.imwrite("contours.png", im)


# """
# fit line
# """
#
# cnt = contours[0]
# M = cv2.moments(cnt)
# img = cv2.imread('tree3.jpg',0)
# rows,cols = img.shape[:2]
# [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

#
# """
# Hough Line Transformation
# """
#
img = cv2.imread('tree2.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)

cv2.imwrite('houghlines5.jpg',img)



# Initiate SIFT detector
orb = cv2.ORB()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = drawMatches(img1,kp1,img2,kp2,matches[:20])
cv2.imwrite('matches.png',img3)
# plt.imshow(img3),plt.show()


"""
train classifier
"""

# color = [0, 255, 0]
# thickness   = 2
# cascadeFile = './opencv-haar-classifier-training/trained_classifiers/tree_classifier.xml'
#
# for i in range(11):
#     fileName = './opencv-haar-classifier-training/positive_images/pos'+str(i+1)+'.png'
#     im = cv2.imread(fileName)
#     trees = cv2.CascadeClassifier(cascadeFile)
#     objects = trees.detectMultiScale(im, 1.3, 5)
#     for (x, y, w, h) in objects:
#         im = cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
#     cv2.imshow('im',im)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


"""
Sorbel
http://www.jayrambhia.com/blog/sobel-operator/
"""
img = cv2.imread('clustered.png',0)
img = cv2.blur(img,(7,7))
dst = cv2.Sobel(img, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=1)
minv = np.min(dst)
maxv = np.max(dst)
cscale = 255/(maxv-minv)
shift =  -1*(minv)
sobel_img = np.zeros(img.shape,dtype='uint8')
sobel_img = cv2.convertScaleAbs(dst, sobel_img, cscale, shift/255.0)
cv2.imwrite('sobel32fblur7.png',sobel_img)
