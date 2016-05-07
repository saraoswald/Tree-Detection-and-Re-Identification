

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import spline

img1 = cv2.imread('tree2.jpg',0)          # queryImage
img2 = cv2.imread('tree3.jpg',0) # trainImage


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
# gray = cv2.imread('canny.png',0)
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
# im = cv2.imread('canny.png')
# imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im,contours,-1,(0,255,0),3)
# cv2.imwrite("contours.png", im)


#
# """
# Hough Line Transformation
# """
#
# img = cv2.imread('tree2.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,150,200,apertureSize = 3)
# minLineLength = 500
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,200,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imwrite('houghlines500.jpg',img)


"""
SIFT feature detection
"""

#
# # Initiate SIFT detector
# orb = cv2.ORB()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
#
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(des1,des2)
#
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
#
# # Draw first 10 matches.
# img3 = drawMatches(img1,kp1,img2,kp2,matches[:20])
# cv2.imwrite('matches.png',img3)
# # plt.imshow(img3),plt.show()


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
Sobel
http://www.jayrambhia.com/blog/sobel-operator/
"""
imgpath = 'tree2.jpg' # which image to do


# get image, reshape
img1 = cv2.imread(imgpath,0)
Z = img1.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img1.shape))
gray = img1
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# apply blur filter
img = cv2.blur(res2,(1,1))

# apply Sobel operator
dst = cv2.Sobel(img, ddepth=cv2.cv.CV_8U, dx=1, dy=0, ksize=5)
minv = np.min(dst)
maxv = np.max(dst)
cscale = 255/(maxv-minv)
shift =  -1*(minv)
sobel_img = np.zeros(img.shape,dtype='uint8')
sobel_img = cv2.convertScaleAbs(dst, sobel_img, cscale, shift/255.0)


_r = float(len(sobel_img)) # get number of x values
# sum along y axis
vert_sum = np.sum(sobel_img,axis=0)
# make it an average value (divide by # of x values)
vert_sum = np.divide(vert_sum,_r)


x = np.arange(0,len(vert_sum)) # for graphing
xnew = np.arange(0,len(vert_sum),50) # for smoothing
#smooth
y_smooth = spline(x, vert_sum, xnew)

#make a sin curve 300px wide
z = np.arange(0,300,1)
def f(x):
    return np.sin(x/90)*-15 + 25

f = [f(i) for i in z] # make sine into an array

# convolve sine and the vertical sum
y_conv = np.convolve(vert_sum, f,'same')

# detect local minima
mins = (np.diff(np.sign(np.diff(y_conv))) > 0).nonzero()[0] + 1

# graph
plt.title("Tree candidates for "+imgpath)
plt.xlabel("X Coordinate of pixels in image")
plt.ylabel("Average grayscale convolved with sine curve")

# plt.plot(x,vert_sum,'g-',x,y_conv,'r-')
plt.plot(x,y_conv,'b-',mins,[y_conv[x] for x in mins],'o')


plt.show()

# cv2.imwrite('sobel.png',sobel_img)

# laplacian = cv2.Laplacian(sobel_img,cv2.CV_64F)
# cv2.imwrite('laplacian.png',laplacian)
