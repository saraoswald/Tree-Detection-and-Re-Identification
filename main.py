import final
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import spline



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
SIFT feature detection
"""

img1 = cv2.imread('data/tree2.jpg',0)          # queryImage
img2 = cv2.imread('data/tree3.jpg',0) # trainImage

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

# """
# Sobel
# http://www.jayrambhia.com/blog/sobel-operator/
# """
# imgpath = "data/tree3.jpg" # which image to do
#
# # get image
# img1 = cv2.imread(imgpath,0)
#
# # reshape
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
# # gray = img1
# # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# # apply blur filter
# img = cv2.blur(res2,(1,1))
#
# # apply Sobel operator
# dst = cv2.Sobel(img, ddepth=cv2.cv.CV_8U, dx=1, dy=0, ksize=5)
# minv = np.min(dst)
# maxv = np.max(dst)
# cscale = 255/(maxv-minv)
# shift =  -1*(minv)
# sobel_img = np.zeros(img.shape,dtype='uint8')
# sobel_img = cv2.convertScaleAbs(dst, sobel_img, cscale, shift/255.0)
#
#
# # num_rows = float(len(sobel_img)) # get number of x values
# # # sum along y axis
# # vert_sum = np.sum(sobel_img,axis=0)
# # # make it an average value (divide by # of x values)
# # vert_sum = np.divide(vert_sum,num_rows)
# #
# #
# # x = np.arange(0,len(vert_sum)) # for graphing
# # xnew = np.arange(0,len(vert_sum),50) # for smoothing
# # #smooth
# # y_smooth = spline(x, vert_sum, xnew)
# #
# # #make a sin curve 300px wide
# # z = np.arange(0,300,1)
# # def f(x):
# #     return np.sin(x/90)*-10 + 25
# #
# # f = [f(i) for i in z] # make sine into an array
# #
# # # convolve sine and the vertical sum
# # y_conv = np.convolve(vert_sum, f,'same')
# # y_conv = np.convolve(y_conv, f,'same')
# #
# # # detect local minima, which are tree candidates
# # mins = (np.diff(np.sign(np.diff(y_conv))) > 0).nonzero()[0] + 1
#
# x_mins = final.xCoordinates(sobel_img)
# y_mins = final.yCoordinates(sobel_img)
#
# # process tree candidates
#
# # # graph
# # plt.title("Tree candidates for "+imgpath)
# # plt.xlabel("X Coordinate of pixels in image")
# # plt.ylabel("Average grayscale convolved with sine curve")
# #
# # # plt.plot(x,vert_sum,'g-',x,y_conv,'r-')
# # plt.plot(x,y_conv,'b-',mins,[y_conv[x] for x in mins],'o')
#
# # plot lines over image where minima are
# img = cv2.imread(imgpath)
# for i in x_mins:
#     cv2.line(img,(i,0),(i,len(sobel_img)),(255, 0, 0),2)
#
# cv2.imwrite('lines.png',img)
#
# plt.show()
#
# # laplacian = cv2.Laplacian(sobel_img,cv2.CV_64F)
# # cv2.imwrite('laplacian.png',laplacian)
