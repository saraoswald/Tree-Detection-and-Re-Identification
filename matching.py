import final
import cv2
import matplotlib.pyplot as plt
from tree import *
import numpy as np
from sklearn.neighbors import NearestNeighbors

"""
matching.py

run with
    python matching.py

Takes two images, uses detect.py to find the trees, and then plots the similar points determined by SIFT and FLANN

"""

"""
drawlines
input: two images, lines between matching points, 2 arrays of matching points
source: http://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html#gsc.tab=0
"""
def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),5,color,-1)
        cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


if __name__ == '__main__':
    # paths to images to compare
    imgpath1, imgpath2 = 'data/tree2.jpg', 'data/tree3.jpg'

    # find trees in first img
    topLeft, bottomRight = detect.find_trees(imgpath1)
    tree1 = Tree(imgpath1, topLeft, bottomRight)

    # clip the first image
    x1, y1 = topLeft
    x2, y2 = bottomRight
    img1 = cv2.imread(imgpath1,0)
    img1 = img1[y1:y2,x1:x2] # clip image


    # find trees in second img
    topLeft, bottomRight = detect.find_trees(imgpath2)
    tree2 = Tree(imgpath2, topLeft, bottomRight)

    # clip the second image
    x1, y1 = topLeft
    x2, y2 = bottomRight
    img2 = cv2.imread(imgpath2,0) # open image
    img2 = img2[y1:y2,x1:x2] # clip image



    """
    Epipolar Geometry
    http://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html#gsc.tab=0
    """

    # Initiate SIFT detector
    sift = cv2.SIFT()
    MIN_MATCH_COUNT = 10

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # apply FLANN feature matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # get only good points by Lowe's paper
    good = []
    pts1 = []
    pts2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)


    # convert to floats
    p1 = []
    p2 = []
    for i in range(len(pts1)):
        p1.append([float(pts1[i][0]),float(pts1[i][1])])
        p2.append([float(pts2[i][0]),float(pts2[i][1])])
    p1 = np.array(p1,dtype=np.float32)
    p2 = np.array(p2,dtype=np.float32)

    #get fundamental matrix
    F, mask = cv2.findFundamentalMat(p1,p2,cv2.cv.CV_FM_RANSAC)

    # We select only inlier points
    pts1 = p1[mask.ravel()==1]
    pts2 = p2[mask.ravel()==1]


    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2, 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1, 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
