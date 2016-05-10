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

Takes two images, uses final.py to find the trees, and then plots the similar points determined by SIFT

"""



"""
drawMatches source:
http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
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
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 2)

    return out

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
    topLeft, bottomRight = final.find_trees(imgpath1)
    tree1 = Tree(imgpath1, topLeft, bottomRight)

    # clip the first image
    x1, y1 = topLeft
    x2, y2 = bottomRight
    img1 = cv2.imread(imgpath1,0)
    # img1 = img1[y1:y2,x1:x2]


    # find trees in second img
    topLeft, bottomRight = final.find_trees(imgpath2)
    tree2 = Tree(imgpath2, topLeft, bottomRight)

    # clip the second image
    x1, y1 = topLeft
    x2, y2 = bottomRight
    img2 = cv2.imread(imgpath2,0) # open image
    # img2 = img2[y1:y2,x1:x2] # clip image



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

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # pts1 = np.array(pts1)
    # pts2 = np.array(pts2)
    p1 = []
    p2 = []
    for i in range(len(pts1)):
        p1.append([float(pts1[i][0]),float(pts1[i][1])])
        p2.append([float(pts2[i][0]),float(pts2[i][1])])
    p1 = np.array(p1,dtype=np.float32)
    p2 = np.array(p2,dtype=np.float32)
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
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    print lines2
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
