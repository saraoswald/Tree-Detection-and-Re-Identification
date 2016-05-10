import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import spline
from tree import *

def xCoordinates(sobel_img):
    num_rows = float(len(sobel_img)) # get number of x values

    # sum along y axis
    vert_sum = np.sum(sobel_img,axis=0)
    # make it an average value (divide by # of x values)
    vert_sum = np.divide(vert_sum,num_rows)

    x = np.arange(0,len(vert_sum)) # for graphing
    xnew = np.arange(0,len(vert_sum),50) # for smoothing
    #smooth
    y_smooth = spline(x, vert_sum, xnew)

    #make a sin curve 1/3 of the width of image
    img_width, img_height = sobel_img.shape
    z = np.arange(0,int(img_width/3),1)
    def f(x):
        return np.sin(x/90)*-15 + 25

    f = [f(i) for i in z] # make sine into an array

    # convolve sine and the vertical sum
    y_conv = np.convolve(vert_sum, f,'same')

    # detect local minima
    mins = (np.diff(np.sign(np.diff(y_conv))) > 0).nonzero()[0] + 1

    return mins

def yCoordinates(sobel_img):
    num_col = float(len(sobel_img[0])) #number of y values
    # sum along x axis
    horiz_sum = np.sum(sobel_img, axis=1)
    #average value
    horiz_sum = np.divide(horiz_sum, num_col)

    y = np.arange(0, len(horiz_sum))
    ynew = np.arange(0, len(horiz_sum))
    x_smooth = spline(y, horiz_sum, ynew)

    #make a sin curve 1/3 of the height
    img_width, img_height = sobel_img.shape
    z = np.arange(0,int(img_height/3),1)
    def f(x):
        return np.sin(x/90)*-15 + 25

    f = [f(i) for i in z] # make sine into an array

    # convolve sine and the vertical sum
    y_conv = np.convolve(horiz_sum, f,'same')

    # detect local minima
    mins = (np.diff(np.sign(np.diff(y_conv))) > 0).nonzero()[0] + 1

    return mins


"""
Sobel reference
http://www.jayrambhia.com/blog/sobel-operator/
"""

def find_trees(imgpath):
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

    #Take the coordinates of the x and y values from the min values
    xMin = xCoordinates(sobel_img)
    yMin = yCoordinates(sobel_img)

    #Take the middle 20 values of the coordinates for the tree
    xIndex = int(len(xMin)/2) - 11
    yIndex = int(len(yMin)/2) - 11

    xMaxIndex = xIndex + 20
    yMaxIndex = yIndex + 20

    #Put the (x,y) coordinates for the tree into an array
    treeLocator = []
    t = 0

    while(xIndex <= xMaxIndex):
        treeLocator.append([])
        treeLocator[t].append(xMin[xIndex])
        treeLocator[t].append(yMin[yIndex])
        xIndex += 1
        yIndex += 1
        t += 1
    #Draw the lines for the four corners of the tree using the min and max coordinates from the array
    img = cv2.imread(imgpath)

    # return coordinates of top left and top right points
    return (treeLocator[0][0],treeLocator[0][1]), (treeLocator[20][0],treeLocator[20][1])

def draw_boxes(imgpath, tree):
    x1, y1 = tree.topLeft
    x2, y2 = tree.bottomRight

    img = cv2.imread(imgpath)

    cv2.line(img,(x1,y1),(x2,y1),(255, 0, 0),3)
    cv2.line(img,(x1,y1),(x1,y2),(255, 0, 0),3)
    cv2.line(img,(x1,y2),(x2,y2),(255, 0, 0),3)
    cv2.line(img,(x2,y1),(x2,y2),(255, 0, 0),3)

    return img

if __name__ == '__main__':
    img1, img2 = 'data/tree2.jpg', 'data/tree3.jpg'

    topLeft, bottomRight = find_trees(img1)
    tree1 = Tree(img1, topLeft, bottomRight)

    topLeft, bottomRight = find_trees(img2)
    tree2 = Tree(img2, topLeft, bottomRight)

    img = draw_boxes(img1, tree1)
    img1 = img1.split('/')[1]
    cv2.imwrite('boxes_'+img1,img)
