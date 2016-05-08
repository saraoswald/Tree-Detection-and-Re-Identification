import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import spline

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
    #plt.title("Tree candidates for "+imgpath)
    #plt.xlabel("X Coordinate of pixels in image")
    #plt.ylabel("Average grayscale convolved with sine curve")

    # plt.plot(x,vert_sum,'g-',x,y_conv,'r-')
    #plt.plot(x,y_conv,'b-',mins,[y_conv[x] for x in mins],'o')

    # plot lines over image where minima are
    #img = cv2.imread(imgpath)
    #for i in mins:
    #    cv2.line(img,(i,0),(i,int(num_rows)),(255, 0, 0),3)

    #cv2.imwrite('lines.png',img)
    #plt.show()
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
    
    #make a sin curve 300px wide
    z = np.arange(0,300,1)
    def f(x):
        return np.sin(x/90)*-15 + 25

    f = [f(i) for i in z] # make sine into an array

    # convolve sine and the vertical sum
    y_conv = np.convolve(horiz_sum, f,'same')

    # detect local minima
    mins = (np.diff(np.sign(np.diff(y_conv))) > 0).nonzero()[0] + 1
    
    #plt.title("Tree candidates for "+imgpath)
    #plt.xlabel("Y Coordinate of pixels in image")
    #plt.ylabel("Average grayscale convolved with sine curve")

    # plt.plot(x,vert_sum,'g-',x,y_conv,'r-')
    #plt.plot(y,y_conv,'b-',mins,[y_conv[x] for x in mins],'o')

    # plot lines over image where minima are
    #img = cv2.imread(imgpath)
    #for i in mins:
    #    cv2.line(img,(0, i),(int(num_col), i),(255, 0, 0),3)

    #cv2.imwrite('lines.png',img)

    #plt.show()
    return mins

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
cv2.line(img,(treeLocator[0][0],treeLocator[0][1]),(treeLocator[20][0],treeLocator[0][1]),(255, 0, 0),3)
cv2.line(img,(treeLocator[0][0],treeLocator[0][1]),(treeLocator[0][0],treeLocator[20][1]),(255, 0, 0),3)
cv2.line(img,(treeLocator[0][0],treeLocator[20][1]),(treeLocator[20][0],treeLocator[20][1]),(255, 0, 0),3)
cv2.line(img,(treeLocator[20][0],treeLocator[0][1]),(treeLocator[20][0],treeLocator[20][1]),(255, 0, 0),3)

cv2.imwrite('lines.png',img)



# cv2.imwrite('sobel.png',sobel_img)

# laplacian = cv2.Laplacian(sobel_img,cv2.CV_64F)
# cv2.imwrite('laplacian.png',laplacian)

