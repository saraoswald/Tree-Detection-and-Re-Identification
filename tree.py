"""
Tree class for storing data about found trees
Parameters:
    image - path to image the tree was found in
    topLeft - [x,y] array of integer points for top left corner of tree
    bottomRight - [x,y] array of integer points for bottom right corner of tree
"""

class Tree:
    def __init__(self, image, topLeft, bottomRight):
        self.image = image
        self.topLeft = topLeft
        self.bottomRight = bottomRight
