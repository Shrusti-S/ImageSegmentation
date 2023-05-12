
#*********************************************************************************************************#

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import os
import cv2
import numpy as np
from time import time

from numpy import *
import numpy
import maxflow
from PIL import Image
from matplotlib import pyplot as plt
from pylab import *
import cv2

#*********************************************************************************************************#

#The function implements graph cut by partitioning a directed graph into two disjoint sets, foreground and background...
def graph(file,k, s, fore, back): 
    I = (Image.open(file).convert('L'))                                                                                 # read image
    If = I.crop(fore)                                                                                                   # take a part of the foreground
    Ib = I.crop(back)                                                                                                   # take a part of the background
    I,If,Ib = array(I),array(If),array(Ib)                                                                              # convert all the images to arrays to calculation
    Ifmean,Ibmean = mean(cv2.calcHist([If],[0],None,[256],[0,256])),mean(cv2.calcHist([Ib],[0],None,[256],[0,256]))     #Taking the mean of the histogram
    F,B =  ones(shape = I.shape),ones(shape = I.shape)                                                                  #initalizing the foreground/background probability vector
    Im = I.reshape(-1,1)                                                                                                #Coverting the image array to a vector for ease.
    m,n = I.shape[0],I.shape[1]                                                                                         # copy the size
    g,pic = maxflow.Graph[int](m,n),maxflow.Graph[int]()                                                                # define the graph
    structure = np.array([[inf, 0, 0],
                          [inf, 0, 0],
                          [inf, 0, 0]
                         ])                                                                                             # initializing the structure....
    source,sink,J = m*n,m*n+1,I                                                                                         # Defining the Source and Sink (terminal)nodes.
    nodes,nodeids = g.add_nodes(m*n),pic.add_grid_nodes(J.shape)                                                        # Adding non-nodes
    pic.add_grid_edges(nodeids,0),pic.add_grid_tedges(nodeids, J, 255-J)
    gr = pic.maxflow()
    IOut = pic.get_grid_segments(nodeids)
    for i in range(I.shape[0]):                                                                                         # Defining the Probability function....
        for j in range(I.shape[1]):
            F[i,j] = -log(abs(I[i,j] - Ifmean)/(abs(I[i,j] - Ifmean)+abs(I[i,j] - Ibmean)))                             # Probability of a pixel being foreground
            B[i,j] = -log(abs(I[i,j] - Ibmean)/(abs(I[i,j] - Ibmean)+abs(I[i,j] - Ifmean)))                             # Probability of a pixel being background
    F,B = F.reshape(-1,1),B.reshape(-1,1)                                                                               # convertingb  to column vector for ease
    for i in range(Im.shape[0]):
        Im[i] = Im[i] / linalg.norm(Im[i])                                                                              # normalizing the input image vector 
    w = structure                                                                                                       # defining the weight       
    for i in range(m*n):                                                                                                #checking the 4-neighborhood pixels
        ws=(F[i]/(F[i]+B[i]))                                                                                           # source weight
        wt=(B[i]/(F[i]+B[i]))                                                                                           # sink weight
        g.add_tedge(i,ws[0],wt)                                                                                         # edges between pixels and terminal
        if i%n != 0:                                                                                                    # for left pixels
            w = k*exp(-(abs(Im[i]-Im[i-1])**2)/s)                                                                       # the cost function for two pixels
            g.add_edge(i,i-1,w[0],k-w[0])                                                                               # edges between two pixels
            '''Explaination of the likelihood function: * used Bayes’ theorem for conditional probabilities
            * The function is constructed by multiplying the individual conditional probabilities of a pixel being either 
            foreground or background in order to get the total probability. Then the class with highest probability is selected.
            * for a pixel i in the image:
                               * weight from sink to i:
                               probabilty of i being background/sum of probabilities
                               * weight from source to i:
                               probabilty of i being foreground/sum of probabilities
                               * weight from i to a 4-neighbourhood pixel:
                                K * e−|Ii−Ij |2 / s
                                 where k and s are parameters that determine hwo close the neighboring pixels are how fast the values
                                 decay towards zero with increasing dissimilarity
            '''
        if (i+1)%n != 0:                                                                                                # for right pixels
            w = k*exp(-(abs(Im[i]-Im[i+1])**2)/s)
            g.add_edge(i,i+1,w[0],k-w[0])                                                                               # edges between two pixels
        if i//n != 0:                                                                                                   # for top pixels
            w = k*exp(-(abs(Im[i]-Im[i-n])**2)/s)
            g.add_edge(i,i-n,w[0],k-w[0])                                                                               # edges between two pixels
        if i//n != m-1:                                                                                                 # for bottom pixels
            w = k*exp(-(abs(Im[i]-Im[i+n])**2)/s)
            g.add_edge(i,i+n,w[0],k-w[0])                                                                               # edges between two pixels
    I = array(Image.open(file))                                                                                         # calling the input image again to ensure proper pixel intensities....
    print ("The maximum flow for %s is %d"%(file,gr))                                                                   # find and print the maxflow
    Iout = ones(shape = nodes.shape)
    for i in range(len(nodes)):
        Iout[i] = g.get_segment(nodes[i])                                                                               # calssifying each pixel as either forground or background
    out = 255*ones((I.shape[0],I.shape[1],3))                                                                           # initialization for 3d input
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):                                                                                     # converting the True/False to Pixel intensity
            if IOut[i,j]==False:
                if len(I.shape) == 2:
                    out[i,j,0],out[i,j,1],out[i,j,2] = I[i,j],I[i,j],I[i,j]                                             # foreground for 2d image
                if len(I.shape) == 3:
                    out[i,j,0],out[i,j,1],out[i,j,2] = I[i,j,0],I[i,j,1],I[i,j,2]                                       # foreground for 3d image
            else:
                out[i,j,0],out[i,j,1],out[i,j,2] = 1,255,255                                                            # red background 
    figure()
    

#*********************************************************************************************************#

def change_color_fuzzycmeans(cluster_membership, clusters):
    img = []
    for pix in cluster_membership.T:
        img.append(clusters[np.argmax(pix)])
    return img

#*********************************************************************************************************#

def readimage():
    global Img_name
    folder = 'image/'
    list_images = os.listdir(folder)
    list_img = []
    for i in list_images:
        path = folder+i
        print(path)
        verify = os.path.split(path)
        Img_name = verify[1]
        
        img = cv2.imread(path)
       
        rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
        list_img.append(rgb_img)
        
    return list_img

#*********************************************************************************************************#

def bwarea(img):
    row = img.shape[0]
    col = img.shape[1]
    total = 0.0
    for r in range(row-1):
        for c in range(col-1):
            sub_total = img[r:r+2, c:c+2].mean()
            if sub_total == 255:
                total += 1
            elif sub_total == (255.0/3.0):
                total += (7.0/8.0)
            elif sub_total == (255.0/4.0):
                total += 0.25
            elif sub_total == 0:
                total += 0
            else:
                r1c1 = img[r,c]
                r1c2 = img[r,c+1]
                r2c1 = img[r+1,c]
                r2c2 = img[r+1,c+1]
                
                if (((r1c1 == r2c2) & (r1c2 == r2c1)) & (r1c1 != r2c1)):
                    total += 0.75
                else:
                    total += 0.5
    return total
#*********************************************************************************************************#
          
def imclearborder(imgBW):

    # Given a black and white image, first find all of its contours
    radius = 2
    imgBWcopy = imgBW.copy()

    contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

#*********************************************************************************************************#

#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()

    contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy      

#*********************************************************************************************************#

def imfill(im_th):
    
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    
    return im_out

#*********************************************************************************************************#

list_img = readimage()

n_data = len(list_img)

clusters = [2,3,6]

graph(Img_name,2,100,(225,142,279,185),(7,120,61,163))
# looping every images
for index,rgb_img in enumerate(list_img):
    img = np.reshape(rgb_img, (200,200,3)).astype(np.uint8)
    shape = np.shape(img)
    
    
    # initialize graph
    plt.figure(figsize=(20,20))
    plt.subplot(1,4,1)
    plt.imshow(img)
    # looping every cluster     
    print('Image '+str(index+1))
    for i,cluster in enumerate(clusters):
            
        # Fuzzy C Means
        #new_time = time()
        
        # error = 0.005
        # maximum iteration = 1000
        # cluster = 2,3,6,8
        
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None,seed=42)

        new_img = change_color_fuzzycmeans(u,cntr)
        
        fuzzy_img = np.reshape(new_img,shape).astype(np.uint8)
        
        ret, seg_img = cv2.threshold(fuzzy_img,np.max(fuzzy_img)-1,255,cv2.THRESH_BINARY)
        
        print('Fuzzy time for cluster',cluster)
        #print(time() - new_time,'seconds')
        seg_img_1d = seg_img[:,:,1]
        
        
        bwfim1 = bwareaopen(seg_img_1d, 100)
        bwfim2 = imclearborder(bwfim1)
        bwfim3 = imfill(bwfim2)
        
        print('Bwarea : '+str(bwarea(bwfim3)))
        print()

        plt.subplot(1,4,i+2)
        plt.imshow(bwfim3)
        name = 'Cluster'+str(cluster)
        plt.title(name)

    name = 'segmented'+str(index)+'.png'
    plt.savefig(name)
    img = Image.open('segmented0.png')
    img.show()
    
    
#*********************************************************************************************************#


