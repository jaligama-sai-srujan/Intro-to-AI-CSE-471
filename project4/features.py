# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples
import matplotlib.pylab as plt # Yantian
from threading import Thread # Yantian
import time

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()


def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.
colorbar()
    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
    I have added 9 new features
    first I have taken the bounds of the Image (i.e., the area in which image is present)
    then i used this area for all the features
    first 4 features are the number of black pixels in each and every quadrant(I have divided the image area into 4 quadrants).
    I used this because different numbers have different values in the quandrant.
    next four features are the one where I have given the value to quadrant for which there are no pixels.
    last feature gives the count of continues black areas.
    ##
    """
    features = basicFeatureExtractor(datum)
    count = np.count_nonzero(features==1)
    count_horizontal = 0
    
    """
    f = features.reshape(28,28)
    count_h1 = 0
    count_h2 = 0
    count_h3 = 0
    for i in range(21):
        for j in range(28):
            if f[i][j] == 1:
                if i<7:
                    count_h1+=1
                elif i<14 and i>=7:
                    count_h2+=1
                elif i<21 and i>=14:
                    count_h3+=1
    count_h4 = count - count_h1 -count_h2 -count_h3
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for i in range(28):
        for j in range(21):
            if f[i][j] == 1:
                if j<7:
                    count_1+=1
                elif j<14 and j>=7:
                    count_2+=1
                elif j<21 and j>=14:
                    count_3+=1
    count_4 = count - count_1 -count_2 -count_3
    
    
    
    count_1/=count
    count_2/=count
    count_3/=count
    count_h1/=count
    count_h2/=count
    count_h3/=count
    count_1 = np.array([count_1])
    count_2 = np.array([count_2])
    count_3 = np.array([count_3])
    count_h1 = np.array([count_h1])
    count_h2 = np.array([count_h2])
    count_h3 = np.array([count_h3])
    features = np.concatenate((features,count_h1),axis=0)
    features = np.concatenate((features,count_h2),axis=0)
    features = np.concatenate((features,count_h3),axis=0)
    features = np.concatenate((features,count_1),axis=0)
    features = np.concatenate((features,count_2),axis=0)
    """
    f = features.reshape(28,28)
    #print(f)
    
    row_1 = 0
    row_n = 0
    column_1 = 0
    column_n = 0
    change = 0
    for i in range(28):
        for j in range(28):
            if f[i][j]==1:
                if change == 0:
                    row_1 = i
                    column_1 = j
                    change = 1
                else:
                    row_n = i
                    if j > column_n:
                        column_n = j
                    if j < column_1:
                        column = j

    count_1 = 0
    count_2 = 0
    count_3 = 0
    for i in range(row_1,row_n+1):
        for j in range(column_1,column_n+1):
            if f[i][j]==1:
                #print(i,j,row_n/2,column_n/2)
                if i<row_n/2 and j<column_n/2:
                    count_1+=1
                elif i<row_n/2 and j>=column_n/2:
                    count_2+=1
                elif i>=row_n/2 and j<column_n/2:
                    count_3+=1
    count_4 = count - count_1 - count_2 - count_3
    count_1 = float(count_1)/count
    count_2 = float(count_2)/count
    count_3 = float(count_3)/count
    count_4 = float(count_4)/count
    count_5=0
    count_6=0
    
    if count_1==0 or count_3==0:
        count_5=1
    if count_2==0 or count_4==0:
        count_6=1
    count_7 =0
    if float(column_n-column_1)/(row_n-row_1)>3:
        count_7 =1 
    
    visited=[[0 for i in range(28)]for j in range(28)]
    def adjacent(x, y):
        N=[]
        if x > 0:
            N.append( (x-1,y) )
        
        if x < 27:
            N.append( (x+1,y) )
        
        if y > 0:
            N.append( (x,y-1) )

        if y < 27:
            N.append( (x,y+1) )
            
        if x > 0 and y > 0:
            N.append( (x-1,y-1) )
        
        if x < 27 and y < 27:
            N.append( (x+1,y+1) )
        
        if x < 27 and y > 0:
            N.append( (x+1,y-1) )

        if x > 0 and y < 27:
            N.append( (x-1,y+1) )
        return N
    def Area_count(N,visited):
        for x,y in N:
            if visited[x][y]==0 and f[x][y]==0:
                N = adjacent(x,y)
                visited[x][y] = 1
                Area_count(N,visited)
    count_8 = 0            
    for i in range(row_1,row_n+1):
        for j in range(column_1,column_n+1):
            if f[i][j] == 0 and visited[i][j]== 0:
                visited[i][j] = 1
                N = adjacent(i,j)
                Area_count(N,visited)
                count_8+=1

    
    #print(count_8)
    
    c = np.array([count_1,count_2,count_3,count_4,count_5,count_6,count_7,count_8])     
    return np.concatenate((features,c),axis=0)
    


def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit
    
    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
