import cv2 as cv
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib
from enum import Enum

global r1, r2, r3_1, r4_1, r4_2, r4_3, r4_4, r4_5
global b1, b2, b3_1, b3_2, b3_3

# SVMcat = ['env', 'turn', 'go', 'b_var', 'danger', 'r_var', 'stop', 'nEnter', 'nPark', 'nl', 'nr', 'nu', 'a100', 'b100']
# SVMlist = enum(env = 0, turn = 1)


###################################################################################################

sgnN = ['env', 'tl', 'tr', 'tlr', 'stra', 'gl', 'gr', 'glr', 'bus', 'ped', 'park', 'SP20', 'SP30', 'SP40',
        'SP50', 'SP60', 'SP70', 'SP80', 'SP90', 'SP100', 'SP110','SP120', 'NL', 'NR', 'NU', 'Npas', 
        'To', 'Tr', 'hei', 'Stop', 'NE', 'NS', 'NSNP', 'DPed', 'DR','DL', 'D', 'BR', 'DlLR', 'DlL',
        'DlR'
       ]

###################################################################################################


def ExtractHoG(img, lvl):

    if lvl == 1:
        winSize = (64, 64)
    elif lvl == 2:
        winSize = (48, 48)
    elif lvl == 3:
        winSize = (32, 32)

    rim = cv.resize(img, winSize)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (4, 4)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (16, 16)
    padding = (0, 0)
    locations = ()

    hist = hog.compute(rim, winStride, padding, locations)

    testData = np.float32(hist).reshape(1, -1)

    return testData

###################################################################################################


def SVMClassifier(im, col):

    global r1, r2, r3_1, r4_1, r4_2, r4_3, r4_4, r4_5
    global b1, b2, b3_1, b3_2, b3_3

    testData = ExtractHoG(im, 1)

    # print(testData.shape[:2])
    if col == 0:
        result = b1.predict(testData)                       # env - signs
        #print("b1 = ", result)
        if result > 0:                                      
            testData = ExtractHoG(im, 2)
            result = b2.predict(testData)                   # turn - go - varius
            #print("b2 = ", result)
            if result == 0:               
                testData = ExtractHoG(im, 3)
                result = b3_1.predict(testData)             # turn left - turn right - turn left,right - straight
                #print("b3_1 = ", result)
            elif result == 1:
                testData = ExtractHoG(im, 3)
                result = b3_2.predict(testData) + 4         # go left - go right - go left,right
                #print("b3_2 = ", result - 4)
            else :
                testData = ExtractHoG(im, 3)
                result = b3_3.predict(testData) + 7         # bus - pedestrian - parking
                #print("b3_3 = ", result - 7)         
        else:
            result = -1              
    else:
        result = r1.predict(testData)                       # env - signs
        #print("r1 = ", result)
        if result > 0:                                      
            testData = ExtractHoG(im, 2)
            result = r2.predict(testData)                   # forbid - no park,stop,noenter - triangles
            #print("r2 = ", result)
            if result == 0:                                 
                result = r3_1.predict(testData)             # splim - noturn - varius
                #print("r3_1 = ", result)
                if result == 0:                             # speed limits
                    testData = ExtractHoG(im, 3)
                    result = r4_1.predict(testData)
                    #print("r4_1 = ", result)                              
                elif result == 1:                           # no - turn
                    testData = ExtractHoG(im, 3)
                    result = r4_2.predict(testData) + 11
                    #print("r4_2 = ", result - 11)  
                elif result == 2:                           # varius forbid signs
                    testData = ExtractHoG(im, 3)
                    result = r4_3.predict(testData) + 14
                    #print("r4_3 = ", result - 14)
            elif result == 1:                               # stop - no enter - no stop - no stop no parking
                testData = ExtractHoG(im, 3) 
                result = r4_4.predict(testData) + 18
                #print("r4_4 = ", result - 18)
            elif result == 2:                               # danger signs
                testData = ExtractHoG(im, 3)
                result = r4_5.predict(testData) + 22
                #print("r4_5 = ", result - 22) 
        else:
            result = -11

    if col == 1:
        result = result + 10

    #print(int(result), sgnN[int(result) + 1])

    return result + 1


###################################################################################################

def Load():

    global r1, r2, r3_1, r4_1, r4_2, r4_3, r4_4, r4_5
    global b1, b2, b3_1, b3_2, b3_3
    
    nm = os.getcwd() + "/svm"

    r1 = joblib.load(nm + "/r1.joblib.pkl")
    r2 = joblib.load(nm + "/r2.joblib.pkl")
    r3_1 = joblib.load(nm + "/r3_1.joblib.pkl")
    r4_1 = joblib.load(nm + "/r4_1.joblib.pkl")
    r4_2 = joblib.load(nm + "/r4_2.joblib.pkl")
    r4_3 = joblib.load(nm + "/r4_3.joblib.pkl")
    r4_4 = joblib.load(nm + "/r4_4.joblib.pkl")
    r4_5 = joblib.load(nm + "/r4_5.joblib.pkl")
    b1 = joblib.load(nm + "/b1.joblib.pkl")
    b2 = joblib.load(nm + "/b2.joblib.pkl")
    b3_1 = joblib.load(nm + "/b3_1.joblib.pkl")
    b3_2 = joblib.load(nm + "/b3_2.joblib.pkl")
    b3_3 = joblib.load(nm + "/b3_3.joblib.pkl")



###################################################################################################

# nm = os.getcwd() + "/96.joblib.pkl"
# clf = joblib.load(nm)
'''

Load()

im = cv.imread('41.jpg')
gim = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

SVMClassifier(gim, 1)
'''