'''
    --> Use ORB detector to match the detected sign to a sign from the database
'''
import cv2 as cv
import numpy as np
import os

###################################################################################################

des = []
kpp = []

###################################################################################################

bs = ['TL', 'TR', 'TLR', 'GF', 'GL', 'GR', 'GLR', 'Bus', 'Ped', 'Park']

rs = ['SP20', 'SP30', 'SP40', 'SP50', 'SP60', 'SP70', 'SP80', 'SP90', 'SP100', 'SP110','SP120', 'NL',
      'NR', 'NU', 'NPas', 'T', 'Tr', 'Hei', 'S', 'NE', 'NP', 'NSNP', 'DPed', 'DR','DL','D', 'BR',
      'DlLR', 'DlL', 'DlR']

###################################################################################################


def ReadData(names):

    im = np.empty(len(names), dtype=object)
    nd = os.getcwd() + "/signs/"
    for i in range(0, len(names)):
        im[i] = cv.imread(nd + names[i] + ".jpg")

    return im

###################################################################################################


def Load():

    global des, kpp

    kpp = []
    des = []
    
    # Load blue signs
    im = ReadData(bs)
    k, d = CreateKeys(im, len(im))
    kpp.append(k)
    des.append(d)

    # Load red signs
    im = ReadData(rs)
    k, d = CreateKeys(im, len(im))
    kpp.append(k)
    des.append(d)


    return des, kpp

###################################################################################################

def CreateKeys(im, ln):

    # Initiate ORB detector
    orb = cv.ORB_create(edgeThreshold=18, patchSize=19, nlevels=6, fastThreshold=10,
                        scaleFactor=1.8, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, nfeatures=100)
    kp = np.empty(ln, dtype=object)
    des = np.empty(ln, dtype=object)

    if ln > 1:
        for i in range(0, ln):
            kp[i], des[i] = orb.detectAndCompute(im[i], None)
    else:
        kp, des = orb.detectAndCompute(im, None)

    return kp, des

#--------------------------------------------------------------------------------------------------


def Comparekeys(d1, k1, d2, k2):

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    detectedSign = -1

    if d1 is not None:

        # Match descriptors.
        matches1 = np.empty(len(d1), dtype=object)
        matches2 = np.empty(len(d1), dtype=object)
        s_m = []
        dist = []
        for i in range(0, len(d2)):

            matches1 = bf.match(d1, d2[i])
            matches2 = bf.match(d2[i], d1)

            tmp_s_m = []
            k2i = k2[i]

            # keep only symmetric matches
            for m1 in matches1:
                for m2 in matches2:
                    if k1[m1.queryIdx] == k1[m2.trainIdx] and k2i[m1.trainIdx] == k2i[m2.queryIdx]:
                        tmp_s_m.append(m1)

            s_m.append(tmp_s_m)

            # Clear matches for which NN ratio is > than threshold
            tmp_d = [m.distance for m in s_m[i]]
            avg = (sum(tmp_d) / len(tmp_d))

            # store the distances
            dist.append(avg)

        avg = sum(dist) / len(dist)
        for i in range(0, len(dist)):
            rat = dist[i] / avg

        dmin = np.argmin(dist)
        rat = len(d1) / len(d2[dmin])

        # find the index to the smaller distance
        if min(dist) < 55 and rat > 0.4:
            detectedSign = dmin

    return detectedSign + 1

###################################################################################################

if (__name__ == '__main__'):    
    des, kpp = Load()

    nam = [0, 4, 5, 20, 31, 37, 55]

    im = []

    for i in range(0, len(nam)):
        tmp = cv.imread(str(nam[i]) + ".jpg")
        im.append(tmp)

    tk, td = CreateKeys(im, len(im))

    for i in range(0, len(im)):
        ds = Comparekeys(td[i], tk[i], des[1], kpp[1])
        if ds > -1:
            print(i, rs[ds])
        print("_________________________")
