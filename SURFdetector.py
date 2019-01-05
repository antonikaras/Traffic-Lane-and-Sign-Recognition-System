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
        print(i, names[i], im[i] is None)

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
    surf = cv.xfeatures2d.SURF_create(hessianThreshold=500,
                                      nOctaves=8, nOctaveLayers=8, extended=True, upright=True)

    kp = np.empty(ln, dtype=object)
    des = np.empty(ln, dtype=object)

    if ln > 1:
        for i in range(0, ln):
            kp[i], des[i] = surf.detectAndCompute(im[i], None)
    else:
        kp, des = surf.detectAndCompute(im, None)

    return kp, des


###################################################################################################

def Comparekeys(d1, k1, d2, k2):

    detectedSign = -1

    if d1 is not None:
        s_m = []
        bf = cv.BFMatcher()
        for i in range(0, len(d2)):
            if d2[i] is not None:
                ms1 = bf.knnMatch(d1, d2[i], k=2)
                ms2 = bf.knnMatch(d2[i], d1, k=2)
                dist1 = [m[0].distance for m in ms1]
                dist2 = [m[0].distance for m in ms2]
                avg1 = sum(dist1) / len(dist1)
                avg2 = sum(dist2) / len(dist2)
                s_m.append(0.5 * (avg1 + avg2))

        avg = sum(s_m) / len(s_m)
        rat = []
        for i in range(0, len(s_m)):
            rat.append(s_m[i] / avg)
            
        mni = np.argmin(s_m)
        if s_m[mni] < 0.8 and rat[mni] < 0.95:
            detectedSign = mni + 1

    return detectedSign

###################################################################################################

#des, kpp = Load()
