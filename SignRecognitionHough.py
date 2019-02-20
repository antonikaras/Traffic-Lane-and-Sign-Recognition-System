'''
------- S T E P S -------
---> Detect red Signs
---> Detect blue signs
---> Recognize Signs seperately using ORB or SURF or SVM
---> If a signs exists more than once its deleted
---> if opposite signs are detected they are deleted

'''

import cv2 as cv
import numpy as np
import time
import SURFdetector as surf
import ORBdetector as orb
import SVMdetector as svm
from scipy.spatial import distance

xs = 30
ys = 100
he = 400
wi = 1250

k = 1

###################################################################################################
                #   0 : SVM
detector = 0    #   1 : ORB
                #   2 : SURF
###################################################################################################

sgnN = ['tl', 'tr', 'tlr', 'stra', 'gl', 'gr', 'glr', 'bus', 'ped', 'park', 'SP20', 'SP30', 'SP40',
        'SP50', 'SP60', 'SP70', 'SP80', 'SP90', 'SP100', 'SP110','SP120', 'NL', 'NR', 'NU', 'Npas', 
        'To', 'Tr', 'hei', 'Stop', 'NE', 'NS', 'NSNP', 'DPed', 'DR','DL', 'D', 'BR', 'DlLR', 'DlL',
        'DlR'
       ]

#####################################################################


def Preprocessing(im):

    # Crop the image in the area the signes are expected
    cim = im[ys:ys + he, xs: xs + wi]
    # cv.imwrite("cim.jpg", cim)

    # apply histogram equalization to each rgb component
    rgb = cv.split(cim)
    rgb[0] = cv.equalizeHist(rgb[0])
    rgb[2] = cv.equalizeHist(rgb[2])
    rbeq = cv.merge(rgb)    # Used for Blue signs

    rgb[1] = cv.equalizeHist(rgb[1])
    rgbeq = cv.merge(rgb)   # Used for Red signs

    hsveq = cv.cvtColor(rgbeq, cv.COLOR_BGR2HSV)
    hq_red, sq, vq = cv.split(hsveq)

    # Convert histogram equalized image to HSV
    hsveq = cv.cvtColor(rbeq, cv.COLOR_BGR2HSV)
    hq_blue, sq, vq = cv.split(hsveq)

    # Convert original image to HSV
    hsv = cv.cvtColor(cim, cv.COLOR_BGR2HSV)

    # Reduce the noise to avoid false circle detection
    hsv = cv.GaussianBlur(hsv, (9, 9), 2, 2)

    # split the image to its components
    h, s, v = cv.split(hsv)
    x, y = h.shape[:2]

    ###########################################################################
    # Detect RED SIGNS

    thr, thr_s = cv.threshold(s, 70, 255, cv.THRESH_TOZERO)
    thr, thr_20 = cv.threshold(hq_red, 10, 255, cv.THRESH_BINARY_INV)
    thr, thr_160 = cv.threshold(hq_red, 160, 255, cv.THRESH_TOZERO)
    thr, thr_190 = cv.threshold(hq_red, 190, 255, cv.THRESH_TOZERO_INV)
    h_red = cv.bitwise_and(thr_160, thr_190)
    h_red = cv.bitwise_or(thr_20, h_red)
    red_sgns = cv.bitwise_and(thr_s, h_red)

    # resize the image
    rds = cv.resize(red_sgns, (0, 0), fx=(1 / k), fy=(1 / k))
    ###########################################################################
    # Detect BLUE SIGNS
    thr, thr_100 = cv.threshold(hq_blue, 100, 255, cv.THRESH_TOZERO)
    thr, thr_130 = cv.threshold(hq_blue, 130, 255, cv.THRESH_TOZERO_INV)
    h_blue = cv.bitwise_and(thr_100, thr_130)
    blue_sgns = cv.bitwise_and(thr_s, h_blue)

    # resize the image
    bds = cv.resize(blue_sgns, (0, 0), fx=(1 / k), fy=(1 / k))

    return bds, rds

#--------------------------------------------------------------------------------------------------

# Detect circles using Hough transform


def Processing(im, im_s, kpt, dest, col):

    circles = cv.HoughCircles(
        im_s, cv.HOUGH_GRADIENT, 1, 60, param1=30, param2=18, minRadius=int(15 / k), maxRadius=int(35 / k))

    Ds = []
    ym, xm = im.shape[:2]
    if circles is not None:

        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:

            c = (k * i[0] + xs, k * i[1] + ys)
            r = k * i[2]

            if k * i[0] + xs - k * i[2] < 0:
                x1 = 0
            else:
                x1 = k * i[0] + xs - k * i[2]

            if k * i[1] + ys - k * i[2] < 0:
                y1 = 0
            else:
                y1 = k * i[1] + ys - k * i[2]

            if k * i[0] + xs + k * i[2] > xm:
                x2 = xm
            else:
                x2 = k * i[0] + xs + k * i[2]

            if k * i[1] + ys + k * i[2] > ym:
                y2 = ym
            else:
                y2 = k * i[1] + ys + k * i[2]

            imp = im[y1:y2, x1:x2]

            ds = -1
            if len(imp) > 0:
                if detector == 0:
                    ds = svm.SVMClassifier(imp, col)
                elif detector == 1:
                    kpp, desp = orb.CreateKeys(imp, 1)
                    ds = orb.Comparekeys(desp, kpp, dest, kpt)
                    if col == 1 and ds > 0:
                        ds = ds + 10
                elif detector == 2:
                    kpp, desp = surf.CreateKeys(imp, 1)
                    ds = surf.Comparekeys(desp, kpp, dest, kpt)
                    if col == 1 and ds > 0:
                        ds = ds + 10 
            
            Ds.append([c, r, 10, 0, ds])

    return Ds

###################################################################################################

def DrawSigns(src, Ds, Dsold): 
    
    i = 0
    while i < len(Ds):
        #print("first pass", Ds[i][0], Ds[i][4])
        if int(Ds[i][4]) < 1:
            #print("del", Ds[i][0])
            del Ds[i]
            i = i - 1
            
        i = i + 1

    if len(Dsold) > 0:
        
        for i in range(0, len(Ds)):
            dist = [distance.euclidean(Ds[i][0], c[0]) for c in Dsold] 

            if Ds[i][4] > 0:

                for j in range(0, len(dist)):
                    if dist[j] < 100 and Ds[i][4] == Dsold[j][4]:
                        Dsold[j][0] = Ds[i][0]
                        Dsold[j][1] = Ds[i][1]
                        Dsold[j][2] = Dsold[j][2] + 1
                        Dsold[j][3] = Dsold[j][3] + 1
                    else:
                        Dsold.append(Ds[i])

    else:
        Dsold = Ds

    i = 0
    while i < len(Dsold):
        Dsold[i][2] = Dsold[i][2] - 1
        if Dsold[i][2] < 0:
            del Dsold[i]

        i = i + 1

    for ds in Dsold:
        
        if ds[3] > 1:
            txt = sgnN[int(ds[4]) - 1]
            cv.circle(src, ds[0], ds[1], (0, 255, 0), 2)
            cv.putText(src, txt, ds[0], cv.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 255), 8, cv.LINE_AA)

    for ds in Ds:

        #
        if int(ds[4]) > 0:
            txt = sgnN[int(ds[4]) - 1]
            cv.putText(src, txt, ds[0], cv.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 255), 8, cv.LINE_AA)
            cv.circle(src, ds[0], ds[1], (0, 0, 255), 2)
        #else:
        #    print("second pass", ds[0], ds[4])

    return src, Dsold


####################################################################################################

if (__name__ == 'main'):

    # Load the video
    cap = cv.VideoCapture('10000022.AVI')

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))


    # --> Load the parameters used for sign recognition <-- ############||||||||

    des = []
    kpp = []

    if detector == 0:
        svm.Load()
        des.append([])
        kpp.append([])
        des.append([])
        kpp.append([])
    elif detector == 1:
        des, kpp = orb.Load()
    elif detector == 2:
        des, kpp = surf.Load()

    n = 0
    t1_t = 0
    t2_t = 0
    cnt = 0

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    Dsold = []

    # Read until video is completed
    while(cap.isOpened()):

        ret, frame = cap.read()
        src = frame
        srcclone = frame.copy()
        if ret == True:

            print(cnt)

            # Capture frame-by-frame
            ts = time.time()

            # Image preprocessing
            if cnt > 0:
                Bim, Rim = Preprocessing(src)
                t1 = time.time() - ts

                Ds = Processing(src, Bim, kpp[0], des[0], 0)
                tmp = Processing(src, Rim, kpp[1], des[1], 1)
                Ds.extend(tmp)
           

                src, Dsold = DrawSigns(src, Ds, Dsold)

                t2 = time.time() - ts
                # Display image
                cv.namedWindow("frame", cv.WINDOW_NORMAL)
                cv.imshow('frame', src)
            
           
            # compute time duration
            t1_t = t1_t + t1
            t2_t = t2_t + t2
            cnt = cnt + 1

            # Press Q on keyboard to  exit

            # out.write(src)

            ch = cv.waitKey()
            if ch & 0xFF == ord('q'):
                print("frame processing time = ", t1_t / cnt, t2_t / cnt)
                break
            elif ch == ord(' '):
                cv.imwrite("o" + str(cnt) + ".jpg", src)
                print("src" + str(cnt) + ".jpg")
            # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()
