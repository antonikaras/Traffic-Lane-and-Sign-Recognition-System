'''
    --> Traffic Lane and Sign Recognition app
    --> Uses TrafficLaneRecognition and SignRecognition
'''
import cv2 as cv
import os 
import numpy as np
import SURFdetector as surf
import SVMdetector as svm
import SignRecognitionHough as tsr
import TrafficLaneRecognition as tlr
import time

###################################################################################################
                #   0 : SVM
detector = 0    #   1 : ORB
                #   2 : SURF
###################################################################################################

recVideo = True

###################################################################################################

# Choose the video or camera 
cap = cv.VideoCapture('01200003.AVI')

if recVideo:
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('proc.avi', fourcc, 30.0, (1280, 720))

# --> Load the parameters used for sign recognition <-- #

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
ts_t = 0
tl_t = 0
cnt = 0

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

Dsold = []

# Read until video is completed
while(cap.isOpened()):

    ret, frame = cap.read()
    src = frame
    if ret == True:

        if cnt % 1 == 0:
            # Capture frame-by-frame
            ts = time.time()

            # Traffic Sign Recognition System

            ## Image preprocessing
            Bim, Rim = tsr.Preprocessing(src)

            ### Detect Blue Signs
            Ds = tsr.Processing(src, Bim, kpp[0], des[0], 0)
            
            ### Detect Red Signs
            tmp = tsr.Processing(src, Rim, kpp[1], des[1], 1)
            Ds.extend(tmp)
           
            
            t1 = time.time() - ts

            # Traffic Lane Recognition
            llx, lly, mlx, mly, rlx, rly = tlr.Processing(src)

            t2 = time.time() - ts



            # Draw the results
            src, Dsold = tsr.DrawSigns(src, Ds, Dsold)
            src = tlr.DrawLanes(src, llx, lly, mlx, mly, rlx, rly)

            # Display image
            cv.imshow('frame', src)

            if recVideo:
                out.write(src)

            # compute time duration
            ts_t = ts_t + t1
            tl_t = tl_t + t2

            out.write(src)


            # Press Q on keyboard to  exit
            ch = cv.waitKey(10)
            if ch & 0xFF == ord('q'):
                print("frame processing time = ", ts_t / cnt, tl_t / cnt)
                break
            elif ch == ord(' '):
                cv.imwrite("src" + str(cnt) + ".jpg", src)
                print("src" + str(cnt) + ".jpg")
        # Break the loop
    else:
        break
    cnt = cnt + 1

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()