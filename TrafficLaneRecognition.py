'''
    ---> crop the image
    ---> convert it to gray
    ---> detect the lines in smaller boxes using Hough
        ---> detect the middle lane 3 times to determine the position of the car
    ---> apply lsq to detect the curve
'''

from cv2 import *
import time
import numpy as np
from math import *
from scipy.spatial import distance  

#--------------------------------------------------------------------------------------------------
''''
xs = 100
ys = 460
he = 200
we = 900
'''
xs = 150
ys = 450
he = 180
we = 900

#--------------------------------------------------------------------------------------------------
# Creates a line grid for help

def CreateGrid(src):
    
    line(src, (xs, ys), (xs + 9 * 100, ys), (0, 0, 255), 2)
    line(src, (xs, ys + 90), (xs + 9 * 100, ys + 90), (0, 0, 255), 2)
    line(src, (xs, ys + 2 * 90), (xs + 9 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 0 * 100, ys), (xs + 0 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 1 * 100, ys), (xs + 1 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 2 * 100, ys), (xs + 2 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 3 * 100, ys), (xs + 3 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 4 * 100, ys), (xs + 4 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 5 * 100, ys), (xs + 5 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 6 * 100, ys), (xs + 6 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 7 * 100, ys), (xs + 7 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 8 * 100, ys), (xs + 8 * 100, ys + 2 * 90), (0, 0, 255), 2)
    line(src, (xs + 9 * 100, ys), (xs + 9 * 100, ys + 2 * 90), (0, 0, 255), 2)
    
    putText(src, "1", (xs + 0 * 100, ys + 50), FONT_HERSHEY_SIMPLEX,
            2, (255, 0, 255), 2, LINE_AA)
    putText(src, "2", (xs + 1 * 100, ys + 50), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "3", (xs + 2 * 100, ys + 50), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "4", (xs + 3 * 100, ys + 50), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "5", (xs + 4 * 100, ys + 50), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "6", (xs + 5 * 100, ys + 50), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "7", (xs + 6 * 100, ys + 50), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "8", (xs + 7 * 100, ys + 50), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "9", (xs + 8 * 100, ys + 50), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "10", (xs + 0 * 100, ys + 150), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "11", (xs + 1 * 100, ys + 150), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "12", (xs + 2 * 100, ys + 150), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "13", (xs + 3 * 100, ys + 150), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "14", (xs + 4 * 100, ys + 150), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "15", (xs + 5 * 100, ys + 150), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "16", (xs + 6 * 100, ys + 150), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "17", (xs + 7 * 100, ys + 150), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)
    putText(src, "18", (xs + 8 * 100, ys + 150), FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 2, LINE_AA)

    return src

#--------------------------------------------------------------------------------------------------

# draw the fitted curve in the image

def DrawCurve(img, c):

    ymin = c[3]
    ymax = c[4]
    for j in range(ymin, ymax):
        i = ceil(c[0] * j**2 + c[1] * j + c[2])
        circle(img, (i, j), 3, (255, 0, 0), -1, 8, 0)

#--------------------------------------------------------------------------------------------------

# draw the fitted curve in the image


def FindCurve(x, y, l):

    ymin = -1
    ymax = -1
    coefs = [np.Inf, np.Inf, np.Inf]

    if len(x) > 4:

        ymin = np.min(y)
        ymax = np.max(y)
         # Delete far points
        avg = sum(x) / len(x)
        i = 0
        prevLen = 0
        #print(len(y))

        while prevLen != len(x):
            i = 0
            prevLen = len(x)
            coefs = np.polyfit(y, x, 2)
            while len(x) > 2 and abs(coefs[0]) > 0.01:

                i = np.argmax([abs(avg - i) for i in x])

                    # print("delete element", avg, x[i])
                if i % 2 == 0:
                    del x[i]
                    del y[i]
                    del x[i]
                    del y[i]
                else:
                    del x[i - 1]
                    del y[i - 1]
                    del x[i - 1]
                    del y[i - 1]
                coefs = np.polyfit(y, x, 2)
         
        coefs = np.polyfit(y, x, 2)
        #print(len(y))
        if l == 0 and coefs[0] < 0.001:
            # print(coefs)
            ymin = np.min(y)
            ymax = np.max(y)
        elif abs(coefs[0]) < 0.01:
            ymin = np.min(y)
            ymax = np.max(y)

    coefs = [coefs[0], coefs[1], coefs[2], ymin, ymax]
    
    return coefs

#--------------------------------------------------------------------------------------------------

# Detect and draw the lines


def Processing(src):

    W, H = src.shape[:2]

    gsrc = cvtColor(src, COLOR_BGR2GRAY)
    csrc = gsrc[ys:ys + he, xs:xs + we]
    

    sx = 100
    sy = 90 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    llx = []
    lly = []
    m1lx = []
    m1ly = []
    m2lx = []
    m2ly = []
    m3lx = []
    m3ly = []
    rlx = []
    rly = []

    cnt = 0

    lb = [2, 3]
    mb1 = [4, 5, 12, 13]
    mb2 = [5, 6, 13, 14]
    mb3 = [6, 7, 14, 15]
    rb = [7, 8, 17, 18]

    # Otsu's thresholding after Gaussian filtering
    blur = GaussianBlur(csrc, (5, 5), 0.1)

    thr, otsu = threshold(blur, 0, 255, THRESH_TOZERO + THRESH_OTSU)

    thr, thr_s = threshold(blur, thr, 255, THRESH_TOZERO)




    # Apply canny edge detection
           
    cnny = Canny(otsu, thr, 3 * thr)

    #imshow("thr", canny)
    #waitKey()

    for j in range(0, 2):
        for i in range(0, 9):

            cnt = cnt + 1
            #print(i, j, cnt)
            x = i * sx
            y = j * sy

            li = cnt in lb
            mi1 = cnt in mb1
            mi2 = cnt in mb2
            mi3 = cnt in mb3
            mi = mi1 or mi2 or mi3
            ri = cnt in rb

            if li or mi or ri:

                b_gsrc = cnny[y:y + sy, x:x + sx]

                lines = HoughLinesP(b_gsrc, 1, np.pi / 180, 22, 20, 8)

                if lines is not None:
                    # print(len(lines))
                    for L in range(0, len(lines)):
                        for x1, y1, x2, y2 in lines[L]:
                            p1 = (x1, y1)
                            p2 = (x2, y2)

                            length = distance.euclidean(p1, p2 )
                            #print(x1 + xs + x, y1 + ys + y, length)
                            if x1 != x2 and length > 5:
                                #print("----> ", x1 + xs + x, y1 + ys + y, length)
                                a = (y1 - y2) / (x1 - x2)
                                th = atan(a) * 180 / np.pi

                                if th > -80 and th < -15:
                                    if li:
                                        llx.append(x1 + xs + x)
                                        llx.append(x2 + xs + x)
                                        lly.append(y1 + ys + y)
                                        lly.append(y2 + ys + y)
                                    if mi1:
                                        #print(x1, xs, x, x1 + xs + x)
                                        m1lx.append(x1 + xs + x)
                                        m1lx.append(x2 + xs + x)
                                        m1ly.append(y1 + ys + y)
                                        m1ly.append(y2 + ys + y)
                                    if mi2:
                                        #print(x1, xs, x, x1 + xs + x)
                                        m2lx.append(x1 + xs + x)
                                        m2lx.append(x2 + xs + x)
                                        m2ly.append(y1 + ys + y)
                                        m2ly.append(y2 + ys + y)
                                    if mi3:
                                        #print(x1, xs, x, x1 + xs + x)
                                        m3lx.append(x1 + xs + x)
                                        m3lx.append(x2 + xs + x)
                                        m3ly.append(y1 + ys + y)
                                        m3ly.append(y2 + ys + y)
                                if th > 15 and th < 80 and ri:
                                    rlx.append(x1 + xs + x)
                                    rlx.append(x2 + xs + x)
                                    rly.append(y1 + ys + y)
                                    rly.append(y2 + ys + y)

                            #line(src, (x1 + xs + x, y1 + ys + y), (x2 + xs + x, y2 + ys + y), (0, 0, 255), 2)

    l1 = -1
    l2 = -1
    l3 = -1

    mlx = []
    mly = []

    if m1lx is not None:
        l1 = len(m1lx)
    if m2lx is not None:
        l2 = len(m2lx)
    if m3lx is not None:
        l3 = len(m3lx)

    l = [l1, l2, l3]
    lmaxi = np.argmax(l)
    #print(l)
    if l[lmaxi] > 0:
        if lmaxi == 0 :
            mlx = m1lx
            mly = m1ly
            if l2 / l1 < 0.7:
                putText(src, "caution-steer left", (400, 50), FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, LINE_AA)
            else:
                putText(src, "in-lane", (400, 50), FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, LINE_AA)    
        elif lmaxi == 1 :
            mlx = m2lx
            mly = m2ly
            putText(src, "in-lane", (400, 50), FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, LINE_AA)
        else:
            mlx = m3lx
            mly = m3ly
            if l2 / l3 < 0.7:
                putText(src, "caution-steer right", (400, 50), FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, LINE_AA)
            else:
                putText(src, "in-lane", (400, 50), FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, LINE_AA)

    return llx, lly, mlx, mly, rlx, rly

###################################################################################################

def DrawLanes(src, llx, lly, mlx, mly, rlx, rly):

    c = FindCurve(llx, lly, 0)
    #if c is not None:
    #    print("l", c)
    DrawCurve(src, c)
    c = FindCurve(mlx, mly, 0)
    #if c is not None:
    #    print("m", c)
    DrawCurve(src, c)
    c = FindCurve(rlx, rly, 1)
    #if c is not None:
    #    print("r", c)
    DrawCurve(src, c)

    return src

#--------------------------------------------------------------------------------------------------

'''
#cap = cv2.VideoCapture('19190014.AVI')
#cap = cv2.VideoCapture('18590010.AVI')
#cap = cv2.VideoCapture('18180002.AVI')
#cap = cv2.VideoCapture('Trip_02.mp4')
cap = VideoCapture('01200003.AVI')


n = 0
t_tot = 0
cnt = 0
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    tstart = time.time()

    ret, frame = cap.read()
    src = frame
    if ret == True:

        llx, lly, mlx, mly, rlx, rly = Processing(src)
        src = DrawLanes(src, llx, lly, mlx, mly, rlx, rly)

        src = CreateGrid(src)

        # Display image
        imshow('frame', src)

        #imwrite("srcgrid.jpg", src)

        # imshow('otsu', cnny)
        # compute time duration
        t_dur = time.time() - tstart
        t_tot = t_tot + t_dur
        cnt = cnt + 1

        # Press Q on keyboard to  exit
        ch = cv2.waitKey()
        if ch & 0xFF == ord('q'):
            print("frame processing time = ", t_tot / cnt)
            break
        elif ch == ord(' '):
            imwrite("src" + str(cnt) + ".jpg", src)
            print("src" + str(cnt) + ".jpg")
        # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
'''