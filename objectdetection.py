#IMANIPOUR Meysam.
#Master factory of the futrue 2020.
#Object and color detection with OpenCV and Python.
#Today 28/02/2021.
#==========================================================================================
#ReadMe : 
# this project will detect some shapes like square,triangle,cross and arrow and it also 
# will detect the color of these shapes if the colors are yellow, red or blue.
# for the arrow shape it will detect also it's direction by drawing another green arrow.
# for the color detection I used different boundary in HSV domain.
# for the shape detection I used contours and hierarchy to define the hole in some shapes.
# for the direction of the arrow I implemented my own functions to find head and bottom 
# of the arrow based specefic angle.
# for more information please contanct me.
# meysam.imani_pour@ensam.eu
# imanipourmeysam@gmail.com
#==========================================================================================
import cv2 as cv    
import numpy as np
import urllib
import math
from numpy import linalg
#===========================================================================================

# function that does nothing.
def relax(): 
    
    pass
   
# function gets coordinate of the corners.
def xy_coor(approx):
    neg_pos = np.array([-1,-1])
    coordinate = np.array((neg_pos,neg_pos,neg_pos,neg_pos,neg_pos,neg_pos,neg_pos,neg_pos,neg_pos))
    it = 0
    for i in range(len(approx)):
        coordinate[i][0] = approx.ravel()[it]
        it += 1
        coordinate[i][1] = approx.ravel()[it]
        it += 1
    return coordinate 

# function calculate angle between three point.
def angle(a,b,c):
    ba = b-a
    bc = b-c
    ans = np.dot(ba,bc)
    mag = np.linalg.norm(ba)*np.linalg.norm(bc)
    theta = np.arccos(ans/mag)
    theta = math.degrees(theta)
    return theta

# function that find head and back of the arrow for the direction based angle.
def head_back(coordinate):
    head_back = np.array(([-1,-1],[-1,-1]))
    for i in range(len(coordinate)):
        a = coordinate[i]
        if (i>6):
            c = coordinate[i+2-9]
        else :
            c = coordinate[i+2]
        if (i>7):
            b = coordinate[i+1-9]
        else:
            b = coordinate[i+1]

        ang = angle(a,b,c)
        if 43<= ang <= 47:
            head_back[0] = b
            ba = np.linalg.norm((b-a))
            bc = np.linalg.norm((b-c))
            if (ba > bc):
                head_back[1] = a
            else:
                head_back[1] = c
    return head_back

cv.namedWindow("frame",cv.WINDOW_NORMAL)
cv.namedWindow("canny",cv.WINDOW_NORMAL)
cv.namedWindow("dilation",cv.WINDOW_NORMAL)
#===========================================================================================

URL = "https://192.168.1.49:8080" # this url differ because it depends to network connection.
phonecam = cv.VideoCapture(URL+"/video")
font = cv.FONT_HERSHEY_COMPLEX
#------temporary-----------
cv.namedWindow("parameter")
cv.createTrackbar("threshold1","parameter",130,255,relax)
cv.createTrackbar("threshold2","parameter",130,255,relax)
#===========================================================================================
#Color boundary of red,yellow and bleu of the arrow.
lower_blue = np.array([90,60,0])
upper_blue = np.array([120,255,255])
lower_red = np.array([0,50,120])
upper_red = np.array([10,255,255])
lower_yellow = np.array([25,70,120])
upper_yellow = np.array([30,255,255])
#==========================================================================================
while True:
    ret,frame = phonecam.read()
    frame_blur = cv.GaussianBlur(frame,(7,7),1)
    frame_gray = cv.cvtColor(frame_blur,cv.COLOR_BGR2GRAY)
    threshold1 = cv.getTrackbarPos("threshold1","parameter")
    threshold2 = cv.getTrackbarPos("threshold2","parameter")
    frame_canny = cv.Canny(frame_gray,threshold1,threshold2)
    kernel = np.ones((5,5))
    frame_dilation = cv.dilate(frame_canny,kernel,iterations = 1)
    contours,hierarchy = cv.findContours(frame_dilation,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area > 1000:
            param_holder = cv.arcLength(contours[i],True)
            approx = cv.approxPolyDP(contours[i],0.02*param_holder,True)
            if(len(approx) == 3):
                if hierarchy[0][i][2] != -1:
                    x,y,w,h = cv.boundingRect(contours[i])
                    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
                    cv.putText(frame,"Triangle",(x,y-4),font,0.5,(0,0,0))
            if(len(approx) == 4):
                state = True
                for j in range(2,4):
                    if hierarchy[0][i][j] == -1:
                        state = False
                if (state):
                    x,y,w,h = cv.boundingRect(contours[i])
                    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
                    cv.putText(frame,"Square",(x,y-4),font,0.5,(0,0,0))
            if(len(approx) == 12):
                x,y,w,h = cv.boundingRect(contours[i])
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
                cv.putText(frame,"Cross",(x,y-4),font,0.5,(0,0,0))

            if(len(approx) == 9):
                x,y,w,h = cv.boundingRect(contours[i])
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
                cv.putText(frame,"Arrow",(x,y-4),font,0.5,(0,0,0))
                pos = xy_coor(approx)
                head_back = head_back(pos)
                p1 = (head_back[1][0],head_back[1][1])#back_point of arrow
                p2 = (head_back[0][0],head_back[0][1])#head_point of arrow
                cv.arrowedLine(frame,p1,p2,(0,255,0),2)
                roi = frame[x-50:x+w+50,y-50:y+h+50]
                frame_hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
                mask_blue = cv.inRange(frame_hsv,lower_blue,upper_blue)
                mask_red = cv.inRange(frame_hsv,lower_red,upper_red)
                mask_yellow = cv.inRange(frame_hsv,lower_yellow,upper_yellow)

                B_cnt,_ = cv.findContours(mask_blue,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                R_cnt,_ = cv.findContours(mask_red,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                Y_cnt,_ = cv.findContours(mask_yellow,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                for cnt in B_cnt:
                    u, v, d, l = cv2.boundingRect(cnt)
                    cv.putText(frame,"BLUE",(u+100,v-10),)
                for cnt in R_cnt:
                    u, v, d, l = cv.boundingRect(cnt)
                    cv.putText(frame,"RED",(u+100,v-10),)
                for cnt in Y_cnt:
                    u, v, d, l = cv2.boundingRect(cnt)
                    cv.putText(frame,"YELLOW",(u+100,v-10),)
                
                
    cv.imshow("frame",frame)
    cv.imshow("canny",frame_canny)
    cv.imshow("dilation",frame_dilation)
    if cv.waitKey(1) == 27:
        break
phonecam.release()
cv.destroyAllWindows()