# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 00:19:26 2021

@author: satish
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import Hand_Tracking_Module as htm


colors = [(255, 0, 0), (0, 255, 0),
		(0, 0, 255), (0, 255, 255)]

drawColor = colors[0]

brushThickness=15
eraserThickness = 50

#previous coordinates of pointer location
xp,yp = 0,0

#previous and current time for fps
pTime=0
cTime=0

paintWindow = np.zeros((720, 1280, 3),np.uint8)


# Loading the default webcam of PC.
cap = cv2.VideoCapture(0)

#setting width and breadth of window
cap.set(3,1280)
cap.set(4,720)

#creating object for hand tracking module for finding landmarks

detector = htm.handDetector(detectionCon=0.85)


while True:
    
    #1.import image
    ret,frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    
    #2.Find hand landmarks and display them
    frame = detector.findHands(frame)
    
    #finding landmark position of a hand
    lmList = detector.findPosition(frame,draw=False)
    
    if len(lmList)!=0:
        #print(lmList)
        
        #tip of index finger
        x1,y1 = lmList[8][1:]
        #tip of middle finger
        x2,y2 = lmList[12][1:]    
    
    
        #3.Check which fingers are up
        fingers = detector.fingersUp()
        
        print(fingers)
        
        #4.Selection mode if two fingers are up
        if fingers[1] and fingers[2]:
            
            #if mode is changed to selection from drawing the previous values should be reseted
            xp,yp=0,0
            print("Selection Mode")
            
            #choosing various options
            if y1<65:
                if 40<x1<140:
                    drawColor = (0,0,0)
                elif 160<x1<255:
                    drawColor = colors[0]
                elif 275<x1<370:
                    drawColor = colors[1]
                elif 390<x1<485:
                    drawColor = colors[2]
                elif 505<x1<600:
                    drawColor = colors[3]
                    
            cv2.rectangle(frame,(x1,y1-20),(x2,y2+20),drawColor,-1)
        
        #5.Drawing mode if index finger is up
        
        if fingers[1] and fingers[2]==0 :
            cv2.circle(frame,(x1,y1),15,drawColor,cv2.FILLED)
            print("Drawing Mode")
            
            #start drawing from current point but initially it draw just a point rather than line
            if xp ==0 and yp == 0:
                xp,yp = x1,y1
            
            
            if drawColor == (0,0,0):
                cv2.line(frame,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(paintWindow,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(frame,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(paintWindow,(xp,yp),(x1,y1),drawColor,brushThickness)
            
            xp,yp = x1,y1
        
    
    # creating the gray image of paintWindow
    grayImage = cv2.cvtColor(paintWindow,cv2.COLOR_BGR2GRAY)
    
    # converting gray image to binary image
    _,invFrame = cv2.threshold(grayImage,50,255,cv2.THRESH_BINARY_INV)
    
    invFrame = cv2.cvtColor(invFrame,cv2.COLOR_GRAY2BGR)
    
    #merging inverse frame and original frame to draw lines in inverse color i.e. black
    frame = cv2.bitwise_and(frame,invFrame)
    
    #now merging canvas and merged frame to colourize the drawn lines on frame
    frame = cv2.bitwise_or(frame,paintWindow)
    
    
    
    frame = cv2.rectangle(frame, (40, 1), (140, 65),
						(122, 122, 122), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65),
						colors[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65),
						colors[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65),
						colors[2], -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65),
						colors[3], -1)
    cv2.putText(frame, "CLEAR ALL", (49, 33),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				(255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				(255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				(255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				(255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				(150, 150, 150), 2, cv2.LINE_AA)
    

    
    cTime = time.time()
    fps = 1//(cTime-pTime)
    pTime = cTime
        
    cv2.putText(frame,str(fps),(10,70),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    				(255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Image",frame)
    #cv2.imshow("Inv",invFrame)
    cv2.imshow("Canvas",paintWindow)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    
cap.release()
cv2.destroyAllWindows()    
    
    
    
    
    
