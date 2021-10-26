# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:31:53 2021

@author: satish
"""


import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        #for displaying landmarks and connections
        self.mpDraw = mp.solutions.drawing_utils
        
        self.tipIds=[4,8,12,16,20]
        
        
    def findHands(self,frame,draw = True):
            
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        
        #print(self.results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                             
                #displaying landmarks on hand along with connections
                if draw:
                    self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)


        return frame
    
    
    
    def findPosition(self,frame,handNo=0,draw=True):
        
        
        self.lmList = []
        
        if self.results.multi_hand_landmarks:
            
            myHand = self.results.multi_hand_landmarks[handNo]
            
              
            #extracting landmark id and coordinates
            for id,lm in enumerate(myHand.landmark):
                #print(id,lm)
                
                #extracting height ,width, channel of frame window
                h,w,c = frame.shape
                
                #converting the decimal coordinates into pixel coordinates
                cx,cy = int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                
                self.lmList.append([id,cx,cy])
                
                #if draw:
                    #cv2.circle(frame,(cx,cy),10,(255,0,255),cv2.FILLED)
                    
        return self.lmList
    #specify whether fingers up or not
    def fingersUp(self):
        fingers = []
        
        #thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        
        #remaining 4 fingers
        for id in self.tipIds[1:]:
            if self.lmList[id][2] < self.lmList[id-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    
    #previous and current time for fps
    pTime=0
    cTime=0
    cap = cv2.VideoCapture(0)
    
    
    detector = handDetector()
    
    while True:
        
        ret, frame = cap.read()
        
        frame = cv2.flip(frame,1)
        
        frame = detector.findHands(frame)
        
        lmList = detector.findPosition(frame)
        
        if len(lmList) != 0:
            print(lmList[8])
        
        cTime = time.time()
        fps = 1//(cTime-pTime)
        pTime = cTime
        
        cv2.putText(frame,str(fps),(10,70),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    				(255, 255, 255), 2, cv2.LINE_AA)
        
        
        cv2.imshow("Image",frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()