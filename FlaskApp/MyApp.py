from flask import Flask,redirect,url_for,render_template,request
import cv2
#import mediapipe as mp
import numpy as np
import time
import Hand_Tracking_Module as htm

app = Flask(__name__)


@app.route("/",methods=["POST","GET"])
def login():
    if request.method == "POST":
        return redirect(url_for("user"))
    return render_template("login.html")

@app.route("/user")
def user():
    
    colors = [(255,127,0), (0, 255, 0),
		(0, 0, 255), (0, 255, 255),(255,158,207),(0,165,255),(255,105,180),(180,105,255)]

    drawColor = colors[0]

    brushThickness=15
    eraserThickness = 50

    #previous coordinates
    xp,yp = 0,0

    #previous and current time for fps
    pTime=0
    cTime=0

    paintWindow = np.zeros((720, 1280, 3),np.uint8)



    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    detector = htm.handDetector(detectionCon=0.85)


    while True:
        
        #1.import image
        ret,frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        
        #2.Find hand landmarks
        frame = detector.findHands(frame)
        
        lmList = detector.findPosition(frame,draw=False)
        
        if len(lmList)!=0:
            #print(lmList)
            
            #tip of index finger
            x1,y1 = lmList[8][1:]
            #tip of middle finger
            x2,y2 = lmList[12][1:]    
        
        
            #3.Check which fingers are up
            fingers = detector.fingersUp()
            
            #print(fingers)
            
            #4.Selection mode if two fingers are up
            if fingers[1] and fingers[2]:
                
                #if mode is changed to selection from drawing the previous values should be reseted
                xp,yp=0,0
                #print("Selection Mode")
                
                #choosing various options
                if y1<65:
                    if 40<x1<140:
                        #something
                        drawColor = (0,0,0)
                    elif 160<x1<255:
                        #something
                        drawColor = colors[0]
                    elif 275<x1<370:
                        drawColor = colors[1]
                    elif 390<x1<485:
                        drawColor = colors[2]
                    elif 505<x1<600:
                        drawColor = colors[3]
                    elif 620<x1<715:
                        drawColor = colors[4]
                    elif 735<x1<830:
                        drawColor = colors[5]
                    elif 850<x1<945:
                        drawColor = colors[6]
                    elif 965<x1<1060:
                        drawColor = colors[7]
                    elif 1080<x1<1175:
                        paintWindow[67:,:,:]=0
                        
                cv2.rectangle(frame,(x1,y1-20),(x2,y2+20),drawColor,-1)
            
            #5.Drawing mode if index finger is up
            
            if fingers[1] and fingers[2]==0 :
                cv2.circle(frame,(x1,y1),15,drawColor,cv2.FILLED)
                #print("Drawing Mode")
                
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
        frame = cv2.rectangle(frame, (620, 1), (715, 65),
                            colors[4], -1)
        frame = cv2.rectangle(frame, (735, 1), (830, 65),
                            colors[5], -1)
        frame = cv2.rectangle(frame, (850, 1), (945, 65),
                            colors[6], -1)
        frame = cv2.rectangle(frame, (965, 1), (1060, 65),
                            colors[7], -1)
        frame = cv2.rectangle(frame, (1080, 1), (1175, 65),
                            (122,122,122), -1)
        
        cv2.putText(frame, "ERASER", (49, 33),
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
        cv2.putText(frame, "PURPLE", (635, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "ORANGE", (750, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "VIOLET", (865, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "PINK", (995, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "CLEAR ALL", (1085, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        
        
        cTime = time.time()
        fps = 1//(cTime-pTime)
        pTime = cTime
            
        cv2.putText(frame,"FPS = "+str(int(fps)),(1195,33),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (241,236,238), 2, cv2.LINE_AA)
        
        
        
        cv2.imshow("Image",frame)
        #cv2.imshow("Inv",invFrame)
        cv2.imshow("Canvas",paintWindow)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        
    cap.release()
    cv2.destroyAllWindows()

    return redirect(url_for("login"))
    




if __name__ == "__main__":
    app.run(debug=True)