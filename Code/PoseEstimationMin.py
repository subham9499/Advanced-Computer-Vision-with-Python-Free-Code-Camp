import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose

pTime = 0
#cap = cv2.VideoCapture('/home/user/work/FreeCodingAcademy/Advanced-Computer-Vision-with-Python-Free-Code-Camp/Videos/1.mp4')
cap = cv2.VideoCapture('../Videos/1.mp4')

while True:
    success, img = cap.read()



    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Image", img)

    cv2.waitKey(1)