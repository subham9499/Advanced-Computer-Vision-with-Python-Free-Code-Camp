import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0) # Camera Index

mpHands = mp.solutions.hands
hands = mpHands.Hands() #finding hands
mpDraw = mp.solutions.drawing_utils  # drawing lines

pTime = 0
cTime = 0

while True:
  success, img = cap.read()
  imRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting to RGB because hands works on RGB
  results = hands.process(imRGB)

  # print(results.multi_hand_landmarks) #prints if it detects hands

  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
      for id, lm in enumerate(handLms.landmark):
        #print(id,lm) #printing id and its x,y,z coordinate
        h, w, c = img.shape #height, width and channels
        cx, cy = int(lm.x*w), int(lm.y*h) #center cordintes of points
        print(id, cy, cy)
        if id == 0:
          cv2.circle(img, (cx,cy), 25, (255,0,255),cv2.FILLED) # 0th id will have circle
      mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # hand connections

  cTime = time.time() #getting the current time
  fps = 1/(cTime - pTime)
  pTime = cTime

  cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) #printing fps


  cv2.imshow("Image" , img)
  cv2.waitKey(1)
