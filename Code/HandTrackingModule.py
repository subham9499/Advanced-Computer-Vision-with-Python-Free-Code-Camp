import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False,maxHands=2, x=1,detectionCon = 0.5, trackCon = 0.5):  #coppied from hands.py
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon= detectionCon
        self.x = x
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        #self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)  # finding hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.x,self.detectionCon, self.trackCon)  # finding hands
        self.mpDraw = mp.solutions.drawing_utils  # drawing lines

    def findHands(self, img, draw=True):
        imRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting to RGB because hands works on RGB
        self.results = self.hands.process(imRGB)

        # print(results.multi_hand_landmarks) #prints if it detects hands

        if self.results.multi_hand_landmarks:
            for self.handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.handLms, self.mpHands.HAND_CONNECTIONS)  # hand connections
        return img

    def findPosition(self, img, handNo=0, draw =True):

        lmList = []
        if self.results.multi_hand_landmarks:
            self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(self.handLms.landmark):
                # print(id,lm) #printing id and its x,y,z coordinate
                h, w, c = img.shape  # height, width and channels
                cx, cy = int(lm.x * w), int(lm.y * h)  # center cordintes of points
                #print(id, cy, cy)
                lmList.append([id, cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)  # 0th id will have circle
        return lmList





def main():
      pTime = 0
      cTime = 0
      cap = cv2.VideoCapture(0)  # Camera Index

      detector = handDetector()
      while True:
          success, img = cap.read()

          img = detector.findHands(img, draw =True)
          lmList = detector.findPosition(img, draw =True)
          if len(lmList) !=0 :
            print(lmList[4])

          cTime = time.time()  # getting the current time
          fps = 1 / (cTime - pTime)
          pTime = cTime

          cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # printing fps

          cv2.imshow("Image", img)
          cv2.waitKey(1)


if __name__ == "__main__" :
    main()