import cv2
import mediapipe as mp
import time


class poseDetector():
    # refer pose.py
    def __init__(self, mode= False, mComp = 1, upperBody= False, smooth = True, detectionCon = 0.5, trackingCon = 0.5 ):
        self.mode = mode  # variable of that object
        self.mComp = mComp
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.mComp, self.upperBody, self.smooth, self.detectionCon, self.trackingCon)

    def findPose(self, img, draw= True):  # method
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw= True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id,lm)
                cx, cy = int(lm.x*w), int(lm.y*h) # gives pixel values for the landmarks
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList






def main():
    pTime = 0
    cap = cv2.VideoCapture('../Videos/3.mp4')

    detector = poseDetector()


    while True:
        success, img = cap.read()

        img = detector.findPose(img)

        lmList = detector.findPosition(img, draw= False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 0), cv2.FILLED)  # tracing point 4

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__": #Will run main() if running this file, if calling another function, it won't run main()
    main()