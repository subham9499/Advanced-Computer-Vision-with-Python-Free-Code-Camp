import cv2
import mediapipe as mp
import time

class faceDetector():
    def __init__(self, detctionCon=0.5, modelSel=0):
        self.detectionCon  = detctionCon
        self.modelSel = modelSel

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.detectionCon, self.modelSel)

    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection)
                # print(id, detection)
                bboxC = detection.location_data.relative_bounding_box  # bounding box
                h, w, c = img.shape  # height width channel
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
        return img

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture("../Videos/3.mp4")

    detector = faceDetector()
    while True:
        success, img = cap.read(0.75)

        img = detector.findFace(img, draw=True)
        #lmList = detector.findPosition(img, draw=True)
        #if len(lmList) != 0:
        #    print(lmList[4])

        cTime = time.time()  # getting the current time
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # printing fps

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":  # Will run main() if running this file, if calling another function, it won't run main()
    main()