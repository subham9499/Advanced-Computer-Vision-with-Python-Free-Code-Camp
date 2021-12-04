import cv2
import mediapipe as mp
import time

pTime = 0

cap = cv2.VideoCapture("../Videos/3.mp4")
# cap = cv2.VideoCapture(0)  # webcam

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


while True:
    success, img = cap.read(0.75)  # 75% con

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            bboxC = detection.location_data.relative_bounding_box  # bounding box
            h, w, c = img.shape  # height width channel
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (255,0,255), 2)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) #printing fps
    cv2.imshow("Image", img)
    cv2.waitKey(1)  #frame rate set, value is more implies slower