import cv2 as cv 
import mediapipe as mp
import time
import HandTrackingModule as htm


pTime = 0
cTime = 0

cap = cv.VideoCapture(0)
detector = htm.HANDDETECTOR()

while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findhand(img)
        lmlist = detector.findposition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("video", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

