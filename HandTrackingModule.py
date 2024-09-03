import cv2 as cv
import mediapipe as mp
import time


class HANDDETECTOR:
    def __init__(self, mode=False, maxhands=2, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxhands,
                                        min_detection_confidence=self.detectioncon,
                                        min_tracking_confidence=self.trackcon)
        self.mpdraw = mp.solutions.drawing_utils

    def findhand(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)
        return img

    def findposition(self, img, handNo=0, draw=True):
        lmlist = []

        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]

                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmlist.append([id, cx, cy])
                    if draw:
                        cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)
        return lmlist


def main():
    pTime = 0
    cTime = 0

    cap = cv.VideoCapture(0)
    detector = HANDDETECTOR()

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

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
