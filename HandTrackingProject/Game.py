
import random
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

A, B = np.polyfit(x, y, 1)  # y = Ax + B

cx, cy = 250, 250
color = (255, 0, 255)
counter = 0
score = 0
timeStart = time.time()
totalTime = 20
paused = False

def restart_game():
    global timeStart, score
    timeStart = time.time()
    score = 0

# Loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if not paused:
        if time.time() - timeStart < totalTime:
            hands = detector.findHands(img, draw=False)

            if hands:
                lmList = hands[0]['lmList']
                x, y, w, h = hands[0]['bbox']
                x1, y1 = lmList[5][:2]
                x2, y2 = lmList[17][:2]

                distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
                distanceCM = A * distance + B

                if distanceCM < 40:
                    if x < cx < x + w and y < cy < y + h:
                        counter = 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10))
            if counter:
                counter += 1
                color = (0, 255, 0)
                if counter == 3:
                    cx = random.randint(100, 1100)
                    cy = random.randint(100, 600)
                    color = (255, 0, 255)
                    score += 1
                    counter = 0
            cv2.circle(img, (cx, cy), 30, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 20, (255, 255, 255), 2)

            cvzone.putTextRect(img, f'Time: {int(totalTime - (time.time() - timeStart))}', (1000, 75), scale=3, offset=20)
            cvzone.putTextRect(img, f'Score: {str(score).zfill(2)}', (100, 75), scale=3, offset=20)
        else:
            cvzone.putTextRect(img, 'Game Over', (400, 400), scale=5, offset=30, thickness=7)
            cvzone.putTextRect(img, f'Score: {score}', (450, 500), scale=3, offset=20)

            # Restart game
            cvzone.putTextRect(img, 'Press "R" to restart', (400, 600), scale=3, offset=20)
            key = cv2.waitKey(1)

            if key == ord('r'):
                restart_game()

    else:
        cvzone.putTextRect(img, 'Paused', (550, 400), scale=5, offset=30, thickness=7)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('p'):
        paused = not paused

    if key == ord('q'):
        break

cv2.destroyAllWindows()
