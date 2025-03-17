import numpy as np
import math 
import time
import cv2

def invisibility():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outvid = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,400))

    vid = cv2.VideoCapture(0)

    time.sleep(2)
    count = 0
    bg = None

    for i in range(60):
        ret, bg = vid.read()

    if bg is not None:
        bg = np.flip(bg, axis = 1)

    start_time = time.time()

    while (vid.isOpened()):
        ret, img = vid.read()
        if not ret:
            break

        if (time.time() - start_time >= 60):
            return

        count += 1
        img = np.flip(img, axis = 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_skin1 = np.array([0, 0, 70])
        upper_skin1 = np.array([100, 255, 255])
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)

        lower_red2 = np.array([100, 150, 0])
        upper_red2 = np.array([140, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = mask1 + mask2

        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
        mask2 = cv2.bitwise_not(mask1)
        res1 = cv2.bitwise_and(img, img, mask=mask2)
        res2 = cv2.bitwise_and(bg, bg, mask = mask1)

        finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
        outvid.write(finalOutput)
        cv2.imshow("Invisible", finalOutput)
        cv2.waitKey(20)

    vid.release()
    outvid.release()
    cv2.destroyAllWindows()

def main():
    invisibility()

if __name__ == "__main__":
    main()