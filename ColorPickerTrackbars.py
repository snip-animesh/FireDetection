import cv2
import numpy as np

FRAMEWIDTH = 480
FRAMEHEIGHT = 400

# cap = cv2.VideoCapture("Resources/Fire.mp4")
cap = cv2.VideoCapture(1)


def empty(a):
    pass


# Create the Trackbars
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 300)
cv2.createTrackbar("Hue Min", "TrackBars", 1, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 52, 179, empty)
cv2.createTrackbar("SAT Min", "TrackBars", 57, 255, empty)
cv2.createTrackbar("SAT Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Value Min", "TrackBars", 103, 255, empty)
cv2.createTrackbar("Value Max", "TrackBars", 255, 255, empty)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (FRAMEWIDTH, FRAMEHEIGHT))
    if not success:
        break
    # Start the video again if it ends
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    imgBlur = cv2.GaussianBlur(img , (21, 21), 0)
    imgHsv = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("SAT Min", "TrackBars")
    s_max = cv2.getTrackbarPos("SAT Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Value Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Value Max", "TrackBars")

    # print(h_min, h_max, s_min, s_max, v_min, v_max)

    lower = np.array([h_min, s_min, v_min])
    higher = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, higher)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Count Fire pixel before converting mask gray to bgr
    pxl = cv2.countNonZero(mask)
    print(pxl)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])
    # Display pixel Counting
    cv2.rectangle(hStack, (500,10),(650,40),(0, 255, 0),thickness=2)
    cv2.putText(hStack, str(pxl) , (525, 33) , fontFace=cv2.FONT_HERSHEY_COMPLEX ,
                fontScale=0.8,thickness=2 , color=(0,0,255) )
    cv2.imshow('Horizontal Stacking', hStack)



    k = cv2.waitKey(25)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
