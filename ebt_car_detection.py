import cv2
import numpy as np

cap = cv2.VideoCapture("video2.mp4")
car_detection = cv2.CascadeClassifier("cars.xml")

while True:
    ret,frame = cap.read()
    if frame is None:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_detection_gray = car_detection.detectMultiScale(frame_gray,1.2,3)
    
    
    for a,b,c,d in car_detection_gray:
        cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),2)
    """
    for a,b,c,d in car_detection_gray:
        x = a+c
        y = b+d
        if a>800 and b>300:
            if x>1000 and y>500:
                cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),2)
    """
    cv2.imshow("Car Detections",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows() 