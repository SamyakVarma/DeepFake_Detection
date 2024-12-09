from ultralytics import YOLO
import cv2 as cv

model = YOLO("DeepFakeDetect\\best.pt")
vid = cv.VideoCapture(0)

while(True):
    ret,frame = vid.read()
    results = model(frame)
    for box in results[0].boxes:
        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bottom_left_x = int(box.xyxy.tolist()[0][2])
        bottom_left_y = int(box.xyxy.tolist()[0][3])
        cv.rectangle(frame,(top_left_x-10,top_left_y-10),(bottom_left_x+10,bottom_left_y+10),(50,200,129),2)
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
