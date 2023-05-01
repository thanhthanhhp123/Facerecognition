import os
import cv2
import time

start = 0
name = input('Enter your name: ')
i = 0
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 640)
os.mkdir('Datasets/'+name)
while True:
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)
    cv2.putText(img, 'Put your face to the rectangle', (350, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.rectangle(img, (400, 200), (800, 600), (255, 0))
    cv2.imshow('Capture', img)
    now = time.time()
    if now - start >= 0.5:
        i+=1
        cv2.imwrite('Datasets/'+name+f'/image{i}.jpg', img[200:600, 400:800])
        start = now
    if cv2.waitKey(1) & 0xFF == ord('q') or i == 100:
        break
cap.release()
cv2.destroyAllWindows()