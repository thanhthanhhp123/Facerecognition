import numpy as np
import pickle
import cv2


model = pickle.load(open('E:\Download\Facerecognition-main (1)\Facerecognition-main\src\models\SVM_model.pkl', 'rb'))
classes = ['dong', 'khai', 'quan', 'thanh']

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 640)

while True:
	ret, frame = cap.read()
	img = cv2.flip(frame, 1)

	cv2.putText(img, 'Put your face under here', (400, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
	cv2.rectangle(img, (400, 200), (800, 600), (255, 0, 0))

	face = img[200:600, 400:800]
	face = face / 255.0
	face = face.reshape((1, -1))
	pred = model.predict(face)
	name = np.squeeze(pred)

	cv2.putText(img, 'Hello {}'.format(classes[name]), (400, 640), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

	cv2.imshow('Image', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()