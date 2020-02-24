
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
	'model' : 'C:/Users/Dell/Desktop/Programming/YOLO/darkflow-master/cfg/yolo.cfg',
	'load' : 'C:/Users/Dell/Desktop/Programming/YOLO/darkflow-master/bin/yolov2.weights',
	'threshold' : 0.15,
}

tfnet = TFNet(options)
capture = cv2. VideoCapture('vid1.mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('labelled.mp4',fourcc,20.0,(640,480))
while (capture.isOpened()):
	ret, frame = capture.read()
	results = tfnet.return_predict(frame)
	if ret:
		for color, result in zip(colors,results):
			tl = (result['topleft']['x'],result['topleft']['y'])
			br = (result['bottomright']['x'],result['bottomright']['y'])
			label = result['label']
			frame = cv2.rectangle(frame, tl,br,(0,255,0), 4)
			frame = cv2.putText(frame,label,tl,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
			output.write(frame)
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
capture.release()
output.release()
cv2.destroyAllWindows()