import cv2
from darkflow.net.build import TFNet
from PIL import Image
import glob


options = {
	'model' : 'yolo.cfg',
	'load' : 'yolov2.weights',
	'threshold' : 0.3,
}

tfnet = TFNet(options)


for filename in glob.glob('photos/*.jpg'):
	img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
	result = tfnet.return_predict(img)
    
	for x in range(len(result)):
		tl = (result[x]['topleft']['x'],result[x]['topleft']['y'])
		br = (result[x]['bottomright']['x'],result[x]['bottomright']['y'])
		label = result[x]['label']
		img = cv2.rectangle(img, tl,br,(0,255,0), 4)
		img = cv2.putText(img,label,tl,cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),4)

	cv2.imwrite(filename+"_labelled.jpg", img)
	