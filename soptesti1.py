# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

redBoundLow = np.array([0,0,100], dtype="uint8")
redBoundUp = np.array([50,56,255], dtype="uint8")
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array

        mask = cv2.inRange(image, redBoundLow, redBoundUp)

        target = cv2.bitwise_and(image, image, mask = mask)
        
	
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(imageGray, 150, 255, cv2.THRESH_BINARY)

        img2rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        
        
        output = img2rgb+target
	# show the frame
	cv2.imshow("Frame", output)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
