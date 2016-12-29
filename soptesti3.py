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
time.sleep(2)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array

        #image_gray = cv2.imread(imfile,cv2.CV_LOAD_IMAGE_GRAYSCALE);

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #retval, dst = cv2.threshold(imageGray, 150, 255, cv2.THRESH_BINARY)

        #img2rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)

        (thresh, image_bw) = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        edges = cv2.Canny(image_bw,100,200)

        #houghLines = cv2.HoughLinesP(edges, 1, 3.14/180, 10, 100, minLineLength=50, maxLineGap=30)
        houghLines = cv2.HoughLinesP(image=edges,rho=0.3,theta=np.pi/200, threshold=10,lines=np.array([]),minLineLength=50,maxLineGap=30)

        #print houghLines

        #minLineLength = 30
        #maxLineGap = 10
        #lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
        if len(houghLines) > 0:
                houghLines = houghLines[0]
                #print len(houghLines)
                #print houghLines
                #for x in range(0, len(houghLines)):
                for x1,y1,x2,y2 in houghLines:
                        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)


        #mask = cv2.inRange(image, redBoundLow, redBoundUp)

        #target = cv2.bitwise_and(image, image, mask = mask)
        
	
        #imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #retval, dst = cv2.threshold(imageGray, 150, 255, cv2.THRESH_BINARY)

        #img2rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        
        
        #output = img2rgb+target
	# show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
