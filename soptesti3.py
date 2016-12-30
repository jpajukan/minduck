from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

def segmentation(arg,image_gray):
        if arg == 1: #threshold segmentaatio
                (thresh, image_bw) = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                return image_bw
        elif arg == 2: #watershed? segmentaatio floodfill?
                print "testi"
        elif arg == 0:
                return image_gray
        return

def blur(arg,image_gra):
        if arg == 1:#Averaging
                image_gray = cv2.blur(image_gra,(5,5))
        elif arg == 2:#Gaussian Blurring
                image_gray = cv2.GaussianBlur(image_gra,(5,5),0)
        elif arg == 3:#Median Blurring
                image_gray = cv2.medianBlur(image_gra,5)
        elif arg == 4:#Bilateral Filtering
                image_gray = cv2.bilateralFilter(image_gra,9,75,75)
        elif arg == 0:
                image_gray = image_gra
        return image_gray

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )        

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))



redBoundLow = np.array([0,0,100], dtype="uint8")
redBoundUp = np.array([50,56,255], dtype="uint8")

# viive kameraa varten
time.sleep(1)

# ota kuvia kamerasta
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	#tee NumPy taulukko kameran kuvasta
	image = frame.array

        #muuta kuva mustavalkoiseksi
        image_gra = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #suodatus
        #0 == ei mitaan
        #1 == Averaging
        #2 == Gaussian Blurring
        #3 == Median Blurring
        #4 == Bilateral Filtering
        image_gray = blur(3,image_gra);

        #segmentointi, jos parametri on:
        #0 == ei mitaan
        #1 == threshold (otsu)
        #2 == watershed?
        image_bw = segmentation(1,image_gray);

        #canny reunantunnistus
        edges = cv2.Canny(image_bw,100,200)

        #dilation
        edges = cv2.dilate(edges, None)
        
        squares = []
        contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
                
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)

                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                        if max_cos < 0.5:
                                squares.append(cnt)

        cv2.drawContours( image, squares, 0, (0, 255, 0), -1 )


##        houghLines = cv2.HoughLinesP(edges,rho=0.3,theta=np.pi/200, threshold=10,lines=np.array([]),minLineLength=50,maxLineGap=30)        
##
##        if houghLines is not None:  # tarkista etta houghlines on olemassa
##                houghLines = houghLines[0]
##                for x1,y1,x2,y2 in houghLines: #piirra houghlinet kuvaan
##                        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
        
        # GrabCut
##        mask = np.zeros(image.shape[:2],np.uint8)
##        bgdModel = np.zeros((1,65),np.float64)
##        fgdModel = np.zeros((1,65),np.float64)
##        rect = (50,50,450,290)
##        cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
##        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

        #mask = cv2.inRange(image, redBoundLow, redBoundUp)
        #target = cv2.bitwise_and(image, image, mask = mask)
       	
        #imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #retval, dst = cv2.threshold(imageGray, 150, 255, cv2.THRESH_BINARY)

        #img2rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
                
        #output = img2rgb+target

                                
	# nayta kuva
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
 
	# tyhjenna stream seuraavaa kuvaa varten
	rawCapture.truncate(0)
 
	# jos painetaan `q` nappia, niin ohjelma loppuu
	if key == ord("q"):
		break
