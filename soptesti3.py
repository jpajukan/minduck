from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import random


def segmentation(arg,image_gray):
        if arg == 1: #threshold segmentaatio
                (thresh, image_bw) = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                return image_bw
        elif arg == 2: #watershed? segmentaatio floodfill?
                print "testi"
        elif arg == 0:
                return image_gray
        return

# kuvan filterointi
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

# laske cosinin kolmen pisteen valilla
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def countour(image,image_bw):
        # etsi suorakulmaiset contourit        
        biggest = []
        contours, hierarchy = cv2.findContours(image_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        maximumarea = 0
        for cnt in contours:
                
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                area = cv2.contourArea(cnt)
                
                if len(cnt) == 4 and area > 1000 and cv2.isContourConvex(cnt):
                        orig = cnt
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                        
                        if max_cos < 0.5:  # kulmien asteiden heitto 90sta asteesta
                                if maximumarea < area: # tallenna suurin contour
                                        if maximumarea != 0:
                                                biggest.pop()
                                        biggest.append(orig)
                                        maximumarea = area

        #piirra suurin contour
        cv2.drawContours( image, biggest, 0, (0, 255, 0), -1 )
        #etsi centroid
        cx = 20
        cy = 40
        if len(biggest) > 0:
                m = cv2.moments(biggest[0])
                cx = int(m['m10']/m['m00'])
                cy = int(m['m01']/m['m00'])
                return biggest[0],(cx,cy) # pelialue loytyi
        return 0,(cx,cy) # pelialuetta ei loytynyt

def findhand(image):#ei toimi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(image, lower, upper)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.dilate(skinMask, None)
        skinMask = cv2.erode(skinMask, None)
	skinMask = cv2.erode(skinMask, None)
	skinMask = cv2.dilate(skinMask, None)
	#skinMask = blur(2,skinMask)
	image.flags.writeable = True
	image[skinMask == 255] = [0, 0, 255]
	return



# initialize the camera and grab a reference to the raw camera capture
width = 640
height = 480
#pallon alustus
pointdx = 0
pointdy = 0
pointspeed = 25
#random aloitus suunta pallolle from -pi to pi
pointangle = random.uniform(-np.pi, np.pi)   
flagoutside = 0

camera = PiCamera()
camera.resolution = (width, height)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(width, height))


lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

redBoundLow = np.array([0,0,100], dtype="uint8")
redBoundUp = np.array([50,56,255], dtype="uint8")

# viive kameraa varten
time.sleep(0.5)

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
        #image_bw = cv2.Canny(image_bw,100,200)

        #dilation
        #image_bw = cv2.dilate(image_bw, None)

        # etsi pelialue ja sen keskipiste countoureilla
        cnt, origin = countour(image,image_bw)

        # pallon liikkuminen
        if type(cnt) is not int:

                # jatka tasta
                
                # tarkista onko pallo pelialueessa
                inside = cv2.pointPolygonTest(cnt, roundedpoint, False)
                if inside == -1: # pallo ei ole pelialueessa
                        # laske pallon kimpoamiskulma
                        minlength = 10000
                        for c in cnt:
                                # etsi lahin piste
                                d1 = c[0][0] - (pointdx + origin[0])
                                d2 = c[0][1] - (pointdy + origin[1])
                                length = np.sqrt(d1*d1+d2*d2)
                                if minlength > length:
                                        minlength = length
                                        nearest = c[0]
                        cv2.circle(image, (nearest[0],nearest[1]), 4, (255,255,255), -1)
                        wallangle = np.arctan2((pointdy + origin[1])-nearest[1], nearest[0]-(pointdx + origin[0])) # seinan kulma
                        print wallangle
                        pointangle = -(pointangle - wallangle)
                        pointangle = pointangle + wallangle

# jatka tasta# jatka tasta
                #laske pallon uudet koordinaatit
                pointdx = np.cos(pointangle)*pointspeed+pointdx
                pointdy = -(np.sin(pointangle)*pointspeed)+pointdy
                roundedpoint = (origin[0]+int(round(pointdx)), origin[1]+int(round(pointdy)))

                # piirra pallo
                cv2.circle(image, roundedpoint, 4, (0,0,255), -1)
        
                
        # piirra keskipiste
        if type(cnt) is not int:
                cv2.circle(image, origin, 2, (255,255,255), -1)

        #findhand(image)

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
cv2.destroyAllWindows()
