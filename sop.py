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
        elif arg == 2: #canny              
                canny = cv2.Canny(image_gray,100,200)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
                return canny
        elif arg == 3: #ss
                print "tyhja"
        elif arg == 4: #grabCut
                print "tyhja"
        elif arg == 5: #ss
                print "tyhja"
        elif arg == 0:
                return image_gray
        return image_gray

# kuvan filterointi
def blur(arg,image_):
        if arg == 1:#Averaging
                blurred = cv2.blur(image_,(5,5))
        elif arg == 2:#Gaussian Blurring
                blurred = cv2.GaussianBlur(image_,(5,5),0)
        elif arg == 3:#Median Blurring
                blurred = cv2.medianBlur(image_,5)
        elif arg == 4:#Bilateral Filtering
                blurred = cv2.bilateralFilter(image_,9,75,75)
        elif arg == 5:#pyramid mean shift filtering   huom vain varikuvalle
                blurred = cv2.pyrMeanShiftFiltering(image_, 5, 7)
        elif arg == 0:
                blurred = image_
        return blurred

# laske cosini kolmen pisteen valilla
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

# etsi suurin contour jossa on 4 kulmaa ja ne on tarpeeksi lahella 90 astetta
def contourfindrectangle(image,image_bw):
        global centroidx, centroidy, areafoundfirsttime, areafound
        biggest = []
        contours, hierarchy = cv2.findContours(image_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        maximumarea = 0
        for cnt in contours:                
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)  # alkuperainen: 0.02*cnt_len
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
                               
        if len(biggest) > 0: #etsi centroid (jos alue on loytynyt)
                areafoundfirsttime = True
                areafound = True
                m = cv2.moments(biggest[0])
                centroidx = int(m['m10']/m['m00'])
                centroidy = int(m['m01']/m['m00'])                
                return biggest[0],(centroidx,centroidy) # pelialue loytyi
        areafound = False
        return 0,(centroidx,centroidy) # pelialuetta ei loytynyt

#etsi contour jossa on edellisen framen centroid
def contourThatHasCentroid(image_bw):
        global centroidx, centroidy, areafound
        contours, hierarchy = cv2.findContours(image_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
            cnt = cv2.convexHull(cnt)
            #tarkista onko edellinen centroid uudessa alueessa
            insidearea = cv2.pointPolygonTest(cnt, (centroidx, centroidy), False) 
            if insidearea == 1:
                #laske centroid
                m = cv2.moments(cnt)
                centroidx = int(m['m10']/m['m00'])
                centroidy = int(m['m01']/m['m00'])
                areafound = True
                return cnt , (centroidx,centroidy) # pelialue loytyi
        areafound = False
        return 0,(centroidx,centroidy) # pelialuetta ei loytynyt

def findhand(image):#ei toimi
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        #kasi
##        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
##        cv2.inRange(divide, ls1, ls2)
##        divide = cv2.divide(hls[:,:,1],hls[:,:,2])        
##        dividemask = cv2.inRange(divide, ls1, ls2)
##        mask1 = cv2.inRange(hls, hlslow1, hlsup1)
##        mask2 = cv2.inRange(hls, hlslow2, hlsup2)
##        mask3 = cv2.bitwise_or(mask1, mask2)
##        mask = cv2.bitwise_and(mask3, dividemask)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        #skinMask = cv2.dilate(skinMask, None)
        #skinMask = cv2.erode(skinMask, None)
	#skinMask = cv2.erode(skinMask, None)
	#skinMask = cv2.dilate(skinMask, None)
	#skinMask = blur(2,skinMask)
	image.flags.writeable = True
	image[mask == 255] = [255, 0, 0]
	return

# kameran alustuksia
width = 320  # 640   400  320  # kuvan leveys
height = 240 # 480   300  240  # kuvan korkeus
camera = PiCamera()
camera.resolution = (width, height)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(width, height))

#alusta maski
watershedMask = np.zeros((height, width), dtype=np.uint8)
#jatka lisaa nelio ja sen jalkeen watershed

#pallon alustus
ballx = 0  # pallon x-koordinaati sub-pixel tarkkuudella (etaisyys centroidista)
bally = 0  # pallon y-koordinaati sub-pixel tarkkuudella (etaisyys centroidista)
pointspeed = 20  # pallon nopeus: pikselia/framessa
pointangle = random.uniform(-np.pi, np.pi) # pallon suunta (random aloitus suunta pallolle from -pi to pi)

#lippuja ja alustuksia
sethsv = False  # onko hsv arvot kadesta luettu
setrgb = False  # onko rgb arvot kadesta luettu
rgbflag = False  # kaytetaanko kaden varien lukemiseen hsv vai rgb variavaruutta. False = hsv, True = rgb
start = time.time() #alusta kello
centroidx = 0  # pelialueen keskipisteen x koordinaatti
centroidy = 0  # pelialueen keskipisteen y koordinaatti
touch = 0 # montako kertaa pallo on osunut kateen
areafoundfirsttime = False # onko suorakulmainen alue loytynyt
areafound = False  # onko talla framella loytynyt pelialuetta

#findhand funktion muuttujia. poista?
ls1 = np.array([0.5], dtype = "float")
ls2 = np.array([2], dtype = "float")
hlslow1 = np.array([165, 0, 50], dtype = "uint8")
hlsup1 = np.array([179, 255, 255], dtype = "uint8")
hlslow2 = np.array([0, 0, 50], dtype = "uint8")
hlsup2 = np.array([14, 255, 255], dtype = "uint8")

# min ja max arvot   hsv  (kaden tunnistus)
hsv_min = np.array([0, 0, 0], dtype = "uint8")
hsv_max = np.array([255, 255, 255], dtype = "uint8")

# min ja max arvot   rgb  (kaden tunnistus)
redBoundLow = np.array([0,0,100], dtype="uint8")
redBoundUp = np.array([50,56,255], dtype="uint8")

# viive kameraa varten
time.sleep(0.5)

# varmista etta key on olemassa
key = cv2.waitKey(1) & 0xFF


# fullscreen, mutta fps laskee paljon (poista molemmat kommentit)
#cv2.namedWindow("sop", cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE  tai cv2.WINDOW_NORMAL
#cv2.setWindowProperty("sop", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)


# ota kuvia kamerasta
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	#tee NumPy taulukko kameran kuvasta
	image = frame.array

        #muuta kuva mustavalkoiseksi
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #muuta kuva hsv vareiksi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        #suodatus
        #0 == ei mitaan
        #1 == Averaging
        #2 == Gaussian Blurring
        #3 == Median Blurring
        #4 == Bilateral Filtering
        image_gray = blur(3, image_gray);

        #segmentointi, jos parametri on:
        #0 == ei mitaan
        #1 == threshold (otsu)
        #2 == canny
        #3 == tyhja
        #4 == tyhja
        #5 == tyhja
        image_bw = segmentation(2,image_gray);

        #testi kuva
	cv2.imshow("canny", image_bw)
	key = cv2.waitKey(1) & 0xFF
	
        #dilation
        #image_bw = cv2.dilate(image_bw, None)
        
        # valitse edellisen framen(tai yleensa aloitus piste) centroid kuvan keskipisteesta, b napilla
        if key == ord("b"):
            areafoundfirsttime = True
            centroidx = width/2
            centroidy = height/2

        # etsi pelialue(=cnt) ja sen keskipiste(=origin)
        if areafoundfirsttime is False:
                cnt, origin = contourfindrectangle(image,image_bw) # eka frame tai kokoajan
        if areafoundfirsttime is True:
                cnt, origin = contourThatHasCentroid(image_bw) # jos tiedetaan edellinen centroid (ei toimi)

        # pallon liikkuminen
        if areafound is True:
                #laske pallon uudet koordinaatit
                ballx = np.cos(pointangle)*pointspeed+ballx # pallon x-koordinaatti centroidista
                bally = -(np.sin(pointangle)*pointspeed)+bally # pallon y-koordinaatti centroidista
                # pyorista pallon koordinaatit lahinpaan pikseliin ja muuta normaalin koordinaatistoon (ei centroid keskinen)
                roundedBallCoordinates = (origin[0]+int(round(ballx)), origin[1]+int(round(bally)))
                
                # tarkista onko pallo pelialueessa
                inside = cv2.pointPolygonTest(cnt, roundedBallCoordinates, False)
                
                if inside == -1: # pallo ei ole pelialueessa                                                
                        # palaa pelialueeseen (palaa painvastaiseen suuntaan)
                        returnangle = -(np.pi-pointangle)
                        outofbounds = 0                        
                        while True:
                                ballx = np.cos(returnangle)+ballx
                                bally = -(np.sin(returnangle))+bally
                                roundedBallCoordinates = (origin[0]+int(round(ballx)), origin[1]+int(round(bally)))
                                inside = cv2.pointPolygonTest(cnt, roundedBallCoordinates, False)
                                if inside >= 0:# pallo on palannut pelialueen reunaan
                                        break
                                if outofbounds > 50: # pallo kaukana pelialueen ulkopuolella
                                        # pallo palautuu pelialueen keskipisteeseen
                                        ballx = 0
                                        bally = 0
                                        roundedBallCoordinates = (origin[0], origin[1])
                                        break
                                outofbounds = outofbounds + 1
                                
                        if outofbounds <= 50: # pallo on palannut pelialueen reunaan -> lasketaan kimpoamiskulma
                                # laske pallon kimpoamiskulma
                                minlength = 10000
                                for c in cnt:
                                        # etsi lahin piste
                                        d1 = c[0][0] - (ballx + origin[0])
                                        d2 = c[0][1] - (bally + origin[1])
                                        length = np.sqrt(d1*d1+d2*d2)
                                        if minlength > length:
                                                minlength = length
                                                nearest = c[0]
                                cv2.circle(image, (nearest[0],nearest[1]), 4, (255,255,255), -1) #testaus: piirra lahin kulma
                                cv2.line(image,(nearest[0],nearest[1]),roundedBallCoordinates,(0,0,255),2) #testaus: piirra kimpoamisseina
                                # kimpoamisseinan kulma
                                wallangle = np.arctan2((bally + origin[1])-nearest[1], nearest[0]-(ballx + origin[0]))
                                # laske uusi pallon suunta
                                pointangle = -(pointangle - wallangle)
                                pointangle = pointangle + wallangle
      

##        if valuesset is True:
##                findhand(image)

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
      

        # ota varit keskineliosta hsv tasoon
        if  (key == ord("v")) and (rgbflag is False):
                sethsv = True
                h = []
                s = []
                v = []
                for i in range(height/2-10, height/2+10):
                        for j in range(width/2-10, width/2+10):
                                h.append(hsv[i][j][0])
                                s.append(hsv[i][j][1])
                                v.append(hsv[i][j][2])
                hsv_min[0] = min(h)
                hsv_min[1] = min(s)
                hsv_min[2] = min(v)
                hsv_max[0] = max(h)
                hsv_max[1] = max(s)
                hsv_max[2] = max(v)

        
        # ota varit keskineliosta rgb tasoon    poista?
        if key == ord("v") and rgbflag is True:
                setrgb = True
                r = []
                g = []
                b = []
                for i in range(height/2-10, height/2+10):
                        for j in range(width/2-10, width/2+10):
                                b.append(image[i][j][0])
                                g.append(image[i][j][1])
                                r.append(image[i][j][2])
                redBoundLow[0] = min(b)
                redBoundLow[1] = min(g)
                redBoundLow[2] = min(r)
                redBoundUp[0] = max(b)
                redBoundUp[1] = max(g)
                redBoundUp[2] = max(r)
       
        if sethsv is True:  # laske missa kasi on hsv arvoilla     huom melko raskas  
                mask = cv2.inRange(hsv, hsv_min, hsv_max)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.dilate(mask,kernel,iterations = 2)
                image.flags.writeable = True
                image[mask == 255] = [255, 0, 0]
        if setrgb is True:  # laske missa kasi on rgb arvoilla       
                mask = cv2.inRange(image, redBoundLow, redBoundUp)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.dilate(mask,kernel,iterations = 1)
                image.flags.writeable = True
                image[mask == 255] = [255, 0, 0]

        # katso osuuko pallo kateen
        if ((sethsv is True) or (setrgb is True)) and (areafound is True):
                if mask[roundedBallCoordinates[1], roundedBallCoordinates[0]] == 255:
                        touch = touch + 1
                        ballx = 0
                        bally = 0
                        roundedBallCoordinates = (origin[0], origin[1])
                        pointangle = random.uniform(-np.pi, np.pi)
        
        #piirra keskinelio
        cv2.line(image,(width/2-10, height/2-10),(width/2-10, height/2+10),(0,255,0),1)
        cv2.line(image,(width/2-10, height/2-10),(width/2+10, height/2-10),(0,255,0),1)
        cv2.line(image,(width/2+10, height/2+10),(width/2+10, height/2-10),(0,255,0),1)
        cv2.line(image,(width/2+10, height/2+10),(width/2-10, height/2+10),(0,255,0),1)

        # piirra centroid, pelialue ja pallo
        if areafound is True:
                # pelialue
                cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1 )
                # centroid
                cv2.circle(image, origin, 2, (255,255,255), -1)
                # pallo
                cv2.circle(image, roundedBallCoordinates, 4, (0,0,255), -1)

        # lisaa fps kuvaan
        fps = 1/(time.time()-start)
        start = time.time()
        cv2.putText(image, str(int(round(fps))),(5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

        # lisaa osumat kuvaan
        cv2.putText(image, str(int(round(touch))),((width-25),15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

        # nayta kuva
	cv2.imshow("sop", image)


	key = cv2.waitKey(1) & 0xFF

        # tyhjenna stream seuraavaa kuvaa varten
	rawCapture.truncate(0)
	# jos painetaan `q` nappia, niin ohjelma loppuu
	if key == ord("q"):
		break
cv2.destroyAllWindows()

