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
        global cx, cy
        biggest = []
        contours, hierarchy = cv2.findContours(image_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        maximumarea = 0
        for cnt in contours:
                
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)  # alkuperainen: 0.02*cnt_len
                area = cv2.contourArea(cnt)
                
                if len(cnt) == 4 and area > 1000 and cv2.isContourConvex(cnt): # len(cnt) == 4 and 
                        orig = cnt
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                        
                        if max_cos < 0.5:  # kulmien asteiden heitto 90sta asteesta
                                if maximumarea < area: # tallenna suurin contour
                                        if maximumarea != 0:
                                                biggest.pop()
                                        biggest.append(orig)
                                        maximumarea = area

        #piirra suurin contour (testausta varten)
        cv2.drawContours( image, biggest, 0, (0, 255, 0), -1 )

                                        
        #etsi centroid
        if len(biggest) > 0:
                m = cv2.moments(biggest[0])
                cx = int(m['m10']/m['m00'])
                cy = int(m['m01']/m['m00'])                
                return biggest[0],(cx,cy) # pelialue loytyi
        return 0,(cx,cy) # pelialuetta ei loytynyt

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
width = 320  # 640   400  320
height = 240 # 480   300  240
camera = PiCamera()
camera.resolution = (width, height)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(width, height))

#pallon alustus
pointdx = 0
pointdy = 0
pointspeed = 20
#random aloitus suunta pallolle from -pi to pi
pointangle = random.uniform(-np.pi, np.pi)

flagoutside = 0
sethsv = False
setrgp = False
rgpflag = False
start = time.time() #alusta kello
cx = 20
cy = 40

ls1 = np.array([0.5], dtype = "float")
ls2 = np.array([2], dtype = "float")
hlslow1 = np.array([165, 0, 50], dtype = "uint8")
hlsup1 = np.array([179, 255, 255], dtype = "uint8")
hlslow2 = np.array([0, 0, 50], dtype = "uint8")
hlsup2 = np.array([14, 255, 255], dtype = "uint8")

hsv_min = np.array([0, 0, 0], dtype = "uint8")
hsv_max = np.array([255, 255, 255], dtype = "uint8")

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

redBoundLow = np.array([0,0,100], dtype="uint8")
redBoundUp = np.array([50,56,255], dtype="uint8")

# viive kameraa varten
time.sleep(0.5)
key = cv2.waitKey(1) & 0xFF


# fullscreen mutta fps laskee paljon
#cv2.namedWindow("sop", cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE  tai cv2.WINDOW_NORMAL
#cv2.setWindowProperty("sop", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)


# ota kuvia kamerasta
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	#tee NumPy taulukko kameran kuvasta
	image = frame.array

        #muuta kuva mustavalkoiseksi
        image_gra = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
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
        #image_bw = cv2.Canny(image_gray,100,200)

        #dilation
        #image_bw = cv2.dilate(image_bw, None)

        # etsi pelialue ja sen keskipiste countoureilla
        cnt, origin = countour(image,image_bw)

        # pallon liikkuminen
        if type(cnt) is not int:
                #laske pallon uudet koordinaatit
                pointdx = np.cos(pointangle)*pointspeed+pointdx
                pointdy = -(np.sin(pointangle)*pointspeed)+pointdy
                roundedpoint = (origin[0]+int(round(pointdx)), origin[1]+int(round(pointdy)))
                
                # tarkista onko pallo pelialueessa
                inside = cv2.pointPolygonTest(cnt, roundedpoint, False)
                if inside == -1: # pallo ei ole pelialueessa
                                                
                        # palaa pelialueeseen
                        returnangle = -(np.pi-pointangle)
                        outofbounds = 0
                        
                        while True:
                                pointdx = np.cos(returnangle)+pointdx
                                pointdy = -(np.sin(returnangle))+pointdy
                                roundedpoint = (origin[0]+int(round(pointdx)), origin[1]+int(round(pointdy)))
                                inside = cv2.pointPolygonTest(cnt, roundedpoint, False)
                                if inside >= 0:
                                        break
                                if outofbounds > 50: # pallo kaukana pelialueen ulkopuolella
                                        pointdx = 0
                                        pointdy = 0
                                        roundedpoint = (origin[0], origin[1])
                                        break
                                outofbounds = outofbounds + 1
                        if outofbounds <= 50:
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
                                cv2.circle(image, (nearest[0],nearest[1]), 4, (255,255,255), -1) #testaus: piirra lahin kulma
                                cv2.line(image,(nearest[0],nearest[1]),roundedpoint,(0,0,255),2) #testaus: piirra kimpoamisseina
                                wallangle = np.arctan2((pointdy + origin[1])-nearest[1], nearest[0]-(pointdx + origin[0])) # kimpoamisseina kulma
                                # laske uusi pallon suunta
                                pointangle = -(pointangle - wallangle)
                                pointangle = pointangle + wallangle

                # piirra pallo
                cv2.circle(image, roundedpoint, 4, (0,0,255), -1)

## jatka tasta
#        onko edellisen framen keskipiste uudessa countourissa?
       

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


        #target = cv2.bitwise_and(image, image, mask = mask)
       	
        #imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #retval, dst = cv2.threshold(imageGray, 150, 255, cv2.THRESH_BINARY)

        #img2rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
                
        #output = img2rgb+target
        #image.flags.writeable = True
        #image[mask == 255] = [255, 0, 0]        

        # ota varit hsv taso
        if key == ord("v") and rgpflag is False:
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

        
        # ota varit rgb taso
        if key == ord("v") and rgpflag is True:
                setrgp = True
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
                
        if sethsv is True:        
                mask = cv2.inRange(hsv, hsv_min, hsv_max)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.dilate(mask,kernel,iterations = 2)
                image.flags.writeable = True
                image[mask == 255] = [255, 0, 0]
        if setrgp is True:        
                mask = cv2.inRange(image, redBoundLow, redBoundUp)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.dilate(mask,kernel,iterations = 1)
                image.flags.writeable = True
                image[mask == 255] = [255, 0, 0]

        # katso osuuko pallo kateen
        if sethsv is True or setrgp is True:
                if mask[roundedpoint[1], roundedpoint[0]] == 255:
                        print "osui"
                
        #piirra keskinelio
        cv2.line(image,(width/2-10, height/2-10),(width/2-10, height/2+10),(0,255,0),1)
        cv2.line(image,(width/2-10, height/2-10),(width/2+10, height/2-10),(0,255,0),1)
        cv2.line(image,(width/2+10, height/2+10),(width/2+10, height/2-10),(0,255,0),1)
        cv2.line(image,(width/2+10, height/2+10),(width/2-10, height/2+10),(0,255,0),1)

        # piirra keskipiste
        if type(cnt) is not int:
                cv2.circle(image, origin, 2, (255,255,255), -1)

        # lisaa fps kuvaan
        fps = 1/(time.time()-start)
        start = time.time()
        cv2.putText(image, str(int(round(fps))),(5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

        # nayta kuva
	cv2.imshow("sop", image)
	key = cv2.waitKey(1) & 0xFF

        # tyhjenna stream seuraavaa kuvaa varten
	rawCapture.truncate(0)
	# jos painetaan `q` nappia, niin ohjelma loppuu
	if key == ord("q"):
		break
cv2.destroyAllWindows()

# region growing seg
