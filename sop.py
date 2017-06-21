from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import random
from math import sqrt

def prune_1_by_min_distance(cnt):
    MinAvgDist = float('inf')
    delete = 4
    for idx1, i in enumerate(cnt):
        distance_sum = 0
        for idx2, k in enumerate(cnt):
            if not (i == k).all():
                dist = np.linalg.norm(i - k)
                distance_sum += dist

        if distance_sum < MinAvgDist:
            MinAvgDist = distance_sum
            delete = idx1

    cnt = np.delete(cnt, delete, 0)
    return cnt


def get_strongest_corner(cnt, exclude):
    MaxAvgDist = 0
    strongest = 4
    for idx1, i in enumerate(cnt):
        if idx1 in exclude:
            continue

        distance_sum = 0
        for idx2, k in enumerate(cnt):
            if not (i == k).all():
                dist = np.linalg.norm(i - k)
                distance_sum += dist

        if distance_sum > MaxAvgDist:
            MaxAvgDist = distance_sum
            strongest = idx1

    return strongest

def get_weakest_corner(cnt, exclude):
    MinAvgDist = float('inf')
    weakest = 4
    for idx1, i in enumerate(cnt):
        if idx1 in exclude:
            continue

        distance_sum = 0
        for idx2, k in enumerate(cnt):
            if not (i == k).all():
                dist = np.linalg.norm(i - k)
                distance_sum += dist

        if distance_sum < MinAvgDist:
            MinAvgDist = distance_sum
            weakest = idx1

    return weakest



def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return -1, -1

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def point_distance(p1, p2):
    return sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )

def corner_pruning2(cnt):
    if len(cnt) == 5:
        # kaksi heikkoa 3 vahvaa
        weaks = []
        strongs = []

        w = get_weakest_corner(cnt, weaks)
        weaks.append(w)

        w = get_weakest_corner(cnt, weaks)
        weaks.append(w)

        for i in range(0, 5):
            if i not in weaks:
                strongs.append(i)

        # Testataan kaikki vahva-heikko suoraparit

        lines = []

        print weaks
        print len(weaks)
        print strongs
        print len(strongs)
        additional_points = []


        #Lasketaan todennakoisimmin vastainen vahva kulma

        excluded_strongpoint = 0
        max_distance = 0
        for s_p in strongs:
            dist_sum = 0
            for w_p in weaks:
                dist_sum += point_distance(cnt[s_p][0], cnt[w_p][0])

            if max_distance < dist_sum:
                excluded_strongpoint = s_p
                max_distance = dist_sum



        for s_point in strongs:
            if s_point == excluded_strongpoint:
                continue
            for w_point in weaks:
                lines.append((cnt[s_point][0], cnt[w_point][0]))

        for line1 in lines:
            for line2 in lines:
                # tarkistus saman weakpointin varalta
                if (line1[1] == line2[1]).all():
                    print "samat"
                    continue

                result = line_intersection(line1, line2)

                # TODO: automaattinen leveys ja korkeus
                if not((result[0] < 0) or (result[1] < 0) or (result[0] > 320) or (result[1] > 240)):
                    #additional_points.append()
                    ec = np.array([[[result[0], result[1]]]])
                    cnt = np.append(cnt, ec, axis=0)


        #return cnt
        print "cnt ennen"
        print cnt
        while len(cnt) > 4:
            cnt = prune_1_by_min_distance(cnt)

        print "cnt jalkeen"
        print cnt
        return cnt

    if len(cnt) == 6:
        # nelja heikkoa 2 vahvaa
        cnt = prune_1_by_min_distance(cnt)

        cnt = corner_pruning2(cnt)


        return cnt

    return cnt


def corner_angle(a, b, c):
    e1 = -b + a
    e2 = -b + c

    num = np.dot(e1, e2)
    denom = np.linalg.norm(e1) * np.linalg.norm(e2)
    angle = np.arccos(num / denom) * 180 / np.pi

    return angle

def corner_pruning3(cnt):
    if len(cnt) >= 5:
        # kaksi heikkoa 3 vahvaa jos cnt on 5
        # 4 heikkoa 2 vahvaa jos cnt on 6

        if len(cnt) > 6:
            cnt = prune_1_by_min_distance(cnt)
            cnt = corner_pruning3(cnt)
            return cnt

        weaks = []
        strongs = []

        angles = []

        for i in range(0, len(cnt)):
            point2 = i + 1

            if point2 >= len(cnt):
                point2 -= len(cnt)

            a = corner_angle(cnt[i-1][0], cnt[i][0], cnt[point2][0])
            angles.append(a)

        angles_np = np.array(angles)

        k = 2

        if len(cnt) == 5:
            k = 2

        if len(cnt) == 6:
            k = 4

        idx = np.argpartition(angles_np, -k)
        weaks = idx[-k:]

        for i in range(0, len(cnt)):
            if i not in weaks:
                strongs.append(i)

        additional_points = np.empty((0, 1, 2), dtype=int)

        for i in range(0, len(cnt)):
            next_point = i + 1
            next_point2 = i + 2
            next_point3 = i + 3

            if next_point >= len(cnt):
                next_point -= len(cnt)

            if next_point2 >= len(cnt):
                next_point2 -= len(cnt)

            if next_point3 >= len(cnt):
                next_point3 -= len(cnt)

            # Vahvoihin reunoihin ei kannata tehda lisalaskentaa
            if (i in strongs) and (next_point in strongs):
                continue

            # Jos seuraava kulmapiste on "varma" niin ei siihen kannata piirtaa suoraa kun se menee ulos alueelta
            if (i in weaks) and (next_point in strongs):
                continue

            # Ei oteta heikkojen kulmien valille piirrettyja linjoja
            if (i in weaks) and (next_point in weaks):
                if (next_point2 in weaks) and (next_point3 in weaks):
                    continue

            # Vahvoihin reunoihin ei voi tulla lisakulmia joten ei lasketa niita
            if (next_point2 in strongs) and (next_point3 in strongs):
                continue

            # Ei voi vahvaan kulmaan tulla lisakulmaa koska menisi vaan alueelta ulos
            if (next_point2 in strongs) and (next_point3 in weaks):
                continue

            line1 = (cnt[i][0], cnt[next_point][0])
            line2 = (cnt[next_point2][0], cnt[next_point3][0])

            result = line_intersection(line1, line2)

            # TODO: automaattinen leveys ja korkeus
            if not ((result[0] < 0) or (result[1] < 0) or (result[0] > 320) or (result[1] > 240)):
                ec = np.array([[[result[0], result[1]]]])
                additional_points = np.append(additional_points, ec, axis=0)

        cnt = np.append(cnt, additional_points, axis=0)

        sorted_weaks = sorted(weaks, key=int, reverse=True)

        # Karsitaan heikot pois ensin
        for w in sorted_weaks:
            if len(cnt) == 4:
                break
            cnt = np.delete(cnt, w, 0)

        # Viimeinen puhdistus, ei pitaisi aina edes menna tahan asti
        while len(cnt) > 4:
            cnt = prune_1_by_min_distance(cnt)

        return cnt

    return cnt


def corner_pruning(cnt):
    if len(cnt) == 5:  # 5 kulmaa
        print "5 kulmaa"
        MinDist = float('inf')
        delete = 4
        for idx1, i in enumerate(cnt):
            for idx2, k in enumerate(cnt):
                if not (i == k).all():
                    dist = np.linalg.norm(i - k)
                    if dist < MinDist:
                        MinDist = dist
                        delete = idx2
        cnt = np.delete(cnt, delete, 0)

    if len(cnt) > 5:  # yli 5 kulmaa
        print 'yli 5 kulmaa'
        cnt = np.delete(cnt, np.s_[4:], 0)

    return cnt


def segmentation(arg,image_gray):
        if arg == 1: #threshold segmentaatio
                (thresh, image_bw) = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                return image_bw
        elif arg == 2: #canny (kayta 3 == Median Blurring)
                #v = np.median(image_gray)
                #sigma = 0.33
                #lower = int(max(0, (1.0 - sigma) * v))
                #upper = int(min(255, (1.0 + sigma) * v))
                canny = cv2.Canny(image_gray,10,200)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
                #canny = cv2.dilate(canny,kernel,iterations = 1)
                #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                #canny = cv2.erode(canny,kernel,iterations = 1)
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
def contourfindrectangle(image,image_bw, centroidx, centroidy, areafoundfirsttime, areafound):
        centroidx, centroidy, areafoundfirsttime, areafound
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
                return biggest[0],(centroidx,centroidy), centroidx, centroidy, areafoundfirsttime, areafound # pelialue loytyi
        areafound = False
        return 0,(centroidx,centroidy), centroidx, centroidy, areafoundfirsttime, areafound # pelialuetta ei loytynyt

#etsi contour jossa on edellisen framen centroid
def contourThatHasCentroid(image_bw, centroidx, centroidy, areafound):
        centroidx, centroidy, areafound
        contours, hierarchy = cv2.findContours(image_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
            cnt = cv2.convexHull(cnt)
            #tarkista onko edellinen centroid uudessa alueessa
            insidearea = cv2.pointPolygonTest(cnt, (centroidx, centroidy), False) 
            if insidearea == 1:
                    
                cnt = corner_pruning3(cnt)

                cnt = cv2.convexHull(cnt)

                # tama voi muuttua -------------------------------------------------------
                # poista kulmat jos kulmia on yli 4
##                if len(cnt) == 5: # 5 kulmaa
##                        print '5 kulmaa'
##                        MinDist = float('inf')
##                        delete = 4
##                        for idx1, i in enumerate(cnt):
##                                for idx2, k in enumerate(cnt):
##                                        if not (i==k).all():
##                                                dist = np.linalg.norm(i-k)
##                                                if dist < MinDist:
##                                                        MinDist = dist
##                                                        delete = idx2
##                        cnt = np.delete(cnt, delete, 0)
##                        
##                if len(cnt) > 5: # yli 5 kulmaa
##                        print 'yli 5 kulmaa'
##                        cnt = np.delete(cnt, np.s_[4:], 0)
                # ------------------------------------------------------------------------
                
                #laske centroid
                m = cv2.moments(cnt)
                centroidx = int(m['m10']/m['m00'])
                centroidy = int(m['m01']/m['m00'])
                areafound = True
                return cnt , (centroidx,centroidy), centroidx, centroidy, areafound # pelialue loytyi
        areafound = False
        return 0,(centroidx,centroidy), centroidx, centroidy, areafound # pelialuetta ei loytynyt

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

def main():
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

                #ota kuva jos painetaan n
                if key == ord("n"):
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    print timestr
                    cv2.imwrite('/home/pi/kuvat/'+timestr+'.png', image)

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
                #2 == canny (kayta 3 == Median Blurring)
                #3 == tyhja
                #4 == tyhja
                #5 == tyhja
                image_bw = segmentation(2,image_gray);

                #testi kuva
                #cv2.imshow("canny", image_bw)
                #key = cv2.waitKey(1) & 0xFF
                
                #dilation
                #image_bw = cv2.dilate(image_bw, None)
                
                # valitse edellisen framen(tai yleensa aloitus piste) centroid kuvan keskipisteesta, b napilla
                if key == ord("b"):
                    areafoundfirsttime = True
                    centroidx = width/2
                    centroidy = height/2

                # etsi pelialue(=cnt) ja sen keskipiste(=origin)
                if areafoundfirsttime is False:
                        # eka frame
                        cnt, origin, centroidx, centroidy, areafoundfirsttime, areafound = contourfindrectangle(image,image_bw, centroidx, centroidy, areafoundfirsttime, areafound)
                if areafoundfirsttime is True:
                        cnt, origin, centroidx, centroidy, areafound = contourThatHasCentroid(image_bw, centroidx, centroidy, areafound) # jos tiedetaan edellinen centroid
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


if __name__ == "__main__":
        main()
