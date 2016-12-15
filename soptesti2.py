import cv2
import numpy as np

imfile = "kuvat/kuva1.jpg"

image = cv2.imread(imfile); 
image_gray = cv2.imread(imfile,cv2.CV_LOAD_IMAGE_GRAYSCALE);

#imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
	



	#output = dst
	# show the frame
	cv2.imshow("Frame", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

