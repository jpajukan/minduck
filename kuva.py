import time
import picamera
import cv2

with picamera.PiCamera() as camera:
    camera.resolution = (320, 240)
    camera.start_preview()
    # Camera warm-up time
    time.sleep(2)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print timestr
    key = cv2.waitKey(0) & 0xFF
    camera.capture('/home/pi/kuvat/'+timestr+'.png', format='png')
