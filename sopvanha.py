import time
import picamera
import numpy as np

# Create an array representing a 1280x720 image of
# a cross through the center of the display. The shape of
# the array must be of the form (height, width, color)
a = np.zeros((720, 1280, 3), dtype=np.uint8)
a[360, 640, 0] = 0xff


camera = picamera.PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 24
camera.start_preview()
# Add the overlay directly into layer 3 with transparency;
# we can omit the size parameter of add_overlay as the
# size is the same as the camera's resolution
o = camera.add_overlay(np.getbuffer(a), layer=3, alpha=128)
try:
    # Wait indefinitely until the user terminates the script

    time.sleep(5)
finally:
    camera.remove_overlay(o)
    camera.stop_preview()
