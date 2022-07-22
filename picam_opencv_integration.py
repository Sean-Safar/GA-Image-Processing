# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from skimage import exposure
import picamera.array

import os
# initialize the camera and grab a reference to the raw camera capture
#out = cv2.VideoWriter('output.h264',-1,30,(640,480))
camera = PiCamera()
#camera.image_effect = 'denoise'
camera.resolution = (640, 480)
#camera.resolution = (1280,720)
camera.framerate = 30
camera.sharpness = -40
camera.contrast = 22
camera.brightness = 46
camera.saturation = -28
camera.exposure_compensation = 9
#maybe just turn exposure mode off
camera.awb_mode = 'incandescent'
rawCapture = PiRGBArray(camera, size=(640, 480))
#rawCapture = PiRGBArray(camera, size=(1280,720))
# allow the camera to warmup
time.sleep(0.1)

def empty(a):
    pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 420,100)
cv2.createTrackbar("Min ImgCanny", "Parameters",0,255,empty)
cv2.createTrackbar("Max ImgCanny", "Parameters",0,255,empty)
cv2.createTrackbar("Min ImgGray", "Parameters",0,255,empty)
cv2.createTrackbar("Max ImgGray", "Parameters",0,255,empty)
cv2.createTrackbar("Dilate", "Parameters",0,30,empty)
cv2.createTrackbar("Erode", "Parameters",0,30,empty)
cv2.createTrackbar("Gamma", "Parameters",0,20,empty)
cv2.createTrackbar("Area", "Parameters",1000,40000,empty)
#cv2.createTrackbar("Blue Gain", "Parameters",1,99,empty)
#cv2.createTrackbar("Red Gain", "Parameters",1,99,empty)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img,imgContour):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area>areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255,0,255), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x-20, y-20), (x + w+20, y + h+20), (0,0,0),3)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w -240, y + 300), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 3)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w - 240, y+350), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 3)




# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	# show the frame
	#cv2.imshow("Frame", image)

	#imgnorm = image
	threshold7 = cv2.getTrackbarPos("Gamma", "Parameters")
	imgGamma = exposure.adjust_gamma(image,threshold7)
	lab = cv2.cvtColor(imgGamma, cv2.COLOR_BGR2LAB)
	l_channel, a, b = cv2.split(lab)
	# Applying CLAHE to L-channel
	# feel free to try different values for the limit and grid size:
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl = clahe.apply(l_channel)
	# merge the CLAHE enhanced L-channel with the a and b channel
	limg = cv2.merge((cl,a,b))
	# Converting image from LAB Color model to BGR color spcae
	enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	# Stacking the original image with the enhanced image
	#imgnorm = enhanced_img
    #success, imgnorm = cv2.imread(cap)

	#img = imgnorm[98:378,188:438]
	img = enhanced_img
    #imgnorm = exposure.adjust_gamma(img,3)
    #img = exposure.adjust_gamma(imgnorm,3)
	imgContour = img.copy()
	threshold6 = cv2.getTrackbarPos("Erode", "Parameters")
	ker = np.ones((threshold6,threshold6), 'uint8')
	imgErode = cv2.erode(src= img,kernel=ker,iterations = 1)
	#imgBlur = cv2.GaussianBlur(imgErode, (7, 7), 1)
	imgGray = cv2.cvtColor(imgErode, cv2.COLOR_BGR2GRAY)
	# imgGray = exposure.adjust_gamma(imgGray,1)
	threshold3 = cv2.getTrackbarPos("Min ImgGray", "Parameters")
	threshold4 = cv2.getTrackbarPos("Max ImgGray", "Parameters")
	_, imgGray = cv2.threshold(imgGray, threshold3, threshold4, cv2.THRESH_BINARY)
	threshold1 = cv2.getTrackbarPos("Min ImgCanny", "Parameters")
	threshold2 = cv2.getTrackbarPos("Max ImgCanny", "Parameters")
	imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
	threshold5 = cv2.getTrackbarPos("Dilate", "Parameters")
	kernel = np.ones((threshold5,threshold5), 'uint8')
	imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
	getContours(imgDil, imgContour)
	imgStack = stackImages(0.3,([image,enhanced_img,imgGamma,imgGray],[imgErode, imgCanny, imgDil, imgContour]))
	cv2.imshow("Normal   Enhanced   Gamma   Gray **** Erode   Canny   Dilate   Contour", imgStack)
	#out.write(imgStack)
	
	#RECORDING
	#file_name = "/media/flexiv-user/LINUXCNC_2_/Preload Video/11th_tephlon_oldblack" + str(time.time()) + ".h264"
	#print("Start recording...")
	#imStack = np.array(imgStack)	
	#imgContour.start_recording(file_name)


	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		#imgStack.stop_recording()
		#print("Done")
		break