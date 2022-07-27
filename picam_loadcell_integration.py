# import the necessary packages
from cv2 import destroyAllWindows
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from skimage import exposure
import picamera.array
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Phidget22.Devices.VoltageRatioInput import *
import keyboard

#**********CAMERA PRE-PROCESSING Parameters (use get_picam_paramters.py for adjustment)**********#
camera = PiCamera()
frame_width = 640 #1920 #2592 #1280
frame_height = 480 #1080 # 1944 #720
camera.resolution = (frame_width, frame_height)
camera.framerate = 30
camera.sharpness = -40
camera.contrast = 22
camera.brightness = 46
camera.saturation = -28
camera.exposure_compensation = 9
camera.awb_mode = 'off'
gain_r = 1.7
gain_b = 1.2 
gAwb = (gain_r, gain_b)
camera.awb_gains = gAwb
rawCapture = PiRGBArray(camera, size=(640, 480))
#rawCapture = PiRGBArray(camera, size=(1920,1080))

# allow the camera to warmup
time.sleep(1)

#**********Creating Global Variables**********#
plt.style.use('fivethirtyeight')
x_vals = []
y_vals = []
index = count()
plt.plot([], [], label = 'Load Cell')
r= [0] #[0.000023778132175]

size_frame = (frame_width,frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
final_result = cv2.VideoWriter('Vid_test_new1.avi',fourcc,1,size_frame)

output = []
it_1 = 1
ch = VoltageRatioInput()
ch.openWaitForAttachment(1000)

Camera_Calibrated = 1.0651237952302004
cameraMatrix = [[471.93662917,0,318.81030519],[0,482.52785357,238.820022],[0,0,1]]
dist_param = [0.10598021, 0.42904652, -0.01172891, -0.02746931, -0.61650973]
cameraMatrix = np.array(cameraMatrix)
dist_param = np.array(dist_param)
#********************************#

#**********Creating Gui Slider Bars**********#
def empty(a):
    pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 420,100)
#cv2.createTrackbar("clipLimit", "Parameters",1,20,empty)
#cv2.createTrackbar("tileGrideSize", "Parameters",1,100,empty)
#cv2.createTrackbar("Min ImgCanny", "Parameters",0,255,empty)
#cv2.createTrackbar("Max ImgCanny", "Parameters",0,255,empty)
cv2.createTrackbar("Min ImgGray", "Parameters",0,255,empty)
#cv2.createTrackbar("Max ImgGray", "Parameters",0,255,empty)
cv2.createTrackbar("Dilate", "Parameters",0,30,empty)
cv2.createTrackbar("Erode", "Parameters",0,30,empty)
cv2.createTrackbar("Gamma", "Parameters",0,20,empty)
cv2.createTrackbar("Area", "Parameters",1000,40000,empty)

#**********Stack Images**********#
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
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
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

#**********Get Contour Function**********#
def getContours(img,imgContour):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #Change RETR for different contours

    for cnt in contours:
        area = cv2.contourArea(cnt) #Greens theorem
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area>areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255,0,255), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x-20, y-20), (x + w+20, y + h+20), (0,0,0),3)
            #cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w -240, y + 300), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 3)
            #Above is used to display perimeter of polydp points used for bounding rectangle
            area = area / 4321.66667
            cv2.putText(imgContour, "Area(mm): " + str(float(area)), (x + w - 240, y+350), cv2.FONT_HERSHEY_COMPLEX, 1.5,(0,69,255), 3)


#**********Animate function for Video and Load Cell**********#
def animate(i,output_,r_,it_1,ch):

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
        key = cv2.waitKey(1) & 0xFF #need this line for sudo command to run with Video
        image_norm = frame.array
        ############## UNDISTORTION #####################################################
        h,  w = image_norm.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist_param, (w,h), 1, (w,h))

        # # Undistort
        image_undist = cv2.undistort(image_norm, cameraMatrix, dist_param, None, newCameraMatrix)
        #image_undist = image_undist[0:640,0:480]
        threshold7 = cv2.getTrackbarPos("Gamma", "Parameters")
        imgGamma = exposure.adjust_gamma(image_undist,threshold7)
        lab = cv2.cvtColor(imgGamma, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
	# Applying CLAHE to L-channel
	# feel free to try different values for the limit and grid size:
        #clipLimit_thres = cv2.getTrackbarPos("clipLimit", "Parameters")
        #tileGridSize_thres = cv2.getTrackbarPos("tileGrideSize", "Parameters")
        tileGridSize_thres = 37 #increase to around 100 if desired to see wedges
        clahe = cv2.createCLAHE(clipLimit=.5, tileGridSize=(tileGridSize_thres,tileGridSize_thres))
        cl = clahe.apply(l_channel)
	# merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))
	# Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        imgContour = enhanced_img.copy()
        threshold6 = cv2.getTrackbarPos("Erode", "Parameters")
        ker = np.ones((threshold6,threshold6), 'uint8')
        imgErode = cv2.erode(src= enhanced_img,kernel=ker,iterations = 1)
        imgGray = cv2.cvtColor(imgErode, cv2.COLOR_BGR2GRAY)
        threshold3 = cv2.getTrackbarPos("Min ImgGray", "Parameters")
        #threshold4 = cv2.getTrackbarPos("Max ImgGray", "Parameters")
        threshold4 = 255
        _, imgGray = cv2.threshold(imgGray, threshold3, threshold4, cv2.THRESH_BINARY)
        #threshold1 = cv2.getTrackbarPos("Min ImgCanny", "Parameters")
        #threshold2 = cv2.getTrackbarPos("Max ImgCanny", "Parameters")
        threshold1 = 255
        threshold2 = 255 
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        threshold5 = cv2.getTrackbarPos("Dilate", "Parameters")
        kernel = np.ones((threshold5,threshold5), 'uint8')
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        getContours(imgDil, imgContour)
        
        imgStack = stackImages(.3,([image_norm,image_undist],[imgGamma,enhanced_img],[imgErode,imgGray],[imgDil, imgContour]))
        cv2.imshow("Normal   Undist   Gamma   Enhanced   ****   Erode   Gray   Dilate   Contour", imgStack)
        #cv2.imshow("Contour", imgContour)
        final_result.write(imgContour)
    
	# clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        rawCapture.seek(0)
	# if the `q` key was pressed, break from the loop
        #if keyboard.is_pressed('t'):
        #if key == ord("q"):
            #final_result.release()
            #destroyAllWindows()

        #*****LOAD CELL CODE*****#
        voltageRatio = ch.getVoltageRatio()
        next_index = next(index)
        x_vals.append(next_index)
        # y_vals.append(random.randint(0,5))
        if keyboard.is_pressed('t'):
            print("Tared")
            r_[0] = ch.getVoltageRatio()
        new_ratio = (voltageRatio - r_[0]) * 33291104.7583953
        new_ratio_F = new_ratio * .009806650028638
        y_vals.append(new_ratio_F)
        ax = plt.gca()
        line1, = ax.lines 
        line1.set_data(x_vals,y_vals)
        xlim_low, xlim_high = ax.get_xlim()
        ylim_low, ylim_high = ax.get_ylim()
        ax.set_xlim(xlim_low, (max(x_vals) + 5)) 
        y_max = max(y_vals) 
        y_min = min(y_vals) 
        ax.set_ylim((y_min - 5), (y_max + 5))
        output_.append(new_ratio_F)
        it_1 += 1
        if it_1 > 1:
            break
#*****************************************#         

#**********Call to Animate Function**********#
ani = FuncAnimation(plt.gcf(), animate, fargs= [output, r, it_1, ch], interval=1)
plt.show()

#print(r)
#**********Output Data to a CSV File**********#
header = ['Test','whatever']
with open('testing_new.csv', 'w', encoding='UTF8') as f:
    f.write(",".join(header) + "\n")
    for x in output:
        f.write((str(x)) + "\n")
