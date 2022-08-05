# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from skimage import exposure
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from Phidget22.Devices.VoltageRatioInput import *
import keyboard
import RPi.GPIO as GPIO

#**********GPIO Assignment for Actuator**********#
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(38,GPIO.OUT) #PWM1
GPIO.setup(40,GPIO.OUT) #PWM2
GPIO.setup(36,GPIO.OUT) #PWM3
soft_pwm1 = GPIO.PWM(38,10)
soft_pwm2 = GPIO.PWM(40,10)
soft_pwm3 = GPIO.PWM(36,10)


#**********CAMERA PRE-PROCESSING Parameters (use get_picam_paramters.py for adjustment)**********#
camera = PiCamera()
frame_width = 640 #1920 #2592 #1280
frame_height = 480 #1080 # 1944 #720
camera.resolution = (frame_width, frame_height)
camera.framerate = 90
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
#plt.style.use('fivethirtyeight')
#x_vals = []
#y_vals = []
index = count() 
#plt.plot([], [], label = 'Load Cell')
x_len = 10 #number of points to display
y_range = [0,80] #range of possible y vals
#create figure for plotting
fig_loadcell = plt.figure() 
ax = fig_loadcell.add_subplot(1,1,1)
xs = list(range(0,10))
ys = [0] * x_len
ax.set_ylim(y_range)
line, = ax.plot(xs,ys)
plt.title('GA Pull Off Force')
plt.xlabel('Samples')
plt.ylabel('Force (N)')
r= [0] #[0.000023778132175] #tare val
output = []#[(0,0,0),(0,0,0)] #[[0,0,0],[0,0,0]]
ch = VoltageRatioInput()
ch.openWaitForAttachment(1000)
ch.setBridgeGain(BridgeGain.BRIDGE_GAIN_128)
ch.setDataInterval(8)

# cameraMatrix = [[471.93662917,0,318.81030519],[0,482.52785357,238.820022],[0,0,1]]
#dist_param = [0.10598021, 0.42904652, -0.01172891, -0.02746931, -0.61650973]
#above is only with 7 images
#cameraMatrix = [[456.43909587,0,315.74349429],[0,467.23961697,242.14258091],[0,0,1]]
#dist_param = [0.09283632, 0.2951744, -0.00903897, -0.02657634, -0.36679998]
#above is for not centered orientation
#cameraMatrix = [[567.12046718,0,309.0439747],[0,584.95315792,230.73561053],[0,0,1]]
#dist_param = [0.06892147,0.21363792,-0.00983466,-0.01510975,-0.29462341]
#above is smaller 10x14
#cameraMatrix = [[541.54736699,0,318.05901744],[0,562.20193836,219.68040138],[0,0,1]]
#dist_param = [0.11350918,0.23187239,-0.0174334,-0.01831228,-0.32450054]
#above is bigger 8x6
cameraMatrix = [[521.48889545,0,329.27954464],[0,532.21046939,237.99577431],[0,0,1]]
dist_param = [0.08646883,0.76458433,-0.00994538,-0.0233169,-1.20937861]
cameraMatrix = np.array(cameraMatrix)
dist_param = np.array(dist_param)

matrix_warp = [[1.10507573e+00,1.43890069e-02,-1.28608943e+01],[ 5.13105986e-02,1.01297053e+00,-5.01999724e+01],[2.60823434e-04,4.32063842e-05,1.00000000e+00]]
matrix_warp = np.array(matrix_warp)
width_warp, height_warp = 580,381

size_frame = (width_warp,height_warp)
file_name_vid = "/media/flexiv-user/LINUXCNC 2_/Actuator Integration/GA_PullOff1_" + str(time.time()) + ".avi" #".h264"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
final_result = cv2.VideoWriter(file_name_vid,fourcc,1,size_frame) # change to 3 for real time
#********************************#

#**********Creating Gui Slider Bars**********#
# def empty(a):
#     pass
# cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters", 420,100)
# #cv2.createTrackbar("clipLimit", "Parameters",1,20,empty)
# #cv2.createTrackbar("tileGrideSize", "Parameters",1,100,empty)
# #cv2.createTrackbar("Min ImgCanny", "Parameters",0,255,empty)
# #cv2.createTrackbar("Max ImgCanny", "Parameters",0,255,empty)
# cv2.createTrackbar("Min ImgGray", "Parameters",0,255,empty)
# #cv2.createTrackbar("Max ImgGray", "Parameters",0,255,empty)
# cv2.createTrackbar("Dilate", "Parameters",0,30,empty)
# cv2.createTrackbar("Erode", "Parameters",0,30,empty)
# cv2.createTrackbar("Gamma", "Parameters",0,20,empty)
# cv2.createTrackbar("Area", "Parameters",1000,40000,empty)

#**********Stack Images**********#
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width_stack = imgArray[0][0].shape[1]
    height_stack = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height_stack, width_stack, 3), np.uint8)
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
        #areaMin = cv2.getTrackbarPos("Area", "Parameters")
        areaMin = 10000
        if area>areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255,0,255), 2)
            #peri = cv2.arcLength(cnt, True)
            #approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            #x,y,w,h = cv2.boundingRect(approx)
            #cv2.rectangle(imgContour, (x-20, y-20), (x + w+20, y + h+20), (0,0,0),3)
            #cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w -240, y + 300), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 3)
            #Above is used to display perimeter of polydp points used for bounding rectangle
            area = np.array(float(area)) 
            #area = area / 147.266013071895 calibration with black 3x3cm square
            final_area = area / 143.441995464853
            cv2.putText(imgContour, "Area(mm2): {:.3f} ".format(float(final_area)), (0,350), cv2.FONT_HERSHEY_COMPLEX, 1,(0,69,255), 2)
            #print(area)
        #else:
            #final_area = 0
            
            return(float(final_area))

#**********Animate function for Video and Load Cell**********#
def animate(i,output_,r_,ch,ys):

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
        key = cv2.waitKey(1) & 0xFF #need this line for sudo command to run with Video
        image_norm = frame.array
        it_1 = 0
        ############## UNDISTORTION #####################################################
        #h,  w = image_norm.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist_param, (frame_width,frame_height), 1, (frame_width,frame_height)) #need roi or get buffer error

        # # Undistort
        image_undist = cv2.undistort(image_norm, cameraMatrix, dist_param, None, newCameraMatrix)
        
        #aspect ratio of 1.52364547
        #Uncommment to get new warp_matrix
        #pts1 = np.float32([[11,49],[623,18],[6,433],[628,463]])
        #pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
        #cv2.circle(image_undist,(int(pts1[0][0]),int(pts1[0][1])),5,(0,255,0),-1)
        #cv2.circle(image_undist,(int(pts1[1][0]),int(pts1[1][1])),5,(0,255,0),-1)
        #cv2.circle(image_undist,(int(pts1[2][0]),int(pts1[2][1])),5,(0,255,0),-1)
        #cv2.circle(image_undist,(int(pts1[3][0]),int(pts1[3][1])),5,(0,255,0),-1)
        #matrix_warp = cv2.getPerspectiveTransform(pts1,pts2)
        #print(matrix_warp)

        image_warp = cv2.warpPerspective(image_undist,matrix_warp,(width_warp,height_warp))
        
        #threshold7 = cv2.getTrackbarPos("Gamma", "Parameters")
        threshold7 = 4
        imgGamma = exposure.adjust_gamma(image_warp,threshold7)
        pix_intensity = np.average(imgGamma) #can add ,axis(0,1) for all channels #can change for bounding rectangle for more accurate but also loss of frames
        #print(pix_intensity)
        #lab = cv2.cvtColor(imgGamma, cv2.COLOR_BGR2LAB)
        #l_channel, a, b = cv2.split(lab)
	# Applying CLAHE to L-channel
	# feel free to try different values for the limit and grid size:
        #clipLimit_thres = cv2.getTrackbarPos("clipLimit", "Parameters")
        #tileGridSize_thres = cv2.getTrackbarPos("tileGrideSize", "Parameters")
        #tileGridSize_thres = 37 #increase to around 100 if desired to see wedges
        #clahe = cv2.createCLAHE(clipLimit=.5, tileGridSize=(tileGridSize_thres,tileGridSize_thres))
        #cl = clahe.apply(l_channel)
	# merge the CLAHE enhanced L-channel with the a and b channel
        #limg = cv2.merge((cl,a,b))
	# Converting image from LAB Color model to BGR color spcae
        #enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        imgContour = imgGamma.copy()
        #threshold6 = cv2.getTrackbarPos("Erode", "Parameters")
        threshold6 = 6
        ker = np.ones((threshold6,threshold6), 'uint8')
        imgErode = cv2.erode(src= imgGamma,kernel=ker,iterations = 1)
        imgGray = cv2.cvtColor(imgErode, cv2.COLOR_BGR2GRAY)
        #threshold3 = cv2.getTrackbarPos("Min ImgGray", "Parameters")
        #threshold4 = cv2.getTrackbarPos("Max ImgGray", "Parameters")
        threshold3 = 30
        threshold4 = 255
        _, imgGray = cv2.threshold(imgGray, threshold3, threshold4, cv2.THRESH_BINARY)
        #threshold1 = cv2.getTrackbarPos("Min ImgCanny", "Parameters")
        #threshold2 = cv2.getTrackbarPos("Max ImgCanny", "Parameters")
        threshold1 = 255
        threshold2 = 255 
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        #threshold5 = cv2.getTrackbarPos("Dilate", "Parameters")
        threshold5 = 6
        kernel = np.ones((threshold5,threshold5), 'uint8')
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        getContours(imgDil, imgContour)
        
        #imgStack = stackImages(.4,([image_norm,image_warp,imgGamma,enhanced_img],[imgErode,imgGray,imgDil,imgContour]))
        #cv2.imshow("Normal   Undist   Gamma   Enhanced   ****   Erode   Gray   Dilate   Contour", imgStack)
    
        cv2.imshow("Contour", imgContour)
        final_result.write(imgContour)
        ##Below is to capture a final picture
        ##filename = 'undist_sq_warp.png'
        ##cv2.imwrite(filename,image_undist)
    
	# clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        rawCapture.seek(0)
        it_1 = 1
        if it_1 > 0:
            break
	# if the `q` key was pressed, break from the loop
        #if keyboard.is_pressed('t'):
        #if key == ord("q"):
            #final_result.release()
            #destroyAllWindows()

        #*****LOAD CELL CODE*****#
    voltageRatio = ch.getVoltageRatio()
    #next_index = next(index)
    force_diff = 0

    if keyboard.is_pressed('t'):
        r_[0] = ch.getVoltageRatio()
        print("Tared")
    new_ratio = (voltageRatio - r_[0]) * 33291104.7583953
    new_ratio_F = new_ratio * .009806650028638
        
    if new_ratio_F == 0:
        GPIO.output(40,0)
        #soft_pwm1.start(100)
        GPIO.output(38,1)
        soft_pwm3.start(10)
    if new_ratio_F >= 10 and new_ratio_F <= 20:
        #soft_pwm3.ChangeFrequency(10)
        soft_pwm3.ChangeDutyCycle(6)
    if new_ratio_F >= 60 and new_ratio_F <= 65:
        soft_pwm3.ChangeFrequency(6)
        soft_pwm3.ChangeDutyCycle(3)

    
    if new_ratio_F >= 8:
        #force_diff = force_list_[-2] - force_list_[-1]
        force_diff = (float(output_[-2][0:4])) - (float(output_[-1][0:4]))
        #print(float(output_[-2][0:4]))
        #print(float(output_[-1][0:4]))
        #print(force_diff)
    if force_diff >= 20:
        GPIO.output(40,0)
        GPIO.output(38,0)

        #ax = plt.gca()
        #line1, = ax.lines 
        #line1.set_data(x_vals,y_vals)
        # xlim_low, xlim_high = ax.get_xlim()
        # ylim_low, ylim_high = ax.get_ylim()
        # ax.set_xlim(xlim_low, (max(x_vals) + 5)) 
        # y_max = max(y_vals) 
        # y_min = min(y_vals) 
        # ax.set_ylim((y_min - 5), (y_max + 5))
    ys.append(new_ratio_F) #add y to list
    ys = ys[-x_len:] #limit y list to set number of items
    line.set_ydata(ys) #update line with new y values
    final_area = getContours(imgDil, imgContour)
    #output_.append([new_ratio_F,final_area])

    output_1 = "{},{},{}".format(new_ratio_F,final_area,pix_intensity)
    #output_1 = (new_ratio_F,final_area)
    output_.append(output_1)

    return line,

#*****************************************#         

#**********Call to Animate Function**********#
ani = FuncAnimation(fig_loadcell, animate, fargs= [output,r,ch,ys],interval=1,blit=True)
plt.show()

#**********Extend Actuator**********#
#soft_pwm1.stop()
soft_pwm3.ChangeFrequency(10)
soft_pwm3.ChangeDutyCycle(100)
GPIO.output(40,1)
GPIO.output(38,0)
#soft_pwm2.start(100)
time.sleep(3)
#soft_pwm2.stop()
soft_pwm3.stop()
GPIO.cleanup()

#**********Output Data to a CSV File**********#
header = ('Force (N)','Area (mm2)','Pixel Intensity (0-255)')
file_name_csv = "/media/flexiv-user/LINUXCNC 2_/Actuator Integration/Force_Area_PI1_" + str(time.time()) + ".csv" #".h264"
#Deleting output[0:2][:] (zero values for actuator actuation to run properly)
#del output[0:2]
with open(file_name_csv, 'w', encoding='UTF8') as f:
    f.write(",".join(header) + "\n")
    for x in output:
        f.write((str(x)) + "\n")
