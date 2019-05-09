# Auto Streering for straight lane with Stop-Sign detector
# Date: Feb 23, 2019
# Jeongkyu Lee

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from picar import back_wheels, front_wheels
import picar
from Line import Line
from lane_detection import color_frame_pipeline
from lane_detection import PID
import time
import math
import os

# construct the argument parse and parse the arguments


# initialize video writer
writer = None

# define the paths to the Stop/Non-Stop Keras deep learning model
MODEL_PATH = "all_signs_model8.model"

# to hide warning message for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# initialize the total number of frames that *consecutively* contain
# stop sign along with threshold required to trigger the sign alarm
TOTAL_CONSEC_STOP = 0
TOTAL_CONSEC_SIGNAL =0
TOTAL_CONSEC_SPEED55=0
TOTAL_CONSEC_SPEED35=0
TOTAL_CONSEC_RAILROAD=0
TOTAL_CONSEC_YIELD=0
TOTAL_THRESH = 4		# fast speed-> low, slow speed -> high
STOP_SEC = 0

# initialize is the sign alarm has been triggered
STOP = False
SPEED55= False
SPEED35 = False
SIGNAL = False
RAILROAD = False
YIELD = False

# load the trained CNN model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# Grab the reference to the webcam
vs = VideoStream(src=0).start()

# detect lane based on the last # of frames
frame_buffer = deque(maxlen=5)

# allow the camera or video file to warm up
time.sleep(2.0)

picar.setup()
db_file = "/home/pi/dmcar-student/picar/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)

bw.ready()
fw.ready()

SPEED = 0
ANGLE = 90			# steering wheel angle: 90 -> straight 
MAX_ANGLE = 60			# Maximum angle to turn right at one time
MIN_ANGLE = -MAX_ANGLE		# Maximum angle to turn left at one time
isMoving = False		# True: car is moving
posError = []			# difference between middle and car position
bw.speed = SPEED		# car speed
fw.turn(90)			# steering wheel angle

# keep looping
while True:
    # grab the current frame
    frame = vs.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=320)
    cv2.rectangle(frame, (240,60),(320,120),(0,255,0),5)
    (h, w) = frame.shape[:2]
    r = 320 / float(w)
    dim = (320, int(h * r))
    frame = cv2.resize(frame, dim, cv2.INTER_AREA)
    # resize to 320 x 180 for wide frame
    frame = frame[0:180, 0:320]
    # crop for CNN model, i.e., traffic sign location
    # can be adjusted based on camera angle
    image = frame[60:120, 240:320]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_buffer.append(frame)
    blend_frame, lane_lines = color_frame_pipeline(frames=frame_buffer, \
                       solid_lines=True, \
                       temporal_smoothing=True)

    # prepare the image to be classified by our deep learning network
    image = cv2.resize(image , (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image and initialize the label and
    # probability of the prediction
    noSign, stop, speed55, speed35, signal, railroad, yield1 = model.predict(image)[0]
    label = "Not Sign"
    proba = noSign
    noSign = 0

    #Checking STOP sign
    # check to see if stop sign was detected using our convolutional
    # neural network
    if max(noSign, stop, speed55, speed35, signal, railroad, yield1) == stop and stop > 0.8:
        # update the label and prediction probability
        label = "Stop"
        proba = stop
        noSign = 1

        # increment the total number of consecutive frames that
        # contain stop
        if isMoving:
            TOTAL_CONSEC_STOP += 1

        # check to see if we should raise the stop sign alarm
        if isMoving and not STOP and TOTAL_CONSEC_STOP >= TOTAL_THRESH:
            # indicate that stop has been found
            STOP = True
            SPEED55= False
            SPEED35 = False
            SIGNAL = False
            RAILROAD = False
            YIELD = False
            bw.stop()
            isMoving = False
            STOP_SEC += 1
            print("Stop Sign..." + str(STOP_SEC))
        elif STOP and STOP_SEC <= 10:
            bw.stop()
            isMoving = False
            STOP_SEC += 1
            print("Stop is going on..." + str(STOP_SEC))
        elif STOP and STOP_SEC > 10:
            STOP = False
            RAILROAD = False
            SPEED55= False
            SPEED35 = False
            SIGNAL = False            
            YIELD = False
            bw.speed = SPEED
            bw.forward()
            isMoving = True
            STOP_SEC = 0
            TOTAL_CONSEC_STOP = 0
            print("Stop is done...Going")
    
    #Checking SPEED55 sign
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == speed55 and speed55 > 0.8:
        # update the label and prediction probability
        label = "Speed 55"
        proba = speed55
        
        # increment the total number of consecutive frames that
        # contain stop
        if isMoving:
            TOTAL_CONSEC_SPEED55 += 1

        # check to see if we should raise the stop sign alarm
        if isMoving and not SPEED55 and TOTAL_CONSEC_SPEED55 >= TOTAL_THRESH:
            # indicate that stop has been found
            SPEED55 = True
            STOP= False
            SPEED35 = False
            SIGNAL = False
            RAILROAD = False
            YIELD = False
            bw.speed = 55
            bw.forward()
            isMoving = True
            STOP_SEC = 0
            TOTAL_CONSEC_SPEED55 = 0
            print("Speed 55 " + str(proba))
           
    #Checking SPEED35 sign
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == speed35 and speed35 > 0.8:
        # update the label and prediction probability
        label = "Speed 35"
        proba = speed35
        
        # increment the total number of consecutive frames that
        # contain stop
        if isMoving:
            TOTAL_CONSEC_SPEED35 += 1

        # check to see if we should raise the stop sign alarm
        if isMoving and not SPEED35 and TOTAL_CONSEC_SPEED35 >= TOTAL_THRESH:
            # indicate that stop has been found
            SPEED35 = True
            STOP= False
            SPEED55 = False
            SIGNAL = False
            RAILROAD = False
            YIELD = False
            bw.speed = 35
            bw.forward()
            isMoving = True
            STOP_SEC = 0
            TOTAL_CONSEC_SPEED35 = 0
            print("Speed35 " + str(proba))
    
    #Checking SIGNAL sign
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == signal and signal > 0.8:
        # update the label and prediction probability
        label = "Signal"
        proba = signal
        
        # increment the total number of consecutive frames that
        # contain stop
        if isMoving:
            TOTAL_CONSEC_SIGNAL += 1

        # check to see if we should raise the stop sign alarm
        if isMoving and not SIGNAL and TOTAL_CONSEC_SIGNAL >= TOTAL_THRESH:
            # indicate that stop has been found
            SIGNAL = True
            STOP= False
            SPEED55 = False
            SPEED55 = False
            RAILROAD = False
            YIELD = False
            bw.speed = SPEED
            bw.forward()
            isMoving = True
            STOP_SEC = 0
            TOTAL_CONSEC_SIGNAL = 0
            print("Signal " + str(proba))
    
    #Checking RAILROAD sign
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == railroad > 0.8:
        # update the label and prediction probability
        label = "Railroad"
        proba = railroad

        # increment the total number of consecutive frames that
        # contain stop
        if isMoving:
            TOTAL_CONSEC_RAILROAD += 1

        # check to see if we should raise the stop sign alarm
        if isMoving and not RAILROAD and TOTAL_CONSEC_RAILROAD >= TOTAL_THRESH:
            # indicate that stop has been found
            RAILROAD = True
            STOP = False
            SPEED55= False
            SPEED35 = False
            SIGNAL = False            
            YIELD = False
            bw.stop()
            isMoving = False
            STOP_SEC += 1
            #print("Stop Sign..." + str(STOP_SEC))
            print("Railroad " + str(proba))
        elif RAILROAD and STOP_SEC <= 10:
            bw.stop()
            isMoving = False
            STOP_SEC += 1
            #print("Stop is going on..." + str(STOP_SEC))
        elif RAILROAD and STOP_SEC > 10:
            RAILROAD = False
            STOP = False
            SPEED55= False
            SPEED35 = False
            SIGNAL = False            
            YIELD = False
            bw.speed = SPEED
            bw.forward()
            isMoving = True
            STOP_SEC = 0
            TOTAL_CONSEC_RAILROAD = 0
            #print("Stop is done...Going")
    
    #Checking YIELD sign
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == yield1 and yield1 > 0.8:
        # update the label and prediction probability
        label = "Yield"
        proba = yield1

        # increment the total number of consecutive frames that
        # contain stop
        if isMoving:
            TOTAL_CONSEC_YIELD += 1

        # check to see if we should raise the stop sign alarm
        if isMoving and not YIELD and TOTAL_CONSEC_YIELD >= TOTAL_THRESH:
            # indicate that stop has been found
            YIELD = True
            STOP = False
            SPEED55= False
            SPEED35 = False
            SIGNAL = False            
            RAILROAD = False
            bw.speed = 10
            isMoving = False
            STOP_SEC += 1
            #print("Stop Sign..." + str(STOP_SEC))
            print("Yield " + str(proba))
        elif YIELD and STOP_SEC > 10:
            YIELD = False
            STOP = False
            SPEED55= False
            SPEED35 = False
            SIGNAL = False            
            RAILROAD = False
            bw.speed = SPEED
            bw.forward()
            isMoving = True
            STOP_SEC = 0
            TOTAL_CONSEC = 0
            #print("Stop is done...Going")

    # otherwise, reset the total number of consecutive frames and the
    # stop sign alarm
    else:
        TOTAL_CONSEC_STOP = 0
        TOTAL_CONSEC_SIGNAL =0
        TOTAL_CONSEC_SPEED55=0
        TOTAL_CONSEC_SPEED35=0
        TOTAL_CONSEC_RAILROAD=0
        TOTAL_CONSEC_YIELD=0
        STOP = False
        SPEED55= False
        SPEED35 = False
        SIGNAL = False
        RAILROAD = False
        YIELD = False
        label="No Sign"
        bw.forward()
        isMoving = True
        STOP_SEC = 0
        #print("No more Stop... Going")

    # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    blend_frame = cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)
    blend_frame = cv2.putText(blend_frame, label, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('blend', blend_frame)

    #for mid-position of lanes
    y2L = h - 1
    x2L = int((y2L - lane_lines[0].bias) / lane_lines[0].slope)
    y2R = h - 1
    x2R = int((y2R - lane_lines[1].bias) / lane_lines[1].slope)
    mid_position_lane = ( x2R + x2L ) / 2

    SPEED = 30

    if isMoving:
        # negative -> + ANGLE, positive -> - ANGLE
        car_position_err = w/2 - mid_position_lane
        car_position_time = time.time()
        posError.append([car_position_err, car_position_time])
        
        # Control Car
        # Adjust P(KP), I(KI), D(KD) values as well as portion
        # angle = PID(posError, KP=0.8, KI=0.05, KD=0.1) * 0.2
        angle = PID(posError, KP=0.8, KI=0.1, KD=0.1) * 0.25
        # print(angle)

        # MAX + - 20 degree	
        if angle > MAX_ANGLE:
            angle = MAX_ANGLE
        elif angle < MIN_ANGLE:
            angle = MIN_ANGLE 

        ANGLE = 90 - angle 

        # Right turn max 135, Left turn max 45
        if ANGLE >= 145:
            ANGLE = 145
        elif ANGLE <= 60:
            ANGLE = 60
        if ANGLE > 88 and ANGLE < 92:
            ANGLE = 90
	
	
        #print(ANGLE)

        fw.turn(ANGLE)

    # Video Writing


    keyin = cv2.waitKey(1) & 0xFF
    keycmd = chr(keyin)

    # if the 'q' key is pressed, end program
    # if the 'w' key is pressed, moving forward
    # if the 'x' key is pressed, moving backword
    # if the 'a' key is pressed, turn left
    # if the 'd' key is pressed, turn right
    # if the 's' key is pressed, straight
    # if the 'z' key is pressed, stop a car
    if keycmd == 'q':
        break
    elif keycmd == 'w':
        isMoving = True
        bw.speed = SPEED
        bw.forward()
    elif keycmd == 'x':
        bw.speed = SPEED
        bw.backward()
    elif keycmd == 'a':
        ANGLE -= 5
        if ANGLE <= 45:
            ANGLE = 45
        #fw.turn_left()
        fw.turn(ANGLE)
    elif keycmd == 'd':
        ANGLE += 5
        if ANGLE >= 135:
            ANGLE = 135
        #fw.turn_right()
        fw.turn(ANGLE)
    elif keycmd == 's':
        ANGLE = 90
        #fw.turn_straight()
        fw.turn(ANGLE)
    elif keycmd == 'z':
        isMoving = False
        bw.stop()

# if we are not using a video file, stop the camera video stream
writer.release()
vs.stop()

# close all windows
cv2.destroyAllWindows()
