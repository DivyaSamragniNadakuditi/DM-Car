

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
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the output video clip, e.g., -v out_video.mp4")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# initialize video writer
writer = None

# define the paths to the Stop/Non-Stop Keras deep learning model
MODEL_PATH = "stop_not_stop.model"

# to hide warning message for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# initialize the total number of frames that *consecutively* contain
# stop sign along with threshold required to trigger the sign alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 4		# fast speed-> low, slow speed -> high
STOP_SEC = 0

# initialize is the sign alarm has been triggered
STOP = False

# load the trained CNN model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# Grab the reference to the webcam
vs = VideoStream(src=0).start()

# detect lane based on the last # of frames
frame_buffer = deque(maxlen=args["buffer"])

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
MAX_ANGLE = 20			# Maximum angle to turn right at one time
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
	(h, w) = frame.shape[:2]
	r = 320 / float(w)
	dim = (320, int(h * r))
	frame = cv2.resize(frame, dim, cv2.INTER_AREA)
	# resize to 320 x 180 for wide frame
	frame = frame[0:180, 0:320]
	# crop for CNN model, i.e., traffic sign location
	# can be adjusted based on camera angle
	image = frame[90:135, 260:320]

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
	(notStop, stop) = model.predict(image)[0]
	label = "Not Stop"
	proba = notStop

	# check to see if stop sign was detected using our convolutional
	# neural network
	if stop > notStop:
		# update the label and prediction probability
		label = "Stop"
		proba = stop

		# increment the total number of consecutive frames that
		# contain stop
		if isMoving:
			TOTAL_CONSEC += 1

		# check to see if we should raise the stop sign alarm
		if isMoving and not STOP and TOTAL_CONSEC >= TOTAL_THRESH:
			# indicate that stop has been found
			STOP = True
			bw.stop()
			isMoving = False
			STOP_SEC += 1
			#print("Stop Sign..." + str(STOP_SEC))
		elif STOP and STOP_SEC <= 10:
			bw.stop()
			isMoving = False
			STOP_SEC += 1
			#print("Stop is going on..." + str(STOP_SEC))
		elif STOP and STOP_SEC > 10:
			STOP = False
			bw.speed = SPEED
			bw.forward()
			isMoving = True
			STOP_SEC = 0
			TOTAL_CONSEC = 0
			#print("Stop is done...Going")

	# otherwise, reset the total number of consecutive frames and the
	# stop sign alarm
	else:
		TOTAL_CONSEC = 0
		STOP = False
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
		if ANGLE >= 135:
			ANGLE = 135
		elif ANGLE <= 45:
			ANGLE = 45

		fw.turn(ANGLE)
	
	# Video Writing
	if writer is None:
		if args.get("video", False):
			writer = cv2.VideoWriter(args["video"], 
				0x00000021, 
				15.0, (320,180), True)

	# if a video path is provided, write a video clip
	if args.get("video", False):
		writer.write(blend_frame)

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
