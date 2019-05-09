#Run the inference 
# USAGE
# python stop_detector.py 

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os

# define the paths to the Not Santa Keras deep learning model and
# audio file
MODEL_PATH = "all_signs_model8.model"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# initialize the total number of frames that *consecutively* contain
# stop sign along with threshold required to trigger the sign alarm
TOTAL_CONSEC_STOP = 0
TOTAL_CONSEC_SIGNAL =0
TOTAL_CONSEC_SPEED55=0
TOTAL_CONSEC_SPEED35=0
TOTAL_CONSEC_RAILROAD=0
TOTAL_CONSEC_YIELD=0
TOTAL_THRESH = 20

# initialize is the sign alarm has been triggered
STOP = False
SPEED55= False
SPEED35 = False
SIGNAL = False
RAILROAD = False
YIELD = False

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 320 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=320)
    cv2.rectangle(frame, (240,60),(320,120),(0,255,0),5)

	# prepare the image to be classified by our deep learning network
    image = frame[60:120, 240:320]
    image = cv2.resize(image , (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

	# classify the input image and initialize the label and
	# probability of the prediction
    noSign, stop, speed55, speed35, signal, railroad, yield1 = model.predict(image)[0]
    label = "No Sign"
    proba = noSign
    noSign = 0

	# check to see if stop sign was detected using our convolutional
	# neural network
    if max(noSign, stop, speed55, speed35, signal, railroad, yield1) == stop and stop > 0.8:
		# update the label and prediction probability
        label = "stop"
        proba = stop
        noSign = 1
        
        
		# increment the total number of consecutive frames that
		# contain stop
        TOTAL_CONSEC_STOP += 1

		# check to see if we should raise the stop sign alarm
        if not STOP and TOTAL_CONSEC_STOP >= TOTAL_THRESH:
			# indicate that stop has been found
            STOP = True
            SIGNAL = False
            SPEED55 = False
            SPEED35 = False
            RAILROAD = False
            YIELD = False
            print("Stop " + str(proba))
    
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == speed55 and speed55 > 0.8:
		# update the label and prediction probability
        label = "Speed 55"
        proba = speed55

		# increment the total number of consecutive frames that
		# contain stop
        TOTAL_CONSEC_SPEED55 += 1

		# check to see if we should raise the stop sign alarm
        if not SPEED55 and TOTAL_CONSEC_SPEED55 >= TOTAL_THRESH:
			# indicate that stop has been found
            SPEED55 = True
            STOP = False
            SIGNAL = False
            SPEED35 = False
            RAILROAD = False
            YIELD = False
            print("Speed 55 " + str(proba))
            
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == speed35 and speed35 > 0.8:
		# update the label and prediction probability
        label = "Speed 35"
        proba = speed35
        noSign = 1
        
        
		# increment the total number of consecutive frames that
		# contain stop
        TOTAL_CONSEC_SPEED35 += 1

		# check to see if we should raise the stop sign alarm
        if not SPEED35 and TOTAL_CONSEC_SPEED35 >= TOTAL_THRESH:
			# indicate that stop has been found
            SPEED35 = True
            STOP = False
            SIGNAL = False
            SPEED55 = False
            RAILROAD = False
            YIELD = False
            print("Speed35 " + str(proba))
    
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == signal and signal > 0.8:
		# update the label and prediction probability
        label = "Signal"
        proba = signal
        noSign = 1

		# increment the total number of consecutive frames that
		# contain stop
        TOTAL_CONSEC_SIGNAL += 1

		# check to see if we should raise the stop sign alarm
        if not SIGNAL and TOTAL_CONSEC_SIGNAL >= TOTAL_THRESH:
			# indicate that stop has been found
            SIGNAL = True
            STOP = False
            SPEED55= False
            SPEED35 = False
            RAILROAD = False
            YIELD = False
            print("Signal " + str(proba))
            
            
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == railroad and railroad > 0.8:
		# update the label and prediction probability
        label = "railroad"
        proba = railroad
        noSign = 1
        
        
		# increment the total number of consecutive frames that
		# contain stop
        TOTAL_CONSEC_RAILROAD += 1

		# check to see if we should raise the stop sign alarm
        if not RAILROAD and TOTAL_CONSEC_RAILROAD >= TOTAL_THRESH:
			# indicate that stop has been found
            RAILROAD = True
            STOP = False
            SIGNAL = False
            SPEED55 = False
            SPEED35 = False
            YIELD = False
            print("Railroaaad " + str(proba))
            
    elif max(noSign, stop, speed55, speed35, signal, railroad, yield1) == yield1 and yield1 > 0.8:
		# update the label and prediction probability
        label = "yield"
        proba = yield1
        noSign = 1
        
        
		# increment the total number of consecutive frames that
		# contain stop
        TOTAL_CONSEC_YIELD += 1

		# check to see if we should raise the stop sign alarm
        if not YIELD and TOTAL_CONSEC_YIELD >= TOTAL_THRESH:
			# indicate that stop has been found
            YIELD = True
            STOP = False
            SIGNAL = False
            SPEED55 = False
            SPEED35 = False
            RAILROAD = False
            print("Yield " + str(proba))
            
 
	# otherwise, reset the total number of consecutive frames and the
	# stop sign alarm
    else:
        TOTAL_CONSEC_STOP = 0
        TOTAL_CONSEC_SPEED55=0
        TOTAL_CONSEC_SIGNAL =0
        TOTAL_CONSEC_SPEED35=0
        TOTAL_CONSEC_RAILROAD=0
        TOTAL_CONSEC_YIELD=0
        STOP = False
        SPEED55 = False
        SPEED35 = False
        SIGNAL = False
        RAILROAD = False
        YIELD = False
        label="No Sign"

	# build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()