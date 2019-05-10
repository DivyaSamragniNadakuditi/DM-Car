![car-image](https://github.com/DivyaSamragniNadakuditi/DM-Car/blob/master/Images/Car-Image.png)


# Autonomous Vehicle using Data Mining Technique: Deep Learning
This Repository contains the setting up of an autonomous raspberry pi-car and training it with data mining models.

# Group 1

## Group Members:
  1. Divya Samragni Nadakuditi
  2. Nitish Soman
  3. Shiva Shankar Ganesan
  4. Amr Attia
  
## Introduction:
In this project, we will be working on the implementation of data mining Techniques- Convolutional Neural Networks(CNN) for real-world problem Autonomous vehicle. For this project, we used the Smart Video Car Kit V2.0 for Raspberry Pi. We will Assemble the car, Set up Raspberry Pi, Train a Model for detecting traffic signs and act accordingly.

## Phase 1: Assembling the  Autonomous Car
SunFounder PiCar-V for Raspberry Pi 3/2/B+

In this stage, we assemble the Smart Video Car for RaspberryPi. We attach fix therear wheels,Upper plate, Battery Holder, Motor Driver, PWM Driver,Robot HATS, Raspberry Pi. Then we build the Circuits.Ater Configuring the Servos(Details mentioned in the Second part), we fix the front wheels and Camera.  Below is the link of GoogleDrive where the images of assembling the car are present.

Assembling Images:
  - Image 1: https://drive.google.com/open?id=1r8d-2CXZPx8imKsb3EGqVvExqdcq32A3
  - Image 2: https://drive.google.com/open?id=1r0HnyBP826RgsnNldvYAd-n5SXQixBBB
  - mage 3: https://drive.google.com/open?id=1G631TKRQsQvd81LVjVBBTpjKsPhmhYhw
  - Image 4: https://drive.google.com/open?id=12L_TfzryScfRlTGc3KfgPZkIn2xKjw1o
  - Image 5: https://drive.google.com/open?id=1JA3jSyyWul4XOrs5HLvjhD0wanJOWCcN
  - Image 6: https://drive.google.com/open?id=1V-ZxzBHeCvuZDKvATOigjgAQ-wSL798E
  - Image 7: https://drive.google.com/open?id=1Ph_FcsrJVG4Len6V5LPcctoVV78qP6hH
  - Image 8: https://drive.google.com/open?id=1YSXAkY4WXzOJj0wjS5uYMim6Uz-I860p

### Car Improvement suggestions:

1. HDMI Port placement can be changed in the car so that it will be easy to attach the cable
2. Better build quality of car
3. Addition of sensors to detect objects
4. Better camera quality
5. The parts of the car are open. This is not good for the safety of the car. It would be better if we can safe gaurd them with a       covering.
6. Camera position can be improved so that we can visualize things more clearly. As in the present position, the camera movement is not that flexible.

## Phase 2: SettingUp and Configuration Of RaspberryPi

We Download Raspbian Image and write it into an SD Card. Later, we edit the wpa configuration file and set the network details. We further use VNC Viewer to connect to the Raspberrypi. The Images of the setting up and Configuration steps are in the link below.

### SettingUp and Configuration Of RaspberryPi Images:

  - Image 1: https://drive.google.com/open?id=1YwErIctL_KvqnlPgY1x6-kXXmzA8L83Z
  - Image 2: https://drive.google.com/open?id=1lwREGghlnn-gaFcf52oveYKyI5b-Q_CP
  - Image 3: https://drive.google.com/open?id=1_6KHYkRYCx_NCKOw_C__Oi8dDsI0sr94
  - Image 4: https://drive.google.com/open?id=1Wj5G3rMENWfm2uRo7AnhsTjVjUePHxBe
  - Image 5: https://drive.google.com/open?id=1sBUN2I8joG4Yo9oHywXWTc8cpyEkzcLJ
  - Image 6: https://drive.google.com/open?id=148yAJqAYnK843zDdHnheA1jRswEieW1g
  - Image 7: https://drive.google.com/open?id=19fTJnMLgC2FV3zK50KWslGEEBzrQRO7l
  - Image 8: https://drive.google.com/open?id=1wdy7uJVFn-21Xpho3CPi4NoDBiWICBx-
  - Image 9: https://drive.google.com/open?id=1cPbhGeS5HvxOKn54OJho9f6wNio3jJfo
  - Image 10:https://drive.google.com/open?id=1B2-_0AkgcORT77ADU3T4zFefmPpqLup2
  

### Improve the connectivity of picar
It should have connected automatically to any open wifi network initially.


## Phase 3:

### How to improve the lane detection
- If we are able to convert the frame to grayscale then we will get better results.
- We need to create marks for yellow and white pixels.
- Using a Canny edge detection technique.

### How to improve the controlling front wheels and back wheels motors (i.e., servos)
- Most of the times car going towards to left-hand side, so we need to adjust the alignment in front wheels.
- We need to calibrate both the wheels so that it will go straight.

Lane Detection Video: https://youtu.be/crSBcHiNZTg

## Phase 4:

### Questions:
  1. Image Size<br/>
  Currently, we are using 28 * 28 image size, but if we increase it then it will increase the performance of the nureal network.
  
  2. How to design CNN architecture including how many layers, what kind of layers, and so on<br/>
  Convolutional Neural Networks are very similar to ordinary Neural Networks. They are made up of neurons that have learnable weights     and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole        network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And  they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for   learning regular Neural Networks still apply.  
   A simple ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a        differentiable function. We use three main types of layers to build ConvNet architectures: Convolutional Layer, Pooling Layer, and   Fully-Connected Layer (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet architecture.
  
  3. How to optimize the model including parameter values, drop out, backpropagation, learning rate, # of epoch and so on<br />
  Learning rate completely depends on epoch value.
  
  
  4. Evaluations<br />
  The evaluation of the Neural Network completely depends on quality dataset so that when you trained the model you will get less jitter   or loss and achieve more accuracy level.
  
  5. How to overcome the limitations in your DM-Car implementation<br/ >
  To train the model using convolution Neural network requires high processing power and good GPU so that it will perform well and you     will get better results.
  
  Trained model sign detection images:
  - Stop Sign: https://drive.google.com/open?id=1oHMZ2Tfo1g5WiY-WAukLalZkMnG8qXdA
  - Signal Sign: https://drive.google.com/open?id=1eUDeU-Zc03yAoZCTlEp1ic5iHjgAGYeB
  - Railroad Sign: https://drive.google.com/open?id=1psmRUF6pFnV2PNnOxr2mGeiCogKscE3n


## Phase 5: Testing Autonomous Vehicle with trained Model

### Video Links and Images:
  1. Lane Detection: https://youtu.be/crSBcHiNZTg
  2. Signs Detection: https://youtu.be/4-usYpNj604
  3. Trained Model Sign Detection: https://drive.google.com/open?id=1psmRUF6pFnV2PNnOxr2mGeiCogKscE3n
  4. All Sign Detection and actions: https://youtu.be/uy3XPrr3few 
  5. Autonomous Car using Data Mining - Competition Video: https://youtu.be/jQg26bn4u0w

## CNN Characteristics:

![LeNetArchitecture](https://github.com/DivyaSamragniNadakuditi/DM-Car/blob/master/Images/LeNetArchitecture.png)

  1. Architecture - LeNet : We have used Lenet architecture for classifying the images. The LetNet architecture is an excellent “first image classifier” for Convolutional Neural Networks.The LeNet architecture consists of two sets of convolutional, activation, and pooling layers, followed by a fully-connected layer, activation, another fully-connected, and finally a softmax classifier. The LeNet architecture is implemented using Keras and Python
File: LeNet.py is the code used for implementing Lenet architecture.

 2. We are using different labels for signboards so that it will know which label used for which sign boards, so according it will      changes it's a label. We also checked the probability of each signboard to get better results.
  
## Training Model for Various Signs using Convolutional Neural Networks

We have trained the Car using Convolutional Neural network for the following 6 Signs:
  - Stop Sign 
  - High Speed Sign 
  - Low Speed Sign 
  - Trafic Signal Sign 
  - RailRoad Sign 
  - Yield Sign
  
 #### Training model accuracy
 
![Training-Model](https://github.com/DivyaSamragniNadakuditi/DM-Car/blob/master/Images/Training-model-accuracy.jpg)


#### Training Model Plot

![Training-Model-Plot](https://github.com/DivyaSamragniNadakuditi/DM-Car/blob/master/Images/Training-model-plot.jpg)

## Dataset
  1. Stop Sign Images: https://drive.google.com/open?id=1cgsu32XoGJW63Nt1Zl2n1vrNx-Tbtros
  2. High Speed Images https://drive.google.com/open?id=1GEFpTuB8Rnm2vT4GvPO0yPBp2KuHxYD1
  3. Low Speed Images: https://drive.google.com/open?id=1zLWLQyFwJd9UaKWByEm5y0dwAQxwDTU5
  4. Traffic Signal images: https://drive.google.com/open?id=1alLO-oaUTNcpbTTGahVil35lidzU4iuu
  5. RailRoad Images: https://drive.google.com/open?id=1o5gVb9AJrc7gbV-2SIZ0JNewsNT8bHVK
  6. Yield Sign Images: https://drive.google.com/open?id=1V1eR2iBYRFvlwVSyOObA1xcLvKadZN8S
  7. No Sign: https://drive.google.com/open?id=14RERMcxZ390g0R4DUvF-1gFQ32ZdoQx8
  
## Final Phase:
Final video: https://youtu.be/-vgHj1xRWt0

