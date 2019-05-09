![car-image](https://github.com/DivyaSamragniNadakuditi/DM-Car/blob/master/Images/Car-Image.png)


# Autonomous Car project in CPSC552 Data Mining 
This Repository contains the setting up of an autonomous raspberry pi-car and training it with data mining models.

# Group 1
<hr />

## Contents:
  1. Introduction
  2. Assembling the Autonomous Car
  3. Setting-up Raspberry Pi
  4. Dataset
  5. Training Model for Various Signs using Convolutional Neural Networks
  6. Improvement Of the Autonomous Car
  
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

### Question: Car Improvement suggestions:

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
  

### Question: improve the connectivity of picar 






## Phase 3:

### How to improve the lane detection

### Question: How to improve the controlling front wheels and back wheels motors (i.e., servos)

Lane Detection Video: https://youtu.be/crSBcHiNZTg

## Phase 4:

### Questions:
  1. Image Size
  
  
  2. How to design CNN architecture including how many layers, what kind of layers, and so on
  
  
  3. How to optimize the model including parameter values, drop out, backpropagation, learning rate, # of epoch and so on
  
  
  4. Evaluations
  
  
  5. How to overcome the limitations in your DM-Car implementation
  
  Trained model sign detection images:
  - Stop Sign: https://drive.google.com/open?id=1oHMZ2Tfo1g5WiY-WAukLalZkMnG8qXdA
  - Signal Sign: https://drive.google.com/open?id=1eUDeU-Zc03yAoZCTlEp1ic5iHjgAGYeB
  - Railroad Sign: https://drive.google.com/open?id=1psmRUF6pFnV2PNnOxr2mGeiCogKscE3n


## Phase 5: Testing Autonomous Vehicle with trained Model









## Dataset
  1. Stop Sign Images: https://drive.google.com/open?id=1fbCMN62OzVuYi596aOx4-TIq_Lqe5W14
  2. High Speed Images https://drive.google.com/open?id=1TRyYTwzTTWxPLIIKIqBa0deLJ2ICVN3K
  3. Low Speed Images: https://drive.google.com/open?id=1GaFCKJrUH7aff9nJqgEJ9990qFFbya9e
  4. Traffic Signal images: https://drive.google.com/open?id=1WWKPTqv2GRUlpx6rw_AZwXDSIHuA2LlQ
  5. RailRoad Images: https://drive.google.com/open?id=1vKr5ShTeaz1iPxNJDMWjB7vQmNkkPnbR
  6. Yield Sign Images: https://drive.google.com/open?id=11uYymdVrB_DhhJbWKtHQ1VmoF72nj3U6
  
## Training Model for Various Signs using Convolutional Neural Networks
We have trained the Car using Convolutional Neural network for the following Signs:
  Stop Sign, 
  High Speed Sign, 
  Low Speed Sign, 
  Trafic Signal Sign, 
  RailRoad Sign, 
  Yield Sign.
  
![stop-NN](https://github.com/DivyaSamragniNadakuditi/DM-Car/blob/master/Images/stop-NN.png)

Video Link:
https://www.youtube.com/watch?v=OOpc0Sy0k70&feature=youtu.be

## Improvement Of the Autonomous Car
1. The parts of the car are open. This is not good for the safety of the car. It would be better if we can safe gaurd them with a covering.
2. Camera position can be improved so that we can visualize things more clearly. As in the present position, the camera movement is not that flexible.

