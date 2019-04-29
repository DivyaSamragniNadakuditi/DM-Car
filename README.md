![car-image](https://github.com/DivyaSamragniNadakuditi/DM-Car/blob/master/Images/Car-Image.png)


# Autonomous Car project in CPSC552 Data Mining 
This Repository contains the setting up of an autonomous raspberry pi-car and training it with data mining models.

## Contents:
  1. Introduction
  2. Assembling the Autonomous Car
  3. Setting-up Raspberry Pi
  4. Dataset
  5. Training Model for Various Signs using Convolutional Neural Networks
  6. Improvement Of the Autonomous Car
  
## Introduction:
In this project, we will be working on the implementation of data mining Techniques- Convolutional Neural Networks(CNN) for real-world problem Autonomous vehicle. For this project, we used the Smart Video Car Kit V2.0 for Raspberry Pi. We will Assemble the car, Set up Raspberry Pi, Train a Model for detecting traffic signs and act accordingly.

## Assembling the  Autonomous Car
In this stage, we assemble the Smart Video Car for RaspberryPi. We attach fix therear wheels,Upper plate, Battery Holder, Motor Driver, PWM Driver,Robot HATS, Raspberry Pi. Then we build the Circuits.Ater Configuring the Servos(Details mentioned in the Second part), we fix the front wheels and Camera.  Below is the link of GoogleDrive where the images of assembling the car are present.

Link of Google Drive: https://drive.google.com/open?id=1_i5RnjMk-N36RWOn_UuZXdj3fTnPk2Ab


## SettingUp and Configuration Of RaspberryPi

We Download Raspbian Image and write it into an SD Card. Later, we edit the wpa configuration file and set the network details. We further use VNC Viewer to connect to the Raspberrypi. The Images of the setting up and Configuration steps are in the link below.

Link of Google Drive:https://drive.google.com/open?id=13gxQwoQr6Or3sMAl9gn7LvNw2mLJWo44

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

