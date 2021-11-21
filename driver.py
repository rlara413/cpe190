from ubidots import ApiClient
from smbus2 import SMBus
from mlx90614 import MLX90614
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

import RPi.GPIO as GPIO 
import time
import keyboard
import threading
import numpy as np
import argparse
#import sleep
import imutils
import time
import sys
import cv2
import os

GPIO.setmode(GPIO.BCM)
GPIO.setup(7, GPIO.IN)

peoplecount = 0
counter = 0
try: 
    api = ApiClient(token='BBFF-jvBXecob5OfXxnHc0Rh5UZP0Nfjz4a')
    people = api.get_variable('61905080d467923fefef3af3')
except: 
    print ("Couldn't connect to the API, check your Internet connection")
    counter = 0
    peoplev = 0
    
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    #condition = 0

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# Check Temperature
def tempsense():
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    print ("Object Temperature :"), sensor.get_object_1()
    temp = sensor.get_object_1()
    bus.close()
    if temp > 37:
        print("temperature too high")
    else:
        unlock()

# Lock System
def unlock():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.OUT)
    print("door unlocked please enter before the door locks again \n")
    GPIO.output(17, GPIO.HIGH)
    time.sleep(5)
    GPIO.output(17, GPIO.LOW)
    time.sleep(2)
    
# def active():
#     go = 0
#     go = input("welcome please press 1 and then enter: ")
#     go = int(go)
#     if go == 1:
#         return go
# 
# def assist():
#     assist = 0
#     assist = input("if you need assistance press 2 then enter")
#     assist = int(assist)
#     
#     if assist == 2:
#         print("calling for assistance")
#         time.sleep(10)
#         
# def override():
#     over = 0
#     over = input("in an override is needed press 3 then enter, in not press 1 then enter: ")
#     over = input(over)
#     if over == 3:
#         print("please wait a momenent while door unlocks")
#         time.sleep(5)
#         print("door is unlocking")
#         unlock()
    
    
    
    
# DRIVER BEGINS HERE
while (1):  
    try:
        presence = GPIO.input(7) 
        if (presence): 
            peoplecount += 1 
            presence = 0
            time.sleep(1.5)
            time.sleep(1)
            counter += 1

        if (counter >= 10):
            print (peoplecount) 
            people.save_value({'value':peoplecount})
            counter = 0
            peoplev = 0
            print('MASK DETECTED')
            tempsense()

        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            #if mask > withoutMask:
            #    print('MASK DETECTED')
            #    tempsense()

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # Comment both these lines to remove the UI
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    except KeyboardInterrupt:
        
        #################
        # CLEAN UP HERE #
        #################
        print('\n\nsafely closing...')
        cv2.destroyAllWindows()
        vs.stop()
        sys.exit()
