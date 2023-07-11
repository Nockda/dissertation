import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import shape,math
from tensorflow.keras import Input,layers,Model
from tensorflow.keras.losses import mse,binary_crossentropy
from tensorflow.keras.utils import plot_model

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_thunder_3.tflite')
interpreter.allocate_tensors()


epochs = 20



def draw_keypoints(frame, keypoints, confidence_threshold):
    # c is channel
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1>confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
        
# 한개만 테스트해보기

file_path = os.path.join("./video/", "walk.mov")
cap = cv2.VideoCapture(file_path)
kp_output = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
    input_image = tf.cast(img, dtype=tf.float32)

    # setup the input and output
    input_details=interpreter.get_input_details()
    output_details=interpreter.get_output_details()


    # make prediction
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    print(keypoints_with_scores)
    kp_output.append(keypoints_with_scores[:,:,:,:2])

    #rendering
    draw_connections(frame, keypoints_with_scores, EDGES, 0.11)
    draw_keypoints(frame, keypoints_with_scores, 0.11)
    

    if ret:  # 프레임이 유효한 경우에만 imshow를 호출합니다.
        cv2.imshow('MoveNet Lightning', frame)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
#             cap = cv2.VideoCapture('walk.mov')

kp_output = np.array(kp_output)
# kp_output = kp_output.squeeze()



#########################################

# network parameters
input_shape = np.shape(x_train[0])[0]
original_dim= input_shape
intermediate_dim = 512
latent_dim = 2