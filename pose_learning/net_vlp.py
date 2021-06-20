#!/usr/bin/env python3
# Developed by Junyi Ma
# This file is covered by the LICENSE file in the root of this project.
# Brief: A devised CNN built for RS-Lidar-16 (16 beams).

import sys
import tensorflow as tf
from tensorflow.python.keras import backend as K
tfk = tf.keras
Input = tfk.layers.Input
Conv2D = tfk.layers.Conv2D
Dense = tfk.layers.Dense 
Flatten = tfk.layers.Flatten
Reshape = tfk.layers.Reshape
Lambda= tfk.layers.Lambda
Dropout= tfk.layers.Dropout
MaxPool2D = tfk.layers.MaxPool2D
Model = tfk.models.Model



# from keras.regularizers import l2
l2 = tfk.regularizers.l2

def generateNetwork(input_shape):

    mc_input = Input(input_shape,name='multicues')  # mc: multiple cues

    kernel_regularizer = None


    conv1 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=input_shape,
                    kernel_regularizer=kernel_regularizer, name="s_conv1", padding="same")
    mc = conv1(mc_input)


    conv2 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                    kernel_regularizer=kernel_regularizer, name="s_conv2", padding="same")
    mc = conv2(mc)

    drop1 = Dropout(0.3)
    mc = drop1(mc)

    mp2d1 = MaxPool2D((2,2))
    mc = mp2d1(mc)


    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                    kernel_regularizer=kernel_regularizer, name="s_conv3", padding="same")
    mc = conv3(mc)

    conv31 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                    kernel_regularizer=kernel_regularizer, name="s_conv31", padding="same")
    mc = conv31(mc)

    drop2 = Dropout(0.3)
    mc = drop2(mc)

    mp2d2 = MaxPool2D((2,2))
    mc = mp2d2(mc)

    conv4 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', name="s_conv4",
                    kernel_regularizer=kernel_regularizer, padding="same")
    mc = conv4(mc)

    conv5 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', name="s_conv5",
                    kernel_regularizer=kernel_regularizer, padding="same")
    mc = conv5(mc)

    drop3 = Dropout(0.3)
    mc = drop3(mc)

    mp2d3 = MaxPool2D((2,2))
    mc = mp2d3(mc)
    
    flattened = Flatten()(mc)
    flattened = Dense(100, activation='relu', name='dense1')(flattened)
    drop4 = Dropout(0.3)
    flattened = drop4(flattened)
    flattened = Dense(100, activation='relu', name='dense2')(flattened)
    drop5 = Dropout(0.3)
    flattened = drop5(flattened)
    prediction = Dense(7, activation='linear', name='mutualpose')(flattened)


    pose_net = Model(inputs=mc_input, outputs=prediction)

    return pose_net

# if __name__ == "__main__":
#     input_shape = (80, 1800, 6)
#     model = generateNetwork(input_shape)
#     # optimizer = keras.optimizers.SGD(
#     # lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#     optimizer = tfk.optimizers.SGD(lr=0.001,decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss="mean_squared_error", optimizer=optimizer)
#     model.summary()