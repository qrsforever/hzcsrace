#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file cam_video_ion.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-24 16:26


import numpy as np # noqa
import time # noqa
import cv2
import matplotlib.pyplot as plt


def grab_frame(cap):
    _, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


cap = cv2.VideoCapture(0)
plt.ion()

img = plt.imshow(grab_frame(cap))
while(True):
    try:
        img.set_data(grab_frame(cap))
        plt.pause(0.02)
    except KeyboardInterrupt:
        break
    except Exception as err:
        print('{}'.format(err))
        break

plt.ioff()
plt.show()
