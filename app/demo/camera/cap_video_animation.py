#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file cap_video_animation.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-24 18:04


import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def grab_frame(cap):
    _, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


cap = cv2.VideoCapture(0)
fig = plt.gcf()
img = plt.imshow(grab_frame(cap))


def init():
    print('init')


def update(i):
    img.set_data(grab_frame(cap))


def key_event(event):
    if event.key == 'q':
        print('quit')
        plt.close(event.canvas.figure)


cid = fig.canvas.mpl_connect("key_press_event", key_event) # noqa

animation = FuncAnimation(
        fig=fig,
        func=update,
        init_func=init,
        interval=10) # noqa
plt.show()
