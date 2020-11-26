#!/usr/bin/python3
# -*- coding: utf-8 -*-

import onnxruntime as rt
import numpy
import cv2
import time

img = cv2.imread('/raceai/data/tmp/det_test.jpeg')
img = cv2.resize(img, dsize=(480, 640), interpolation=cv2.INTER_AREA)
img.resize((1, 3, 480, 640))

t0 = time.time()
sess = rt.InferenceSession("/raceai/data/tmp/faces_s/weights/best.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: img.astype(numpy.float32)})
t1 = time.time()
print(pred_onx)
print('Time: ', t1 - t0)
