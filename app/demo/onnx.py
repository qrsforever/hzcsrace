#!/usr/bin/python3
# -*- coding: utf-8 -*-

import onnxruntime as rt
import numpy

sess = rt.InferenceSession("/raceai/data/tmp/faces_s/weights")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)
