#!/usr/bin/python3
# -*- coding: utf-8 -*-


import zmq
import time

context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5655))

zmqsub.subscribe('a/b')

while True:
    msg = ''.join(zmqsub.recv_string().split(' ')[1:])
    print('msg: %s' % msg)
    time.sleep(1)

