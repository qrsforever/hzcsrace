#!/usr/bin/python3
# -*- coding: utf-8 -*-

import zmq
import time

context = zmq.Context()
zmqpub = context.socket(zmq.PUB)

zmqpub.bind("tcp://*:5655")

while True:
    zmqpub.send_string('%s %s' % ('a/b', 'hello world'))
    time.sleep(1)
