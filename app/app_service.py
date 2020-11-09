#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-09

import os, time, json
import argparse
import redis

from flask import Flask, request
from flask_cors import CORS
from threading import Thread

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/raceai/framework/fit', methods=['POST'], endpoint='fit')
def _framework_fit():
    try:
        reqjson = json.loads(request.get_data().decode())
    except Exception:
        pass

@app.route('/raceai/framework/test', methods=['POST'], endpoint='test')
def _framework_test():
    try:
        reqjson = json.loads(request.get_data().decode())
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--host',
            default='0.0.0.0',
            type=str,
            dest='host',
            help="host to run raceai service")
    parser.add_argument(
            '--port',
            default=9119,
            type=int,
            dest='port',
            help="port to run raceai service")

    args = parser.parse_args()

    try:
        app.run(host=args.host, port=args.port)
    finally:
        pass
