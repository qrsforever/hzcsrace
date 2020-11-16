#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-09


import json
import argparse

from raceai.utils.misc import race_timeit, race_subprocess
from raceai.utils.error import catch_error
from flask import Flask, request
from flask_cors import CORS
from omegaconf import OmegaConf
from raceai.runner.test import (
    image_classifier_test,
    image_dectection_test
)


app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/raceai/framework/fit', methods=['POST'], endpoint='fit')
@race_timeit(app.logger.info)
@catch_error
def _framework_fit():
    try:
        reqjson = json.loads(request.get_data().decode()) # noqa
    except Exception:
        pass

    return 'not impl'


@app.route('/raceai/framework/test', methods=['POST'], endpoint='test')
@race_timeit(app.logger.info)
@catch_error
def _framework_test():
    reqjson = json.loads(request.get_data().decode())
    if reqjson['task'] == 'cls.test':
        cfg = OmegaConf.create(reqjson['cfg'])
        with race_subprocess(image_classifier_test, cfg) as queue:
            return queue.get()
    elif reqjson['task'] == 'det.test':
        cfg = OmegaConf.create(reqjson['cfg'])
        with race_subprocess(image_dectection_test, cfg) as queue:
            return queue.get()
    raise NotImplementedError(f"reqjson['task']")


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
