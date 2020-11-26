#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-09


import json
import argparse
import tempfile

import raceai.runner # noqa

from raceai.utils.registrable import Registrable
from raceai.utils.misc import race_timeit, race_subprocess
from raceai.utils.error import catch_error
from flask import Flask, request
from flask_cors import CORS
from omegaconf import OmegaConf


app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/raceai/framework/training', methods=['POST'], endpoint='training')
@race_timeit(app.logger.info)
@catch_error
def _framework_training():
    try:
        reqjson = json.loads(request.get_data().decode()) # noqa
    except Exception:
        pass

    return 'not impl'


@app.route('/raceai/framework/inference', methods=['POST'], endpoint='inference')
@race_timeit(app.logger.info)
@catch_error
def _framework_inference():
    with tempfile.TemporaryDirectory() as tmp_dir:
        reqjson = json.loads(request.get_data().decode())

        #### debug
        with open('/raceai/data/tmp/raceai.json', 'w') as fw:
            json.dump(reqjson, fw)
        ####
        reqjson['cfg']['runner']['cache_dir'] = tmp_dir

        cfg = OmegaConf.create(reqjson['cfg'])
        runner = Registrable.get_caller(reqjson['task'])
        with race_subprocess(runner, cfg) as queue:
            return queue.get()


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
