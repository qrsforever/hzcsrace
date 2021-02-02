#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-09


import json
import argparse
import raceai.runner # noqa
import zmq

from raceai.utils.registrable import Registrable
from raceai.utils.misc import race_timeit, race_subprocess
from raceai.utils.error import catch_error
from flask import Flask, request
from flask_cors import CORS
from omegaconf import OmegaConf

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.debug = True
app_logger = app.logger.info

context = zmq.Context()
zmqpub = context.socket(zmq.PUB)


@app.route('/raceai/framework/training', methods=['POST'], endpoint='training')
@catch_error
@race_timeit(app_logger)
def _framework_training():
    reqjson = json.loads(request.get_data().decode())

    #### debug
    with open('/raceai/data/tmp/raceai.training.json', 'w') as fw:
        json.dump(reqjson, fw)
    ####

    cfg = OmegaConf.create(reqjson['cfg'])
    runner = Registrable.get_runner(reqjson['task'])
    with race_subprocess(runner, cfg) as queue:
        return queue.get()


@app.route('/raceai/framework/inference', methods=['POST'], endpoint='inference')
@catch_error
@race_timeit(app_logger)
def _framework_inference():
    reqjson = json.loads(request.get_data().decode())

    #### debug
    with open('/raceai/data/tmp/raceai.inference.json', 'w') as fw:
        json.dump(reqjson, fw)
    ####

    if reqjson['task'].startswith('zmq'):
        zmqpub.send_string('%s %s' % (reqjson['task'], json.dumps(reqjson['cfg'], separators=(',',':'))))
        return json.dumps({'errno': 0})

    cfg = OmegaConf.create(reqjson['cfg'])
    runner = Registrable.get_runner(reqjson['task'])
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
            '--debug',
            default=0,
            type=int,
            dest='debug',
            help="debug mode")
    parser.add_argument(
            '--port',
            default=9119,
            type=int,
            dest='port',
            help="port to run raceai service")

    args = parser.parse_args()

    zmqpub.bind("tcp://*:5555")
    try:
        app.run(host=args.host, port=args.port, debug=bool(args.debug))
    finally:
        pass
