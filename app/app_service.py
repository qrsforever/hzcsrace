#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-09


import json
import traceback
import os # noqa
import time
import argparse
# import raceai.runner # noqa
import zmq
import redis

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
g_topics = []
g_redis = None


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
    # if not os.path.exists('/raceai/data/tmp'):
    #     os.makedirs('/raceai/data/tmp')
    # with open('/raceai/data/tmp/raceai.inference.json', 'w') as fw:
    #     json.dump(reqjson, fw)
    ####

    if reqjson['task'].startswith('zmq'):
        if reqjson['task'] not in g_topics:
            app_logger(f"service topic:{reqjson['task']} is not start")
            # return json.dumps({'errno': -2})
        cfg = reqjson['cfg']
        if isinstance(cfg, str):
            cfg = json.loads(cfg)

        # app_logger(cfg)
        # TODO
        msgkey = cfg['pigeon']['msgkey']
        if msgkey[:2] == 'nb': # notebook call
            task_topic = f'{reqjson["task"]}_{msgkey}'
        else:
            task_topic = reqjson['task']
        for i in range(3):
            res = g_redis.get(task_topic)
            if res is None or res.decode() != '1':
                break
            if i == 2:
                app_logger('error: task already run.')
                return json.dumps({'errno': -3, 'errtxt': 'task already run'})
            time.sleep(0.3)
        try:
            len = g_redis.llen(msgkey)
            if len > 0:
                app_logger(f'error: redis key {msgkey} length is not 0: {len}')
            g_redis.delete(msgkey)
            zmqpub.send_string('%s %s' % (reqjson['task'], json.dumps(cfg, separators=(',',':'))))
        except Exception:
            app_logger('error: task run error.')
            return json.dumps({'errno': -4, 'errtxt': traceback.format_exc(limit=3)})
        return json.dumps({'errno': 0})

    cfg = OmegaConf.create(reqjson['cfg'])
    runner = Registrable.get_runner(reqjson['task'])
    with race_subprocess(runner, cfg) as queue:
        return queue.get()


@app.route('/raceai/private/pushmsg', methods=['POST', 'GET'], endpoint='pushmsg')
@catch_error
@race_timeit(app_logger)
def _framework_message_push():
    try:
        key = request.args.get("key", default='unknown')
        val = request.get_data().decode()
        if key == 'add_topic':
            if val not in g_topics:
                app_logger('add topic: %s' % val)
                g_topics.append(val)
        elif key == 'del_topic':
            if val in g_topics:
                app_logger('del topic: %s' % val)
                g_topics.remove(val)
        elif key == 'zmp_run':
            topic, secs = val.split(':')
            g_redis.getset(topic, 1)
            g_redis.expire(topic, secs)
        elif key == 'zmp_end':
            g_redis.delete(val)
        else:
            if g_redis:
                g_redis.lpush(key, val)
                g_redis.expire(key, 3600) # 1 days
    except Exception as err:
        app_logger(err)
        return "-1"
    return "0"


@app.route('/raceai/private/popmsg', methods=['GET'])
def _framework_message_pop():
    response = []
    try:
        if g_redis:
            key = request.args.get("key", default='unknown')
            while True:
                item = g_redis.rpop(key)
                if item is None:
                    break
                response.append(json.loads(item.decode()))
    except Exception as err:
        app_logger(err)
    finally:
        return json.dumps(response)


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
    parser.add_argument(
            '--redis_addr',
            default=None,
            type=str,
            dest='redis_addr',
            help="redis address")
    parser.add_argument(
            '--redis_port',
            default=10090,
            type=int,
            dest='redis_port',
            help="redis port")
    parser.add_argument(
            '--redis_passwd',
            default='123456',
            type=str,
            dest='redis_passwd',
            help="redis passwd")

    args = parser.parse_args()

    try:
        zmqpub.bind("tcp://*:5555")
        g_redis = redis.StrictRedis(args.redis_addr,
                port=args.redis_port,
                password=args.redis_passwd)
        print(f'{args.redis_addr} {args.redis_port}')
    except Exception as err:
        app_logger('{}'.format(err))
    try:
        app.run(host=args.host, port=args.port, debug=bool(args.debug))
    finally:
        pass
