#!/usr/local/bin/python3

import os
import sys
import time
import json
import functools
import importlib
import tempfile
import multiprocessing
import requests

from urllib import request, parse
from omegaconf.dictconfig import DictConfig
from contextlib import contextmanager
from raceai.utils.error import errmsg

DEBUG = False
TEMPDIR = '/tmp/'


def race_blockprint(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        results = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return results
    return decorator


def race_timeit(handler):
    def decorator(func):
        @functools.wraps(func)
        def _timed(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            if handler:
                handler('{} took {:.3f}'.format(func.__name__, (te - ts) * 1000))
            return result
        return _timed
    return decorator


def race_load_class(impstr):
    if DEBUG:
        print(f'load class: {impstr}')
    module_name, class_name = impstr.rsplit('.', 1)
    module_ = importlib.import_module(module_name)
    return getattr(module_, class_name)


def race_convert_dictkeys(x, uppercase=True):
    fcvt = str.upper if uppercase else str.lower
    if isinstance(x, (dict, DictConfig)):
        return dict((fcvt(k), race_convert_dictkeys(v, uppercase)) for k, v in x.items())
    return x


def race_data(x):
    if x.startswith('http') or x.startswith('ftp'):
        x = parse.quote(x, safe=':/?-=')
        r = request.urlretrieve(x, os.path.join(TEMPDIR, os.path.basename(x)))
        x = r[0]
    elif x.startswith('oss://'):
        raise NotImplementedError('weight schema: oss')
    elif x.startswith('file://'):
        x = x[7:]
    if DEBUG:
        print(x)
    return x


def race_report_result(key, data):
    api = 'http://{}/raceai/private/pushmsg?key={}'.format('0.0.0.0:9119', key)
    if isinstance(data, str):
        data = json.loads(data)
    requests.post(api, json=data)


@contextmanager
def race_subprocess(func, *args):
    queue = multiprocessing.Queue()

    def _target(queue, *args):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                global TEMPDIR
                TEMPDIR = tmp_dir
                queue.put(func(*args))
        except Exception as err:
            queue.put(errmsg(90099, err))

    proc = multiprocessing.Process(target=_target, args=(queue, *args))
    proc.start()
    yield queue
    proc.join()
