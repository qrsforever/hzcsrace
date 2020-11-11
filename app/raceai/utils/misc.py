#!/usr/local/bin/python3

import os
import sys
import time
import importlib
import multiprocessing

from contextlib import contextmanager


def race_blockprint(func):
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
        def _timed(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            if handler:
                handler('{} took {:.3f}'.format(func.__name__, (te - ts) * 1000))
            return result
        return _timed
    return decorator


@contextmanager
def race_subprocess(func, *args):
    queue = multiprocessing.Queue()

    def _target(queue, *args):
        try:
            queue.put(func(*args))
        except Exception as err:
            queue.put({'error': '{}'.format(err)})

    proc = multiprocessing.Process(target=_target, args=(queue, *args))
    proc.start()
    yield queue
    proc.join()


def race_load_class(impstr):
    module_name, class_name = impstr.rsplit('.', 1)
    module_ = importlib.import_module(module_name)
    return getattr(module_, class_name)
