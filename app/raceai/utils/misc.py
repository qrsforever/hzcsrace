#!/usr/local/bin/python3

import os
import sys
import time

def race_blockprint(func):
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        results = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return results
    return wrapper


def race_timeit(handler):
    def decorator(func):
        def _timed(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            if handler:
                handler('"{}" took {:.3f} ms to execute'.format(func.__name__, (te - ts) * 1000))
            return result
        return _timed
    return decorator
