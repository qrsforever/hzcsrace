#!/usr/bin/python3
# -*- coding: utf-8 -*-

import functools
import traceback
import json

DEBUG = False


def errmsg(code, err):
    msg = {
        'errno': code,
        'result': {
            'errtext': str(err),
            'traceback': traceback.format_exc(limit=10)
        }
    }
    if DEBUG or code != 0:
        print(msg)
    return msg


def catch_error(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except json.decoder.JSONDecodeError as err1:
            return json.dumps(errmsg(90001, err1))
        except ImportError as err2:
            return json.dumps(errmsg(90002, err2))
        except KeyError as err3:
            return json.dumps(errmsg(90003, err3))
        except ValueError as err4:
            return json.dumps(errmsg(90004, err4))
        except AssertionError as err5:
            return json.dumps(errmsg(90005, err5))
        except AttributeError as err6:
            return json.dumps(errmsg(90006, err6))
        except Exception as err99:
            return json.dumps(errmsg(90099, err99))
    return decorator
