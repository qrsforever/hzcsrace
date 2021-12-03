#!/usr/local/bin/python3

import os
import sys
import io
import json
import time
import functools
import importlib
import tempfile
import multiprocessing
import requests
import errno
import ssl

from urllib import request, parse
from omegaconf.dictconfig import DictConfig
from contextlib import contextmanager
from raceai.utils.error import errmsg

ssl._create_default_https_context = ssl._create_unverified_context

DEBUG = False
TEMPDIR = '/tmp/'


class AssertionError(Exception):
    """
    Args:
        errcode: int
        errtext: str
    """
    def __init__(self, errcode, errtext=''):
        self.errcode = errcode
        self.errtext = errtext


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
        for _ in range(5):
            try:
                r = request.urlretrieve(x, os.path.join('/raceai/data/tmp', os.path.basename(x)))
                x = r[0]
            except Exception:
                time.sleep(1)
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
        requests.post(api, data=data)
        # data = json.loads(data)
    else:
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


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def race_oss_client(server_url=None, access_key=None, secret_key=None,
        region='gz', bucket_name=None):
    if server_url is None:
        server_url = os.environ.get('MINIO_SERVER_URL')
    if access_key is None:
        access_key = os.environ.get('MINIO_ACCESS_KEY')
    if secret_key is None:
        secret_key = os.environ.get('MINIO_SECRET_KEY')

    from minio import Minio

    mc = Minio(
        endpoint=server_url,
        access_key=access_key,
        secret_key=secret_key,
        secure=True)

    if bucket_name:
        if not mc.bucket_exists(bucket_name):
            mc.make_bucket(bucket_name, location=region)

    return mc


def race_object_put(client, local_path,
        bucket_name=None, prefix_map=None,
        content_type='application/octet-stream', metadata=None):

    if bucket_name is None:
        bucket_name = 'raceai'

    result = []

    def _upload_file(local_file):
        if not os.path.isfile(local_file):
            return
        if prefix_map and isinstance(prefix_map, list):
            lprefix = prefix_map[0].rstrip(os.path.sep)
            rprefix = prefix_map[1].strip(os.path.sep)
            remote_file = local_file.replace(lprefix, rprefix, 1)
        else:
            remote_file = local_file.lstrip(os.path.sep)

        file_size = os.stat(local_file).st_size
        if local_file.endswith('.json'):
            content_type = 'text/json'
        elif local_file.endswith('.csv'):
            content_type = 'text/csv'
        elif local_file.endswith('.mp4'):
            content_type = 'video/mp4'
        elif local_file.endswith('.avi'):
            content_type = 'video/avi'
        else:
            content_type = 'application/octet-stream'
        with open(local_file, 'rb') as file_data:
            btime = time.time()
            etag = client.put_object(bucket_name,
                    remote_file, file_data, file_size,
                    content_type=content_type, metadata=metadata)
            if not isinstance(etag, str):
                etag = etag.etag
            etime = time.time()
            result.append({'etag': etag,
                'bucket': bucket_name,
                'object': remote_file,
                'size': file_size,
                'time': [btime, etime]})

    if os.path.isdir(local_path):
        for root, directories, files in os.walk(local_path):
            for filename in files:
                _upload_file(os.path.join(root, filename))
    else:
        _upload_file(local_path)

    return result


def race_object_get(client, remote_path, bucket_name=None, prefix_map=None):

    if bucket_name is None:
        bucket_name = 'raceai'

    remote_path = remote_path.lstrip(os.path.sep)

    result = []
    for obj in client.list_objects(bucket_name, prefix=remote_path, recursive=True):
        if prefix_map:
            local_file = obj.object_name.replace(prefix_map[0], prefix_map[1], 1)
        else:
            local_file = '/' + obj.object_name
        dfile = os.path.dirname(local_file)
        if dfile:
            mkdir_p(dfile)
        btime = time.time()
        data = client.get_object(bucket_name, obj.object_name)
        with open(local_file, 'wb') as file_data:
            for d in data.stream():
                file_data.write(d)
        etime = time.time()
        result.append({'etag': obj.etag,
            'bucket': obj.bucket_name,
            'object': obj.object_name,
            'size': obj.size,
            'time': [btime, etime]})
    return result


def race_object_remove(client, remote_path, bucket_name=None):
    if bucket_name is None:
        bucket_name = 'raceai'
    result = []
    for obj in client.list_objects(bucket_name, prefix=remote_path, recursive=True):
        client.remove_object(obj.bucket_name, obj.object_name)
        result.append({'etag': obj.etag,
            'bucket': obj.bucket_name,
            'object': obj.object_name,
            'size': obj.size})


def race_object_put_jsonconfig(client, data, path, bucket_name=None):
    if bucket_name is None:
        bucket_name = 'raceai'
    if isinstance(data, (dict, list)):
        data = json.dumps(data, ensure_ascii=False, indent=4)
    with io.BytesIO(data.encode()) as bio:
        size = bio.seek(0, 2)
        bio.seek(0, 0)
        if path[0] == '/':
            path = path[1:]
        etag = client.put_object(bucket_name, path, bio, size, content_type='text/json')
        if not isinstance(etag, str):
            etag = etag.etag
        return etag
