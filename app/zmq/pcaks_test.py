#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file pcaks_test.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-12-27 18:18


## noqa

import argparse
import time, os, json
import zmq
import shutil
import numpy as np
import cv2
import traceback
import pickle

import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats
import pandas as pd
from sklearn import preprocessing

import itertools 
import scipy.signal as signal
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import hashlib

import io
import requests

from omegaconf import OmegaConf
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)
from raceai.utils.misc import ( # noqa
        race_oss_client,
        race_object_put,
        race_object_put_jsonconfig,
        race_object_remove,
        race_report_result,
        race_data)

race_set_loglevel('info')
race_set_logfile('/tmp/raceai-pcaks_test.log')

context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))

osscli = race_oss_client()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--topic", type=str, default="a.b.c", help="topic")
main_args = parser.parse_args()

if __name__ == "__main__":
    zmqsub.subscribe(main_args.topic)
    race_report_result('add_topic', main_args.topic)
    Logger.info('main_args.topic: %s' % main_args.topic)

    ##TODO code snippets
    scaler = preprocessing.Normalizer()
    NC = 200
    pca = PCA(n_components=NC)

    scaler.fit(cdata)
    data_out = kspca.fit_transform(scaler.transform(cdata))

    ecdfs = [ECDF(sample) for sample in data_out.T]

    ksmodel = {
        'pca': kspca,
        'scaler': scaler,
        'ecdfs': ecdfs,
    }

    pcaks_pkl_path = '/data/ksmodel.pkl'

    with io.BytesIO() as bio:
        pickle.dump(ksmodel, bio)
        md5 = hashlib.md5(bio.getvalue()).hexdigest()[:12]
        oss_put_bytes(bio.getvalue(), f'datasets/weights/{md5}.pkl')
        print(f'https://frepai.s3.didiyunapi.com/datasets/weights/{md5}.pkl')
