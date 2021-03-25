#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file asr_inference.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-03-25 14:57

import traceback
import argparse
import torch
from omegaconf import OmegaConf

from raceai.utils.misc import race_load_class, race_report_result
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)

import time # noqa
import zmq

from speechbrain.pretrained import EncoderDecoderASR

TEST_DIR = '/raceai/data/ckpts/asr/asr-crdnn-rnnlm-librispeech'

asr_model = EncoderDecoderASR.from_hparams(source=TEST_DIR, savedir=TEST_DIR)
result = asr_model.transcribe_file(f'{TEST_DIR}/example.wav')

print(type(result))
print(result)

