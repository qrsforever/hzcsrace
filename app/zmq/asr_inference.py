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
import os
from omegaconf import OmegaConf

from raceai.utils.misc import race_load_class, race_report_result
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)

import time # noqa
import zmq

from speechbrain.pretrained import EncoderDecoderASR, TransformerASR

race_set_loglevel('info')

context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))


def inference(opt):
    asr_model = EncoderDecoderASR.from_hparams(
            source=opt.ckpts,
            savedir=opt.ckpts,
            run_opts={'device': opt.device})
    Logger.info('load model finish...')
    zmq_stats_count = 0
    while True:
        try:
            cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
            zmq_stats_count += 1
            stime = time.time()
            cfg = OmegaConf.create(cfg)
            Logger.info(cfg)
            if 'pigeon' not in cfg:
                continue
            msgkey = opt.topic
            if 'msgkey' in cfg.pigeon:
                msgkey = cfg.pigeon.msgkey
            resdata = {'pigeon': dict(cfg.pigeon), 'task': opt.topic, 'errno': 0, 'result': []}
            data_loader = race_load_class(cfg.data.class_name)(cfg.data.params).get()
            i = 0
            for audio_path, source in data_loader:
                Logger.info(audio_path)
                if not os.path.exists(audio_path):
                    Logger.warn(f'{audio_path} is not found!')
                    continue
                result = asr_model.transcribe_file(audio_path)
                resdata['result'].append({'id': i, 'text': result})
                i += 1
            resdata['running_time'] = round(time.time() - stime, 3)
            Logger.info(resdata)
            race_report_result(msgkey, resdata)
            Logger.info('[%6d] time consuming: [%.2f]s' % (zmq_stats_count % 99999, resdata['running_time']))
            # Logger.info(resdata)
        except Exception:
            resdata['errno'] = -1 # todo
            resdata['traceback'] = traceback.format_exc()
            race_report_result(msgkey, resdata)
            Logger.error(resdata)
        time.sleep(0.01)


if __name__ == '__main__':
    Logger.info('start yolo main')

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', type=str, default='/ckpts/', help='model ckpts path(s)')
    parser.add_argument('--topic', default='zmq.asr.librispeech.inference', help='sub topic')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or cuda:0, cuda:1 or cpu')
    opt = parser.parse_args()
    Logger.info(opt)

    os.chdir("/tmp") # Workaround for `transcribe_file` make soft link in current directory

    zmqsub.subscribe(opt.topic)
    race_report_result('add_topic', opt.topic)

    race_set_logfile(f'/tmp/raceai-{opt.topic}.log')

    try:
        with torch.no_grad():
            Logger.info('start speechbrain interence')
            inference(opt)
            Logger.info('never run here')
    finally:
        race_report_result('del_topic', opt.topic)
