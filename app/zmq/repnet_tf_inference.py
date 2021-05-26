#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file repnet_tf_inference.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-05-26 17:48


import argparse
import time, os
import zmq
from utils import get_model, read_video
from repnet import get_counts, create_count_video

from omegaconf import OmegaConf
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)
from raceai.utils.misc import ( # noqa
        race_oss_client,
        race_object_put,
        race_report_result,
        race_data)

race_set_loglevel('info')
race_set_logfile('/tmp/raceai-repnet_tf.log')

_DEBUG_ = True
context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))

osscli = race_oss_client(bucket_name='raceai')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("path", type=str, help="Input video file path or root.")
parser.add_argument("--topic", type=str, help="topic")
parser.add_argument("--out", default="/tmp/export", help="Output video file path or root")
parser.add_argument("--ckpt", default="/tmp/weights", type=str, help="Checkpoint weights root.")
main_args = parser.parse_args()


def inference(model, opt):
    msgkey = main_args.topic
    if 'msgkey' in opt.pigeon:
        msgkey = opt.pigeon.msgkey
    user_code = 'unkown'
    if 'user_code' in opt.pigeon:
        user_code = opt.pigeon.user_code
    batch_size = 20
    if 'batch_size' in opt:
        batch_size = opt.batch_size
    threshold = 0.2
    if 'threshold' in opt:
        threshold = opt.threshold
    in_threshold = 0.5
    if 'in_threshold' in opt:
        in_threshold = main_args.in_threshold
    strides = [1, 2, 3, 4]
    if 'strides' in opt:
        strides = opt.strides
    constant_speed = False
    if 'constant_speed' in opt:
        constant_speed = opt.constant_speed
    median_filter = True
    if 'median_filter' in opt:
        median_filter = opt.median_filter
    fully_periodic = False
    if 'fully_periodic' in opt:
        fully_periodic = opt.fully_periodic
    save_video = True
    if 'save_video' in opt:
        save_video = opt.save_video
    viz_reps = True
    if 'viz_reps' in opt:
        viz_reps = opt.viz_reps

    outfile = os.path.join(main_args.out, user_code, 'target.mp4')

    frames, vid_fps = read_video(opt.video, rot=None)

    s_time = time.time()
    (pred_period, pred_score,
            within_period, per_frame_counts, chosen_stride) = get_counts(
            model,
            frames,
            strides=strides,
            batch_size=batch_size,
            threshold=threshold,
            within_period_threshold=in_threshold,
            constant_speed=constant_speed,
            median_filter=median_filter,
            fully_periodic=fully_periodic)
    infer_time = time.time() - s_time
    Logger.info('model inference using time: %d' % infer_time)

    if save_video:
        create_count_video(frames, per_frame_counts, within_period, score=pred_score,
                fps=vid_fps, output_file=outfile, delay=1000 / vid_fps, vizualize_reps=viz_reps)


if __name__ == "__main__":
    if not _DEBUG_:
        race_report_result('add_topic', main_args.topic)
        zmqsub.subscribe(main_args.topic)

    try:
        # Load model
        repnet_model = get_model(main_args.ckpt)

        if not _DEBUG_:
            while True:
                Logger.info('wait task')
                zmq_cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
                zmq_cfg = OmegaConf.create(zmq_cfg)
                Logger.info(zmq_cfg)
                if 'pigeon' not in zmq_cfg:
                    continue
                Logger.info(zmq_cfg.pigeon)
                inference(repnet_model, zmq_cfg)
                time.sleep(0.01)
        else:
            zmq_cfg = {
                    "pigeon": {"msgkey": "123", "user_code": "123"},
                    "video": "https://raceai.s3.didiyunapi.com/data/media/videos/repnet_test.mp4"
            }
            zmq_cfg = OmegaConf.create(zmq_cfg)
            Logger.info(zmq_cfg)
            inference(repnet_model, zmq_cfg)
    except Exception:
        pass
    finally:
        if not _DEBUG_:
            race_report_result('del_topic', main_args.topic)
        Logger.info('end')
