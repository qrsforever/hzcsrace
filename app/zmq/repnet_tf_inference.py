#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file repnet_tf_inference.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-05-26 17:48


import argparse
import time, os, json
import zmq
import shutil
import numpy as np

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

_DEBUG_ = False
context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))

osscli = race_oss_client(bucket_name='raceai')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", default='', type=str, help="Input video file path or root.")
parser.add_argument("--topic", type=str, default="a.b.c", help="topic")
parser.add_argument("--out", default="/tmp/export", help="Output video file path or root")
parser.add_argument("--ckpt", default="/tmp/weights", type=str, help="Checkpoint weights root.")
main_args = parser.parse_args()


def _report_result(msgkey, resdata):
    if not _DEBUG_:
        race_report_result(msgkey, resdata)
        race_report_result('zmp_run', f'{main_args.topic}:120')
    else:
        pass


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
        in_threshold = opt.in_threshold
    strides = [1, 2, 3, 4]
    if 'strides' in opt:
        strides = list(opt.strides)
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
    # viz_reps = True
    # if 'viz_reps' in opt:
    #     viz_reps = opt.viz_reps

    if save_video:
        progress_strides_weight = 0.25
    else:
        progress_strides_weight = 0.95

    resdata = {'pigeon': dict(opt.pigeon), 'task': main_args.topic, 'errno': 0}

    outdir = os.path.join(main_args.out, user_code)
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    resdata['progress'] = 1.0
    _report_result(msgkey, resdata)
    frames, vid_fps = read_video(race_data(opt.video), rot=None)
    resdata['progress'] += 3.0
    _report_result(msgkey, resdata)

    def _model_strides_progress(x):
        resdata['progress'] = 4.0 + x * progress_strides_weight
        Logger.info(resdata['progress'])
        _report_result(msgkey, resdata)

    def _video_save_progress(x):
        resdata['progress'] = 29 + x * (0.95 - progress_strides_weight)
        Logger.info(resdata['progress'])
        _report_result(msgkey, resdata)

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
            fully_periodic=fully_periodic,
            progress_cb=_model_strides_progress)
    infer_time = time.time() - s_time
    Logger.info('model inference using time: %d' % infer_time)

    json_result = {}
    json_result['period'] = pred_period
    json_result['score'] = np.float(pred_score.numpy())
    json_result['stride'] = chosen_stride
    json_result['fps'] = vid_fps
    json_result['num_frames'] = len(frames)
    json_result['infer_time'] = infer_time
    frames_info = []
    spf = 1 / vid_fps # time second for per frame
    sum_counts = np.cumsum(per_frame_counts)
    for i, (in_period, p_count) in enumerate(zip(within_period, per_frame_counts)):
        frames_info.append({
            'image_id': '%d.jpg' % i,
            'at_time': round((i + 1) * spf, 3),
            'within_period': float(in_period),
            'pframe_counts': float(p_count),
            'cum_counts': sum_counts[i]
        })
    json_result['frames_period'] = frames_info

    prefix = 'https://raceai.s3.didiyunapi.com'
    if save_video:
        outfile = os.path.join(outdir, 'repnet_tf-target.mp4')
        create_count_video(frames, per_frame_counts, within_period, score=pred_score,
                fps=vid_fps, output_file=outfile, delay=1000 / vid_fps,
                vizualize_reps=True, progress_cb=_video_save_progress)
        mkvid_time = time.time() - s_time - infer_time
        json_result['mkvideo_time'] = mkvid_time 
        json_result['target_mp4'] = prefix + outfile
        resdata['target_mp4'] = prefix + outfile

    json_result_file = os.path.join(outdir, 'repnet_tf-results.json')
    with open(json_result_file, 'w') as fw:
        fw.write(json.dumps(json_result, indent=4))

    if not _DEBUG_:
        race_object_put(osscli, outdir, bucket_name='raceai')
    resdata['progress'] = 100.0
    resdata['target_json'] = prefix + json_result_file
    _report_result(msgkey, resdata)


if __name__ == "__main__":
    if not _DEBUG_:
        zmqsub.subscribe(main_args.topic)
        race_report_result('add_topic', main_args.topic)

    try:
        # Load model
        repnet_model = get_model(main_args.ckpt)

        if not _DEBUG_:
            while True:
                Logger.info('wait task')
                race_report_result('zmp_end', main_args.topic)
                zmq_cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
                race_report_result('zmp_run', f'{main_args.topic}:30')
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
                    "video": main_args.path,
                    'save_video': False
            }
            # "video": "https://raceai.s3.didiyunapi.com/data/media/videos/repnet_test.mp4"
            zmq_cfg = OmegaConf.create(zmq_cfg)
            Logger.info(zmq_cfg)
            inference(repnet_model, zmq_cfg)
    finally:
        if _DEBUG_:
            race_report_result('del_topic', main_args.topic)
        Logger.info('end')
