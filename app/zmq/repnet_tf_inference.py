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
import cv2

from utils import get_model, read_video
from repnet import get_counts

from omegaconf import OmegaConf
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)
from raceai.utils.misc import ( # noqa
        race_oss_client,
        race_object_put,
        race_object_remove,
        race_report_result,
        race_data)

race_set_loglevel('info')
race_set_logfile('/tmp/raceai-repnet_tf.log')

_RELEASE_ = True
context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))

osscli = race_oss_client()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", default='', type=str, help="Input video file path or root.")
parser.add_argument("--topic", type=str, default="a.b.c", help="topic")
parser.add_argument("--out", default="/tmp/export", help="Output video file path or root")
parser.add_argument("--ckpt", default="/tmp/weights", type=str, help="Checkpoint weights root.")
main_args = parser.parse_args()


def _report_result(msgkey, resdata, errcode=0):
    if _RELEASE_:
        if errcode < 0:
            resdata['errno'] = errcode
            resdata['progress'] = 100
        race_report_result(msgkey, resdata)
        race_report_result('zmp_run', f'{main_args.topic}:120')
    else:
        pass


def inference(model, opt, resdata):
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
    strides = [3, 5, 7, 9, 13]
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
    save_video = False
    if 'save_video' in opt:
        save_video = opt.save_video
    rm_still = False
    if 'rm_still' in opt:
        rm_still = opt.rm_still
    area_rate_thres = 0.002
    if 'area_rate_threshold' in opt:
        area_rate_thres = opt.area_rate_threshold
    best_stride_video = True
    if 'best_stride_video' in opt:
        best_stride_video = opt.best_stride_video

    if save_video or best_stride_video:
        model_progress_weight = 0.40
    else:
        model_progress_weight = 0.78

    outdir = os.path.join(main_args.out, user_code, 'repnet_tf')
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    # parse path info
    if _RELEASE_:
        if 'https://' in opt.video and 's3.didiyun' in opt.video:
            uri = opt.video[8:]
        else:
            _report_result(msgkey, resdata, errcode=-10)
            raise RuntimeError('video url invalid: %s' % opt.video)

        segs = uri.split('/')
        bucketname = segs[0].split('.')[0]
        oss_domain = 'https://%s' % segs[0]
        oss_path = os.path.join('/', *segs[1:-2], 'outputs', segs[-1].split('.')[0], 'repnet_tf')
    else:
        oss_domain = 'file://'
        oss_path = '/tmp/debug/repent_tf'

    def _video_read_progress(x):
        resdata['progress'] = x * 0.2
        # Logger.info(resdata['progress'])
        _report_result(msgkey, resdata)

    def _model_strides_progress(x):
        resdata['progress'] = 20 + x * model_progress_weight
        Logger.info(resdata['progress'])
        _report_result(msgkey, resdata)

    def _video_save_progress(x):
        resdata['progress'] = 60 + x * 0.38
        Logger.info(resdata['progress'])
        _report_result(msgkey, resdata)

    resdata['progress'] = 0.0
    _report_result(msgkey, resdata)
    try:
        frames, vid_fps, still_frames = read_video(
                race_data(opt.video), width=112, height=112, rot=None,
                progress_cb=_video_read_progress,
                rm_still=rm_still, area_rate_thres=area_rate_thres)
    except:
        _report_result(msgkey, resdata, errcode=-20)
        raise RuntimeError('read video error: %s' % opt.video)
    if len(frames) <= 64:
        _report_result(msgkey, resdata, errcode=-21)
        raise RuntimeError('read video error: %s num_frames[%d]' % (opt.video, len(frames)))

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

    all_frames_count = len(frames) + len(still_frames)
    is_still_frames = [False] * all_frames_count
    stride_skip_mask = [True] * all_frames_count

    # final_frames = [None] * all_frames_count
    final_within_period = [.0] * all_frames_count
    final_per_frame_counts = [.0] * all_frames_count
    i, j = 0, 0
    for k in range(all_frames_count):
        if j < len(still_frames) and k == still_frames[j][0]:
            # final_frames[k] = still_frames[j][1]
            is_still_frames[k] = True
            j += 1
        elif i < len(frames):
            # final_frames[k] = frames[i]
            final_within_period[k] = within_period[i]
            final_per_frame_counts[k] = per_frame_counts[i]
            if i % chosen_stride == 0:
                stride_skip_mask[k] = False
            i += 1
        else:
            _report_result(msgkey, resdata, errcode=-30)
            raise RuntimeError('%d vs %d vs %d' % (i, j, k))
    # frames = final_frames
    within_period = final_within_period
    per_frame_counts = np.asarray(final_per_frame_counts)
    sum_counts = np.cumsum(per_frame_counts)

    del frames

    json_result = {}
    json_result['period'] = pred_period
    json_result['score'] = np.float(pred_score.numpy())
    json_result['stride'] = chosen_stride
    json_result['fps'] = vid_fps
    json_result['num_frames'] = all_frames_count
    if rm_still:
        json_result['num_still_frames'] = len(still_frames)
        json_result['area_rate_threshold'] = area_rate_thres
    json_result['infer_time'] = infer_time
    frames_info = []
    spf = 1 / vid_fps # time second for per frame
    for i, (in_period, p_count, is_still) in enumerate(zip(within_period, per_frame_counts, is_still_frames)):
        frames_info.append({
            'image_id': '%d.jpg' % i,
            'at_time': round((i + 1) * spf, 3),
            'is_still': is_still,
            'within_period': float(in_period),
            'pframe_counts': float(p_count),
            'cum_counts': sum_counts[i]
        })
    json_result['frames_period'] = frames_info

    if save_video or best_stride_video:
        cap = cv2.VideoCapture(opt.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fmt = cv2.VideoWriter_fourcc(*'mp4v')
        if save_video:
            outfile = os.path.join(outdir, 'target.mp4')
            vid = cv2.VideoWriter(outfile, fmt, fps, (width, height))
            resdata['target_mp4'] = oss_domain + os.path.join(oss_path, os.path.basename(outfile))
            json_result['target_mp4'] = resdata['target_mp4']
        if best_stride_video:
            outfile = os.path.join(outdir, 'target-stride.mp4')
            stride_vid = cv2.VideoWriter(outfile, fmt, fps, (width, height))
            resdata['stride_mp4'] = oss_domain + os.path.join(oss_path, os.path.basename(outfile))
            json_result['stride_mp4'] = resdata['stride_mp4']
        if cap.isOpened():
            idx = 0
            while True:
                success, frame_bgr = cap.read()
                if not success:
                    break
                cv2.putText(frame_bgr,
                        'count: %.3f' % sum_counts[idx], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0, 255), 2)
                if idx % 100 == 0:
                    _video_save_progress(round((100 * float(idx)) / all_frames_count, 2))
                if save_video:
                    vid.write(frame_bgr)
                if best_stride_video and not stride_skip_mask[idx]:
                    stride_vid.write(frame_bgr)
                idx += 1
        cap.release()
        if save_video:
            vid.release()
        if best_stride_video:
            stride_vid.release()
        mkvid_time = time.time() - s_time - infer_time
        json_result['mkvideo_time'] = mkvid_time

    json_result_file = os.path.join(outdir, 'results.json')
    with open(json_result_file, 'w') as fw:
        fw.write(json.dumps(json_result, indent=4))

    del json_result, frames_info

    if _RELEASE_:
        race_object_remove(osscli, outdir[1:] + '/', bucket_name=bucketname)
        prefix_map = [outdir, oss_path]
        Logger.info(prefix_map)
        result = race_object_put(osscli, outdir,
                bucket_name=bucketname, prefix_map=prefix_map)
    resdata['progress'] = 100.0
    resdata['target_json'] = oss_domain + os.path.join(oss_path, os.path.basename(json_result_file))
    Logger.info(json.dumps(resdata))
    _report_result(msgkey, resdata)


if __name__ == "__main__":
    if _RELEASE_:
        zmqsub.subscribe(main_args.topic)
        race_report_result('add_topic', main_args.topic)

    try:
        # Load model
        repnet_model = get_model(main_args.ckpt)

        if _RELEASE_:
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
                resdata = {'pigeon': dict(zmq_cfg.pigeon), 'task': main_args.topic, 'errno': 0}
                try:
                    inference(repnet_model, zmq_cfg, resdata)
                except Exception as err:
                    if 'OOM' in str(err):
                        _report_result(zmq_cfg.pigeon.msgkey, resdata, errcode=-9)
                        raise err
                    _report_result(zmq_cfg.pigeon.msgkey, resdata, errcode=-99)
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
            resdata = {'pigeon': zmq_cfg['pigeon'], 'task': main_args.topic, 'errno': 0}
            inference(repnet_model, zmq_cfg)
    except Exception as err: 
        Logger.error(err)
    finally:
        if _RELEASE_:
            race_report_result('del_topic', main_args.topic)
        Logger.info('end')
