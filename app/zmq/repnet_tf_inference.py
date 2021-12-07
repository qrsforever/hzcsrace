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
import traceback

import matplotlib.pyplot as plt
import io
import requests

from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist # noqa
from matplotlib.colors import LogNorm
from utils import get_model, read_video, cal_rect_points
from repnet import get_counts, get_sims

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

ffmpeg_args = '-preset ultrafast -vcodec libx264 -pix_fmt yuv420p'

input_width = 112
input_height = 112


def _report_result(msgkey, resdata, errcode=0):
    if _RELEASE_:
        if errcode < 0:
            resdata['errno'] = errcode
            resdata['progress'] = 100
        race_report_result(msgkey, resdata)
        if msgkey[:2] == 'nb':
            race_report_result('zmp_run', f'{main_args.topic}_{msgkey}:220')
        else:
            race_report_result('zmp_run', f'{main_args.topic}:220')
    else:
        pass


def _detect_focus(msgkey, vfile, retrieve_count, conf_thresh, iou_thresh, box_size, progress_cb):
    msgkey = msgkey + '.det'
    reqdata = '''{
        "task": "zmq.yolov5.ladder.l.inference",
        "cfg": {
            "pigeon": {
                "msgkey": "%s",
                "user": "private",
                "uuid": "repnet_tf"
            },
            "data": {
                "class_name": "raceai.data.process.VideoDataLoader",
                "params": {
                    "data_source": "%s",
                    "dataset": {
                        "class_name": "raceai.data.VideoFramesDataset",
                        "params": {
                            "retrieve_count": %d
                        }
                    }
                }
            },
            "nms":{
                "conf_thres": %f,
                "iou_thres": %f
            }
        }
    }''' % (msgkey, vfile, retrieve_count, conf_thresh, iou_thresh)
    RACEURL = 'http://0.0.0.0:9119'
    API_INFERENCE = f'{RACEURL}/raceai/framework/inference'
    API_POPMSG = f'{RACEURL}/raceai/private/popmsg'
    json.loads(requests.get(url=f'{API_POPMSG}?key={msgkey}').text)
    json.loads(requests.post(url=API_INFERENCE, json=eval(reqdata)).text)
    for i in range(200):
        resdata = json.loads(requests.get(url=f'{API_POPMSG}?key={msgkey}').text)
        for item in resdata:
            detinfo = {
                'focus_skewing': True,
            }
            centers = []
            detect_box = {}
            for result in item['result']:
                frame_idx = int(result['image_path'].split('.')[0])
                imagew = result['image_width']
                imageh = result['image_height']
                box = result['predict_box']
                if len(box) > 0:
                    detect_box[frame_idx] = box[-1]
                    xyxy = box[-1]['xyxy'] # last is the max confidence score
                    centers.append((int(0.5 * (xyxy[0] + xyxy[2])), int(0.5 * (xyxy[1] + xyxy[3]))))
            centers = np.array(centers)
            if len(centers) < 0.2 * retrieve_count:
                Logger.warning('not detect focus box')
                return None
            detinfo['valid_count'] = len(centers)
            half_cnt = int(0.5 * len(centers))
            model = KMeans(n_clusters=3)
            model.fit_predict(centers[:half_cnt])
            centroid1 = model.cluster_centers_
            model.fit_predict(centers[half_cnt:])
            centroid2 = model.cluster_centers_

            # Logger.info(f'{centroid1} vs {centroid2}')

            dist = np.linalg.norm(centroid1.mean(axis=0) - centroid2.mean(axis=0))
            if dist > min(50, 0.5 * max(box_size)):
                detinfo['centroid_dist'] = dist
            else:
                detinfo['focus_skewing'] = False

            clusters = model.fit_predict(centers)
            most_idx = Counter(clusters).most_common()[0][0]
            center = [int(x) for x in model.cluster_centers_[most_idx]]

            detinfo['focus_box'] = [
                min(max(center[0] - box_size[0], 0), imagew - box_size[0]),
                min(max(center[1] - box_size[1], 0), imageh - box_size[1]),
                min(center[0] + box_size[0], imagew),
                min(center[1] + box_size[1], imageh)]
            Logger.info(detinfo)
            detinfo['detect_box'] = detect_box
            return detinfo
        progress_cb(100 * min(float(i * 15) / retrieve_count, 1))
        time.sleep(1)
    return None


def _denormal_image(x):
    x -= x.mean()
    x /= x.std()
    x *= 64
    x += 128
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def draw_osd_sim(sim, size=128):
    fig, ax = plt.subplots()
    plt.axis('off')
    fig.set_size_inches(size / 100.0, size / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.imshow(sim, cmap='hot', interpolation='nearest', norm=LogNorm())
    with io.BytesIO() as fw:
        plt.savefig(fw, dpi=100.0, bbox_inches=0)
        buffer_ = np.frombuffer(fw.getvalue(), dtype=np.uint8)
        plt.close()
        return cv2.imdecode(buffer_, cv2.IMREAD_COLOR)
    raise


def draw_osd_feat(feat, X, Y, width, height):
    W, H, _ = feat.shape
    osd_feat = np.zeros((H * Y, W * X))
    fk = 0
    for hi in range(Y):
        for wj in range(X):
            fmap = _denormal_image(feat[:, :, fk])
            osd_feat[hi * H:(hi + 1) * H, wj * W:(wj + 1) * W] = fmap
            fk += 1
    fig, ax = plt.subplots()
    plt.axis('off')
    fig.set_size_inches(width / 100.0, height / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.imshow(osd_feat, cmap='viridis', interpolation='nearest')
    with io.BytesIO() as fw:
        plt.savefig(fw, dpi=100.0, bbox_inches=0)
        buffer_ = np.frombuffer(fw.getvalue(), dtype=np.uint8)
        plt.close()
        return cv2.imdecode(buffer_, cv2.IMREAD_COLOR)
    raise


def inference(model, opt, resdata):
    msgkey = opt.pigeon.msgkey
    best_stride_video = False
    user_code = 'unkown'
    if 'user_code' in opt.pigeon:
        user_code = opt.pigeon.user_code
    mac = opt.pigeon.get('mac_addr', None)
    if mac:
        if mac == '00856405d389':
            opt.focus_box = [0.448, 0.41, 0.53, 0.54]
            opt.focus_box_repnum = 6
            opt.area_rate_threshold = 0.01
            opt.angle = 90
        elif mac == '00047dd87188':
            opt.focus_box_repnum = 2
            opt.area_rate_threshold = 0.01
            opt.angle = 90
        elif mac == '00232ee8876d':
            opt.area_rate_threshold = 0.002
            opt.focus_box = [0.45, 0.5, 0.65, 0.75]
            opt.focus_box_repnum = 3
            opt.reg_factor = 2
        elif mac == '002b359e3931':
            opt.focus_box = [0.25, 0.05, 0.58, 0.99]
            opt.area_rate_threshold = 0.01
            opt.focus_box_repnum = 2
    batch_size = 2
    if 'batch_size' in opt:
        batch_size = opt.batch_size
    threshold = 0.2
    if 'threshold' in opt:
        threshold = opt.threshold
    in_threshold = 0.5
    if 'in_threshold' in opt:
        in_threshold = opt.in_threshold
    strides = [5, 7, 9]
    debug_mode = False
    if 'strides' in opt:
        strides = list(opt.strides)
        if strides[0] == -1: # TODO for debug
            debug_mode = True
            strides = strides[1:]
    constant_speed = False
    if 'constant_speed' in opt:
        constant_speed = opt.constant_speed
    median_filter = True
    if 'median_filter' in opt:
        median_filter = opt.median_filter
    fully_periodic = False
    if 'fully_periodic' in opt:
        fully_periodic = opt.fully_periodic
    angle = None
    if 'angle' in opt and opt['angle'] != 0:
        angle = opt.angle
    reg_factor = None
    if 'reg_factor' in opt and opt['reg_factor'] != 1:
        reg_factor = opt['reg_factor']
    rm_still = True
    if 'rm_still' in opt:
        rm_still = opt.rm_still
    temperature = 13.544
    if 'temperature' in opt:
        temperature = opt.temperature
    model.temperature = temperature
    area_rate_thres = 0.0002
    if 'area_rate_threshold' in opt:
        area_rate_thres = opt.area_rate_threshold
    save_video = False
    if 'save_video' in opt:
        save_video = opt.save_video
    best_stride_video = True if debug_mode else False
    if 'best_stride_video' in opt:
        best_stride_video = opt.best_stride_video
    osd_feat = False
    if 'osd_feat' in opt:
        osd_feat = opt.osd_feat
    osd_sims = True if debug_mode else False
    if 'osd_sims' in opt:
        osd_sims = opt.osd_sims
    if osd_sims or osd_feat:
        best_stride_video = True

    #### focus
    detect_focus, retrieve_count, box_size = False, 1, (10, 10)
    conf_thresh, iou_thresh = 0.5, 0.5
    focus_box, focus_box_repnum = None, 1
    black_box, black_overlay = None, True if debug_mode else False
    if 'detect_focus' in opt:
        detect_focus = opt.detect_focus
    if not detect_focus:
        if 'focus_box' in opt:
            if 0 == opt.focus_box[0] and 0 == opt.focus_box[1] \
                    and 1 == opt.focus_box[2] and 1 == opt.focus_box[3]:
                # Logger.warning(f'error box: {opt.focus_box}')
                pass
            else:
                focus_box = opt.focus_box
        else:
            if 'center_rate' in opt:
                w_rate, h_rate = opt.center_rate
                if w_rate != 0 and h_rate != 0:
                    focus_box = [
                        (1 - w_rate) * 0.5, (1 - h_rate) * 0.5,
                        (1 + w_rate) * 0.5, (1 + h_rate) * 0.5,
                    ]
        if 'block_box' in opt or 'black_box' in opt:
            if 'black_box' in opt:
                if 0 == opt.black_box[0] and 0 == opt.black_box[1] \
                        and 0 == opt.black_box[2] and 0 == opt.black_box[3]:
                    pass
                else:
                    black_box = opt.black_box
            else:
                if 0 == opt.block_box[0] and 0 == opt.block_box[1] \
                        and 0 == opt.block_box[2] and 0 == opt.block_box[3]:
                    pass
                else:
                    black_box = opt.block_box
            if 'black_overlay' in opt:
                black_overlay = opt['black_overlay']
    else:
        if 'retrieve_count' in opt:
            retrieve_count = opt.retrieve_count
        if 'box_size' in opt:
            box_size = opt.box_size
            if isinstance(box_size, int):
                box_size = (box_size, box_size)
        if 'conf_thresh' in opt:
            conf_thresh = opt.conf_thresh
        if 'iou_thresh' in opt:
            iou_thresh = opt.iou_thresh
    if 'focus_box_repnum' in opt:
        focus_box_repnum = opt.focus_box_repnum

    if detect_focus:
        read_video_weight = 0.2
    else:
        read_video_weight = 0.3

    if save_video or best_stride_video:
        model_progress_weight = 0.30
    else:
        model_progress_weight = 0.68

    if msgkey[:2] == 'nb':
        ts_token = '%d' % time.time()
    else:
        ts_token = 'repnet_tf'
    outdir = os.path.join(main_args.out, user_code, ts_token)
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    # parse path info
    if _RELEASE_:
        if 'https://' in opt.video and 'didiyunapi.com' in opt.video:
            uri = opt.video[8:]
        else:
            _report_result(msgkey, resdata, errcode=-10)
            Logger.warning('video url invalid: %s' % opt.video)
            return
        video_file = race_data(opt.video.replace('s3.didiyunapi', 's3-internal.didiyunapi'))

        segs = uri.split('/')
        bucketname = segs[0].split('.')[0]
        oss_domain = 'https://%s' % segs[0]
        if 's3-internal' in oss_domain:
            oss_domain = oss_domain.replace('s3-internal', 's3')

        oss_path_counts = os.path.join('/', *segs[1:-2], 'counts')
        if msgkey[:2] == 'nb':
            oss_path = os.path.join('/', *segs[1:-2], 'outputs', segs[-1].split('.')[0], ts_token)
        else:
            oss_path = os.path.join('/', *segs[1:-2], 'outputs', segs[-1].split('.')[0], 'repnet_tf')
    else:
        oss_domain = 'file://'
        oss_path = '/tmp/debug/repent_tf'
        oss_path_counts = '/tmp/debug/repent_tf_counts'
        video_file = opt.video

    def _detect_focus_progress(x):
        resdata['progress'] = round(x * 0.1, 2)
        Logger.info(resdata['progress'])
        _report_result(msgkey, resdata)

    def _video_read_progress(x):
        if detect_focus:
            resdata['progress'] = round(10 + x * read_video_weight, 2)
        else:
            resdata['progress'] = round(x * read_video_weight, 2)
        _report_result(msgkey, resdata)

    def _model_strides_progress(x):
        resdata['progress'] = round(30 + x * model_progress_weight, 2)
        Logger.info(resdata['progress'])
        _report_result(msgkey, resdata)

    def _video_save_progress(x):
        resdata['progress'] = round(60 + x * 0.4, 2)
        Logger.info(resdata['progress'])
        _report_result(msgkey, resdata)

    resdata['progress'] = 0.0
    _report_result(msgkey, resdata)
    try:
        if detect_focus:
            Logger.info('detect focus...')
            detinfo = _detect_focus(msgkey, video_file, retrieve_count, conf_thresh, iou_thresh, box_size, _detect_focus_progress)
            if detinfo:
                focus_box = detinfo['focus_box']

        frames, vid_fps, still_frames = read_video(
                video_file, width=input_width, height=input_height, rot=angle,
                black_box=black_box, focus_box=focus_box, focus_box_repnum=focus_box_repnum,
                progress_cb=_video_read_progress,
                rm_still=rm_still, area_rate_thres=area_rate_thres)
    except Exception:
        _report_result(msgkey, resdata, errcode=-20)
        Logger.warning('read video error: %s' % opt.video)
        Logger.error(traceback.format_exc(limit=3))
        os.remove(video_file)
        return
    if len(frames) <= 64:
        _report_result(msgkey, resdata, errcode=-21)
        Logger.warning('read video error: %s num_frames[%d]' % (opt.video, len(frames)))
        os.remove(video_file)
        return

    _report_result(msgkey, resdata)

    s_time = time.time()

    (pred_period, pred_score,
            within_period, per_frame_counts,
            chosen_stride, final_embs, feature_maps) = get_counts(
            model,
            frames,
            strides=strides,
            batch_size=batch_size,
            threshold=threshold,
            within_period_threshold=in_threshold,
            constant_speed=constant_speed,
            median_filter=median_filter,
            fully_periodic=fully_periodic,
            osd_feat=osd_feat,
            progress_cb=_model_strides_progress)
    infer_time = time.time() - s_time
    Logger.info('model inference using time: %d, chosen_stride:%d' % (infer_time, chosen_stride))

    all_frames_count = len(frames) + len(still_frames)
    is_still_frames = [False] * all_frames_count

    final_within_period = [.0] * all_frames_count
    final_per_frame_counts = [.0] * all_frames_count
    i, j = 0, 0
    for k in range(all_frames_count):
        if j < len(still_frames) and k == still_frames[j][0]:
            is_still_frames[k] = True
            j += 1
        elif i < len(frames):
            final_within_period[k] = within_period[i]
            final_per_frame_counts[k] = per_frame_counts[i]
            i += 1
        else:
            _report_result(msgkey, resdata, errcode=-30)
            Logger.warning('frames count invalid: %d vs %d vs %d' % (i, j, k))
            os.remove(video_file)
            return

    within_period = final_within_period
    per_frame_counts = np.asarray(final_per_frame_counts)
    if reg_factor:
        per_frame_counts = reg_factor * per_frame_counts
    sum_counts = np.cumsum(per_frame_counts)

    # del frames

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
            'cum_counts': float(sum_counts[i])
        })
    json_result['frames_period'] = frames_info

    if osd_sims:
        embs_sims = get_sims(final_embs, temperature=temperature)
        embs_sims = np.squeeze(embs_sims, -1)
        Logger.info(f'embs_sims.shape: {embs_sims.shape}')

    detect_box = None
    if detect_focus and detinfo:
        detect_box = detinfo.pop('detect_box')
        resdata['detinfo'] = detinfo

    del within_period, per_frame_counts, final_embs

    if save_video or best_stride_video:
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fmt = cv2.VideoWriter_fourcc(*'mp4v')
        if save_video:
            mp4v_file = os.path.join(outdir, 'tmp_target.mp4')
            h264_file = os.path.join(outdir, 'target.mp4')
            vid = cv2.VideoWriter(mp4v_file, fmt, fps, (width, height))
        if best_stride_video:
            mp4v_stride_file = os.path.join(outdir, 'tmp-target-stride.mp4')
            h264_stride_file = os.path.join(outdir, 'target-stride.mp4')
            stride_vid = cv2.VideoWriter(mp4v_stride_file, fmt, fps, (width, height))

        if black_box is not None:
            bx1, by1, bx2, by2 = cal_rect_points(width, height, black_box)
        if focus_box is not None:
            fx1, fy1, fx2, fy2 = cal_rect_points(width, height, focus_box)
        if cap.isOpened():
            cur_db = None
            idx, valid_idx = 0, 0
            th = int(0.08 * height)
            osd, osd_size, alpha = 0, int(width*0.25), 0.8 # noqa
            osd_blend = None
            while True:
                success, frame_bgr = cap.read()
                if not success:
                    break
                if black_box is not None:
                    if black_overlay:
                        frame_bgr[by1:by2, bx1:bx2, :] = 0
                    else:
                        cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 0, 0), 2)
                    cv2.putText(frame_bgr,
                            '%d,%d' % (bx1, by1),
                            (bx1 + 2, by1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
                    cv2.putText(frame_bgr,
                            '%d,%d' % (bx2, by2),
                            (bx2 - 65, by2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
                cv2.rectangle(frame_bgr, (0, 0), (width, th), (0, 0, 0), -1)
                cv2.rectangle(frame_bgr, (0, height - th), (width, height), (0, 0, 0), -1)
                try:
                    if osd_sims and not is_still_frames[idx] \
                            and valid_idx % (chosen_stride * model.num_frames) == 0:
                        Logger.info(f'valid_idx: {valid_idx} idx: {idx} osd: {osd}')
                        osd_blend = draw_osd_sim(embs_sims[osd], osd_size)
                        # osd_blend = cv2.addWeighted(
                        #         osd_blend, alpha,
                        #         frame_bgr[:osd_size, width - osd_size:, :], 1 - alpha,
                        #         0, frame_bgr)
                        osd += 1
                    if osd_blend is not None:
                        frame_bgr[:osd_size, width - osd_size:, :] = osd_blend
                except Exception as err:
                    Logger.info(err)
                    Logger.error(traceback.format_exc(limit=3))
                cv2.putText(frame_bgr,
                        '%dX%d %.1f S:%d C:%.1f/%.1f T:%.3f %s' % (width, height,
                            fps, chosen_stride, sum_counts[idx], sum_counts[-1],
                            temperature, 'ST' if is_still_frames[idx] else ''),
                        (2, int(0.06 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 if height < 500 else 2,
                        (255, 255, 255), 2)
                if focus_box is not None:
                    cv2.putText(frame_bgr,
                            '%d,%d' % (fx1, fy1),
                            (fx1 + 2, fy1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame_bgr,
                            '%d,%d' % (fx2, fy2),
                            (fx2 - 65, fy2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(frame_bgr, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

                if osd_sims and valid_idx < len(frames):
                    frame_bgr[height - input_height - 10:, :input_width + 10, :] = 222
                    frame_bgr[height - input_height - 5:height - 5, 5:input_width + 5, :] = frames[valid_idx][:,:,::-1]

                if detect_box:
                    if idx in detect_box:
                        cur_db = detect_box[idx]
                    if cur_db:
                        xyxy = cur_db['xyxy']
                        cv2.rectangle(frame_bgr, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (211, 211, 211), 2)
                        cv2.putText(frame_bgr, cur_db['conf'],
                                (xyxy[0] + 2, xyxy[1] + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(frame_bgr,
                        'S:[%s] P:%d %s %s F:%.3f' % (','.join(map(str, strides)),
                            focus_box_repnum,
                            'A:%.4f' % area_rate_thres if rm_still else '',
                            'R:%.1f' % angle if angle else '',
                            float(len(frames)) / all_frames_count),
                        (input_width + 12, height - int(th * 0.35)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 if height < 500 else 2,
                        (255, 255, 255), 2)

                if idx % 100 == 0:
                    _video_save_progress((90 * float(idx)) / all_frames_count)

                if save_video:
                    vid.write(frame_bgr)
                if best_stride_video and not is_still_frames[idx] \
                        and valid_idx % chosen_stride == 0:
                    if osd_feat:
                        fmap = draw_osd_feat(feature_maps[0][int(valid_idx / chosen_stride)], 7, 3, 450, 200)
                        frame_bgr[th:th + 200, 0:450, :] = fmap
                        fmap = draw_osd_feat(feature_maps[2][int(valid_idx / chosen_stride)], 8, 1, 500, 70)
                        frame_bgr[-th - 80: -th - 10, width - 500:, :] = fmap
                    stride_vid.write(frame_bgr)
                if not is_still_frames[idx]:
                    valid_idx += 1
                idx += 1
        cap.release()
        _video_save_progress(91)
        if save_video:
            vid.release()
            os.system(f'ffmpeg -an -i {mp4v_file} {ffmpeg_args} {h264_file} 2>/dev/null')
            os.remove(mp4v_file)
            resdata['target_mp4'] = oss_domain + os.path.join(oss_path, os.path.basename(h264_file))
            json_result['target_mp4'] = resdata['target_mp4']
        _video_save_progress(94)
        if best_stride_video:
            stride_vid.release()
            os.system(f'ffmpeg -an -i {mp4v_stride_file} {ffmpeg_args} {h264_stride_file} 2>/dev/null')
            os.remove(mp4v_stride_file)
            resdata['stride_mp4'] = oss_domain + os.path.join(oss_path, os.path.basename(h264_stride_file))
            json_result['stride_mp4'] = resdata['stride_mp4']
        _video_save_progress(96)
        if osd_sims:
            np.save(os.path.join(outdir, 'embs_sims.npy'), embs_sims)
            resdata['embs_sims'] = oss_domain + os.path.join(oss_path, 'embs_sims.npy')
        mkvid_time = time.time() - s_time - infer_time
        json_result['mkvideo_time'] = mkvid_time
        _video_save_progress(98)

        if debug_mode:
            prefix_map = [h264_stride_file, '/'.join(opt.video[8:].split('/')[1:])]
            race_object_put(osscli, h264_stride_file,
                    bucket_name=bucketname, prefix_map=prefix_map)

    resdata['sumcnt'] = round(float(sum_counts[-1]), 2)

    json_result_file = os.path.join(outdir, 'results.json')
    with open(json_result_file, 'w') as fw:
        json.dump(json_result, fw, indent=4)

    json_config_file = os.path.join(outdir, 'config.json')
    with open(json_config_file, 'w') as fw:
        opt.sumcnt = resdata['sumcnt']
        json.dump(OmegaConf.to_container(opt), fw, indent=4)

    # oss_path_counts
    try:
        ts = os.path.basename(opt.video)[:-4]
        Logger.info(f'touch {oss_path_counts}/{ts}_{resdata["sumcnt"]}.json')
        race_object_put_jsonconfig(
                osscli, [], f'{oss_path_counts}/{ts}_{resdata["sumcnt"]}.json',
                bucket_name=bucketname)
    except Exception as err:
        Logger.warning('%s' % err)

    del frames, still_frames, json_result, frames_info
    if osd_sims:
        del embs_sims
    if osd_feat:
        del feature_maps
    os.remove(video_file)

    if _RELEASE_:
        race_object_remove(osscli, outdir[1:] + '/', bucket_name=bucketname)
        prefix_map = [outdir, oss_path]
        race_object_put(osscli, outdir,
                bucket_name=bucketname, prefix_map=prefix_map)

    _video_save_progress(100)
    resdata['progress'] = 100.0
    resdata['target_json'] = oss_domain + os.path.join(oss_path, os.path.basename(json_result_file))
    Logger.info(json.dumps(resdata))
    _report_result(msgkey, resdata)

    del resdata


if __name__ == "__main__":
    if _RELEASE_:
        zmqsub.subscribe(main_args.topic)
        race_report_result('add_topic', main_args.topic)
        Logger.info('main_args.topic: %s' % main_args.topic)

    os.system('rm /tmp/*.mp4 2>/dev/null')
    os.system('rm /tmp/tmp*.py 2>/dev/null')

    try:
        # Load model
        repnet_model = get_model(main_args.ckpt)

        if _RELEASE_:
            while True:
                Logger.info('wait task')
                zmq_cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
                zmq_cfg = OmegaConf.create(zmq_cfg)
                Logger.info(zmq_cfg)
                if 'pigeon' not in zmq_cfg:
                    continue
                Logger.info(zmq_cfg.pigeon)
                resdata = {'pigeon': dict(zmq_cfg.pigeon), 'task': main_args.topic, 'errno': 0}
                if zmq_cfg.pigeon.msgkey[:2] == 'nb':
                    race_report_result('zmp_run', f'{main_args.topic}_{zmq_cfg.pigeon.msgkey}:30')
                else:
                    race_report_result('zmp_run', f'{main_args.topic}:30')
                try:
                    inference(repnet_model, zmq_cfg, resdata)
                except Exception as err:
                    if 'OOM' in str(err):
                        _report_result(zmq_cfg.pigeon.msgkey, resdata, errcode=-9)
                        raise err
                    Logger.error(traceback.format_exc(limit=3))
                    _report_result(zmq_cfg.pigeon.msgkey, resdata, errcode=-99)
                    os.system('rm /tmp/*.mp4 2>/dev/null')
                    os.system('rm /tmp/tmp*.py 2>/dev/null')
                if zmq_cfg.pigeon.msgkey[:2] == 'nb':
                    race_report_result('zmp_end', f'{main_args.topic}_{zmq_cfg.pigeon.msgkey}')
                else:
                    race_report_result('zmp_end', main_args.topic)
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
        Logger.error(traceback.format_exc(limit=3))

    finally:
        if _RELEASE_:
            race_report_result('del_topic', main_args.topic)
        Logger.info('end')
