"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import time
import zmq

import torch
from tqdm import tqdm

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.writer import DataWriter
from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt

from omegaconf import OmegaConf
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)
from raceai.utils.misc import (
        race_oss_client,
        race_object_put,
        race_report_result, 
        race_data)


race_set_loglevel('info')
race_set_logfile('/tmp/raceai-alphapose.log')

_DEBUG_ = False

context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))

osscli = race_oss_client(bucket_name='raceai')

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--topic', type=str, default='123',
                    help='zmq topic')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="/raceai/data/tmp/alphapose/output/halpe26")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def print_finish_info():
    Logger.info('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        Logger.info('===========================> Rendering remaining images in the queue...')
        Logger.info('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1


def result_report(resdata, res):
    im_name = res['imgname']
    for human in res['result']:
        result = {}
        keypoints = []
        result['image_id'] = int(os.path.basename(im_name).split('.')[0].split('_')[-1])
        result['category_id'] = 1

        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        pro_scores = human['proposal_score']
        for n in range(kp_scores.shape[0]):
            keypoints.append(float(kp_preds[n, 0]))
            keypoints.append(float(kp_preds[n, 1]))
            keypoints.append(float(kp_scores[n]))
        result['keypoints'] = keypoints
        result['score'] = float(pro_scores)
        if 'box' in human.keys():
            result['box'] = human['box']
        if 'idx' in human.keys():
            result['idx'] = human['idx']
        resdata['result'].append(result)
        race_report_result(resdata['pigeon']['msgkey'], resdata)


def inference(pose_model, det_model, opt):
    msgkey = args.topic
    if 'msgkey' in opt.pigeon:
        msgkey = opt.pigeon.msgkey
    else:
        opt.pigeon.msgkey = msgkey

    user_code = 'unkown'
    if 'user_code' in opt.pigeon:
        user_code = opt.pigeon.user_code

    outputpath = os.path.join(args.outputpath, user_code)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    if 'qsize' in opt:
        args.qsize = opt.qsize
    else:
        args.qsize = 1024

    if 'save_img' in opt:
        args.save_img = opt.save_img
    else:
        args.save_img = False

    if 'save_video' in opt:
        args.save_video = opt.save_video
    else:
        args.save_video = False

    if 'vis_fast' in opt:
        args.vis_fast = opt.vis_fast
    else:
        args.vis_fast = True

    report_cb = None
    if 'report_kps' in opt and opt['report_kps']:
        report_cb = result_report 

    resdata = {'pigeon': dict(opt.pigeon), 'task': args.topic, 'errno': 0}

    # Load detection loader
    det_loader = DetectionLoader(race_data(opt.video),
            det_model, cfg, args, batchSize=args.detbatch, mode='video', queueSize=args.qsize)
    det_worker = det_loader.start() # noqa

    # Init data writer
    if args.save_video:
        video_save_opt['savepath'] = os.path.join(outputpath, 'alphapose-target.mp4')
        video_save_opt.update(det_loader.videoinfo)
        writer = DataWriter(cfg, args, save_video=True,
                video_save_opt=video_save_opt, queueSize=args.qsize,
                result_callback=report_cb, resdata=resdata).start()
    else:
        writer = DataWriter(cfg, args, save_video=False, queueSize=args.qsize,
                result_callback=report_cb, resdata=resdata).start()

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True, disable=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            # heart beating
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.pose_track:
                    boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                Logger.info('[%d]/[%d]' % (i + 1, data_len))
                if i % 50 == 0:
                    writer.resdata['progress'] = float(96 * (i + 1) / data_len)
                    if not _DEBUG_:
                        race_report_result(msgkey, resdata)
                    race_report_result('zmp_run', args.topic)

        print_finish_info()
        while(writer.running()):
            time.sleep(1)
            Logger.info('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            race_report_result('zmp_run', args.topic)
        writer.stop()
        det_loader.stop()

        if not _DEBUG_:
            writer.resdata['progress'] = 100.0
            prefix = 'https://raceai.s3.didiyunapi.com'
            resdata['target_json'] = prefix + os.path.join(outputpath, 'alphapose-results.json')
            if args.save_video:
                resdata['target_mp4'] = prefix + os.path.join(outputpath, 'alphapose-target.mp4')
            race_object_put(osscli, outputpath, bucket_name='raceai')
            race_report_result(msgkey, resdata)

    except Exception as e:
        Logger.info(repr(e))
        Logger.info('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                Logger.info('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()
        raise


if __name__ == "__main__":

    if not _DEBUG_:
        race_report_result('add_topic', args.topic)
        zmqsub.subscribe(args.topic)

    try:
        # Load model
        Logger.info('Loading detector model...')
        det_model = get_detector(args)
        det_model.load_model()
        pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        Logger.info('Loading pose model from %s...' % (args.checkpoint,))
        pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        if args.pose_track:
            tracker = Tracker(tcfg, args)
        if len(args.gpus) > 1:
            pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
        else:
            pose_model.to(args.device)
        pose_model.eval()

        if not _DEBUG_:
            while True:
                Logger.info('wait task')
                zmq_cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
                zmq_cfg = OmegaConf.create(zmq_cfg)
                Logger.info(zmq_cfg)
                if 'pigeon' not in zmq_cfg:
                    continue
                Logger.info(zmq_cfg.pigeon)
                inference(pose_model, det_model, zmq_cfg)
                time.sleep(0.01)
        else:
            zmq_cfg = {
                "pigeon": {"msgkey": "123"},
                "video": "https://raceai.s3.didiyunapi.com/data/media/videos/alphapose_test.mp4"
            }
            zmq_cfg = OmegaConf.create(zmq_cfg)
            Logger.info(zmq_cfg)
            inference(pose_model, det_model, zmq_cfg)
    except Exception:
        pass
    finally:
        if not _DEBUG_:
            race_report_result('del_topic', args.topic)
        Logger.info('end')
