from repnet import get_repnet_model
from tqdm import tqdm
import cv2
import numpy as np
import requests
import math
import os

DEBUG = False


def cal_rect_points(w, h, box):
    if box[0] < 1.0 and box[1] < 1.0 and box[2] <= 1.0 and box[3] <= 1.0:
        x1, y1 = int(w * box[0]), int(h * box[1])
        x2, y2 = int(w * box[2]), int(h * box[3])
    else:
        x1, y1 = box[0], box[1]
        x2, y2 = box[2], box[3]
    return x1, y1, x2, y2


def read_video(
        video_filename, width=224, height=224,
        rot=None, black_box=None, focus_box=None, focus_box_repnum=1,
        progress_cb=None, rm_still=False, area_rate_thres=0.0025):
    """Read video from file."""
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=n_frames, desc=f"Getting frames from {video_filename} {len ...")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if black_box is not None:
        black_x1, black_y1, black_x2, black_y2 = cal_rect_points(w, h, black_box)
    if focus_box is not None:
        focus_x1, focus_y1, focus_x2, focus_y2 = cal_rect_points(w, h, focus_box)
        w = focus_x2 - focus_x1
        h = focus_y2 - focus_y1
    if rm_still: # remove still frames
        pre_frame = None
        area_thres = math.ceil(area_rate_thres * w * h)
        print(f'rm_still: {area_thres} = ceil({area_rate_thres} * {w} * {h})')
    frames = []
    still_frames = []

    if DEBUG:
        debug_file = os.path.join('/raceai/data', 'debug_file.mp4')
        debug_vid = cv2.VideoWriter(debug_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if cap.isOpened():
        frame_idx = 0
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            keep_flag = False
            if black_box is not None:
                frame_bgr[black_y1:black_y2, black_x1:black_x2, :] = 0
            if focus_box is not None:
                frame_bgr = frame_bgr[focus_y1:focus_y2, focus_x1:focus_x2, :]
                # frame_bgr = cv2.normalize(frame_bgr, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)
                # frame_bgr = frame_bgr * float(2)
                # frame_bgr = np.round(frame_bgr)
                # frame_bgr[frame_bgr > 255] = 255
                # frame_bgr = frame_bgr.astype(np.uint8)
                # image color reverse
                # frame_bgr = 255 - frame_bgr
            if rm_still:
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if pre_frame is not None:
                    frame_tmp = cv2.absdiff(frame_gray, pre_frame)
                    frame_tmp = cv2.threshold(frame_tmp, 20, 255, cv2.THRESH_BINARY)[1]
                    frame_tmp = cv2.dilate(frame_tmp, None, iterations=2)
                    contours, _ = cv2.findContours(frame_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if len(contours) > 0:
                        # contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
                        # if cv2.contourArea(contours[0]) > area_thres:
                        #     cv2.drawContours(frame_bgr, [contours[0]], 0, (255,255,255), 3)
                        #     keep_flag = True
                        for contour in contours:
                            if cv2.contourArea(contour) > area_thres:
                                keep_flag = True
                                break
                pre_frame = frame_gray

            if focus_box is not None:
                if focus_box_repnum > 1:
                    frame_bgr = np.hstack([frame_bgr] * focus_box_repnum)
                    frame_bgr = np.vstack([frame_bgr] * focus_box_repnum)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (width, height))
            if rot:
                # frame_rgb = cv2.rotate(frame_rgb, rot)
                M = cv2.getRotationMatrix2D(center=(int(width / 2), int(height / 2)), angle=rot, scale=1.0)
                frame_rgb = cv2.warpAffine(frame_rgb, M, (width, height))
            if frame_idx and rm_still and not keep_flag:
                still_frames.append((frame_idx, frame_rgb))
            else:
                frames.append(frame_rgb)
                if DEBUG:
                    debug_vid.write(frame_bgr)

            frame_idx += 1

            if progress_cb:
                if frame_idx % 100 == 0:
                    progress_cb((100 * float(frame_idx)) / n_frames)

            del frame_bgr

            pbar.update()
    pbar.close()
    if DEBUG:
        debug_vid.release()
    print(n_frames, 'vs', len(frames))
    # if rm_still:
    #     fps = len(frames) * fps / n_frames
    frames = np.asarray(frames)
    still_frames = np.asarray(still_frames)
    return frames, fps, still_frames


def wget(url, path):
    """
    Source from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    Args:
        url (str):
        path (str):

    Returns:

    """
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {url} to {path} ...")
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def get_model(weight_root):
    os.makedirs(weight_root, exist_ok=True)

    weight_urls = [
        "https://storage.googleapis.com/repnet_ckpt/checkpoint",
        "https://storage.googleapis.com/repnet_ckpt/ckpt-88.data-00000-of-00002",
        "https://storage.googleapis.com/repnet_ckpt/ckpt-88.data-00001-of-00002",
        "https://storage.googleapis.com/repnet_ckpt/ckpt-88.index"
    ]
    for url in weight_urls:
        path = f"{weight_root}/{url.split('/')[-1]}"
        if os.path.isfile(path):
            continue

        wget(url, path)

    return get_repnet_model(weight_root)
