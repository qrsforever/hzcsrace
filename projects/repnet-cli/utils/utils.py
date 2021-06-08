from repnet import get_repnet_model
from tqdm import tqdm
import cv2
import numpy as np
import requests
import os


def read_video(video_filename, width=224, height=224, rot=None, rm_still=False, area_rate_thres=0.0625):
    """Read video from file."""
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=n_frames, desc=f"Getting frames from {video_filename} ...")

    if rm_still: # remove still frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pre_frame = None
        area_thres = area_rate_thres * width * height
    frames = []
    still_frames = []
    if cap.isOpened():
        frame_idx = 0
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            keep_flag = False
            if rm_still:
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if pre_frame is not None:
                    frame_tmp = cv2.absdiff(frame_gray, pre_frame)
                    frame_tmp = cv2.threshold(frame_tmp, 20, 255, cv2.THRESH_BINARY)[1]
                    frame_tmp = cv2.dilate(frame_tmp, None, iterations=2)
                    contours, _ = cv2.findContours(frame_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for contour in contours:
                        if cv2.contourArea(contour) > area_thres:
                            keep_flag = True
                            break
                pre_frame = frame_gray
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (width, height))
            if rot:
                frame_rgb = cv2.rotate(frame_rgb, rot)
            if rm_still and not keep_flag:
                still_frames.append((frame_idx, frame_rgb))
            else:
                frames.append(frame_rgb)
            frame_idx += 1

            pbar.update()
    pbar.close()
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
