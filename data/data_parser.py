import pathlib
from typing import Optional, Union

import numpy as np
import imageio
import json
import cv2
from tqdm import tqdm

from camera.camera_model import CameraModel
from data.shm_helper import NRDataSHMArrayWriter


def parse_load_nr_data(
        basedir: Union[pathlib.Path, str],
        splits: Optional[list] = None,
        half_res: bool = False,
        white_background: bool = True
):
    """
    Parse and load NR data into shared memory.
    :param basedir: NR data directory.
    :param splits: list of splits to load. Default to ['train', 'val', 'test']
    :param half_res: whether to load half resolution images.
    :param white_background: whether to load images with white background.
    :return: NRDataSHMArrayWriter object
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    if type(basedir) == str:
        basedir = pathlib.Path(basedir)

    # load metadata
    metas = {}
    for s in splits:
        with open(basedir / 'transforms_{}.json'.format(s), 'r') as fp:
            metas[s] = json.load(fp)
    num_image_per_split = [len(metas[s]['frames']) for s in splits]
    total_image_num = sum(num_image_per_split)

    zn = 3.
    zf = 10.
    meta = metas[splits[0]]
    if 'camera_near' in meta:
        zn = meta['camera_near']
    if 'camera_far' in meta:
        zf = meta['camera_far']

    # load first image to get image size
    first_image = imageio.v3.imread(basedir / (metas[splits[0]]['frames'][0]['file_path'] + '.png'))
    H, W = first_image.shape[:2]

    # load camera intrinsic, single camera model
    if 'camera_intrinsics' in meta:
        intrinsics = meta['camera_intrinsics']
        cx = intrinsics[0]
        cy = intrinsics[1]
        fx = intrinsics[2]
        fy = intrinsics[3]
    else:  # fall back to camera_angle_x
        camera_angle_x = float(meta['camera_angle_x'])
        focal = float(.5 * W / np.tan(.5 * camera_angle_x))
        cx = W / 2.
        cy = H / 2.
        fx = focal
        fy = focal

    if half_res:  # half intrinsics
        H = H // 2
        W = W // 2
        cx = cy / 2.
        cy = cy / 2.
        fx = fx / 2.
        fy = fy / 2.

    # load data into shared memory
    shm_data_writer = NRDataSHMArrayWriter(
        total_image_num=total_image_num,
        num_image_per_split=num_image_per_split,
        camera=CameraModel(H, W, cx, cy, fx, fy, zn, zf)
    )
    imgs, poses, pls = shm_data_writer.get_shm_arrays()

    global_index = 0
    for s in splits:
        meta = metas[s]
        for frame in tqdm(meta['frames'], desc=f'Loading {s} data'):
            frame_image_ext = frame.get('file_ext', '.png')
            filename = basedir / (frame['file_path'] + frame_image_ext)

            pl_pos = frame.get('pl_pos', [0, 0, 0])
            if frame_image_ext == '.npy':
                img = np.load(filename)
            elif frame_image_ext == '.exr':
                img = imageio.v3.imread(filename)
                # TODO: add log encoding
            else:
                img = imageio.v3.imread(filename) / 255.
            if half_res:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            if white_background:
                img = img[..., :3] * img[..., 3:] + (1. - img[..., 3:])
            else:
                img = img[..., :3]

            pls[global_index] = np.array(pl_pos, dtype=np.float32)
            imgs[global_index] = img.astype(np.float32)
            poses[global_index] = np.array(frame['transform_matrix']).astype(np.float32)

            global_index += 1

    return shm_data_writer
