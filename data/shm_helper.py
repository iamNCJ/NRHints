from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List

import numpy as np
import tyro

from camera.camera_model import CameraModel


@dataclass
class NRDataSHMInfo:
    """
    Shared memory dataset meta info for NR data.
    """

    total_image_num: int
    """number of images in the dataset"""
    num_image_per_split: List[int]
    """number of images in each split, train / val / test"""
    camera: CameraModel
    """camera model, H, W, cx, cy, fx, fy, zn, zf"""
    imgs_shm_name: str
    """shared memory name for images"""
    poses_shm_name: str
    """shared memory name for poses"""
    pls_shm_name: str
    """shared memory name for pls"""


class SHMArray(np.ndarray):
    """
    A numpy ndarray backed by shared memory buffer.
    We need to store the shared memory object in the ndarray object to prevent it from being garbage collected.
    Ref: https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    """

    def __new__(cls, shape, dtype, shm_name=None, create=False):
        shm = shared_memory.SharedMemory(create=create, name=shm_name, size=np.prod(shape) * np.dtype(dtype).itemsize)
        obj = super().__new__(cls, shape=shape, dtype=dtype, buffer=shm.buf)
        obj.shm = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.shm = getattr(obj, 'shm', None)


class NRDataSHMArrayWriter(object):
    """
    Create shared memory arrays for NR data to store, generate meta info of the dataset for sharing between processes.
    """
    def __init__(self, total_image_num, num_image_per_split, camera: CameraModel):
        self.total_image_num = total_image_num
        self.num_image_per_split = num_image_per_split
        self.camera = camera
        self.imgs = SHMArray((total_image_num, camera.H, camera.W, 3), dtype=np.float32, create=True)
        self.poses = SHMArray((total_image_num, 4, 4), dtype=np.float32, create=True)
        self.pls = SHMArray((total_image_num, 3), dtype=np.float32, create=True)

    def get_shm_arrays(self) -> (SHMArray, SHMArray, SHMArray):
        return self.imgs, self.poses, self.pls

    def get_shm_info(self) -> NRDataSHMInfo:
        return NRDataSHMInfo(
            total_image_num=self.total_image_num,
            num_image_per_split=self.num_image_per_split,
            camera=self.camera,
            imgs_shm_name=self.imgs.shm.name,
            poses_shm_name=self.poses.shm.name,
            pls_shm_name=self.pls.shm.name
        )

    def release_shm(self) -> None:
        """
        Release shared memory in case of leakage.
        """

        self.imgs.shm.close()
        self.imgs.shm.unlink()
        del self.imgs

        self.poses.shm.close()
        self.poses.shm.unlink()
        del self.poses

        self.pls.shm.close()
        self.pls.shm.unlink()
        del self.pls


class NRDataSHMArrayReader(object):
    """
    Get shared memory arrays for NR data according to shared dataset meta info.
    """
    def __init__(self, shm_info: NRDataSHMInfo):
        self.total_image_num = shm_info.total_image_num
        self.num_image_per_split = shm_info.num_image_per_split
        self.camera = shm_info.camera
        H, W = self.camera.H, self.camera.W
        self.imgs = SHMArray((self.total_image_num, H, W, 3), dtype=np.float32, shm_name=shm_info.imgs_shm_name)
        self.poses = SHMArray((self.total_image_num, 4, 4), dtype=np.float32, shm_name=shm_info.poses_shm_name)
        self.pls = SHMArray((self.total_image_num, 3), dtype=np.float32, shm_name=shm_info.pls_shm_name)

    def get_shm_arrays(self) -> (SHMArray, SHMArray, SHMArray):
        return self.imgs, self.poses, self.pls

    def release_shm(self) -> None:
        """
        Release shared memory in case of leakage.
        """

        self.imgs.shm.close()
        del self.imgs

        self.poses.shm.close()
        del self.poses

        self.pls.shm.close()
        del self.pls


def serialize_shm_info(shm_info: NRDataSHMInfo) -> str:
    """Serialize NRDataSHMInfo to str, to pass to subprocesses."""
    return tyro.to_yaml(shm_info)


def deserialize_shm_info(serialized_shm_info_str: str) -> NRDataSHMInfo:
    """Deserialize NRDataSHMInfo from str, to get shm from process 0."""
    return tyro.from_yaml(NRDataSHMInfo, serialized_shm_info_str)


if __name__ == '__main__':
    from data.data_parser import parse_load_nr_data

    shm_data_writer = parse_load_nr_data('../../../datasets/Basket_Plane_PL_500')
    shm_info = shm_data_writer.get_shm_info()
    yaml_serialized = serialize_shm_info(shm_info)
    shm_info_deserialized = deserialize_shm_info(yaml_serialized)

    shm_data_writer.release_shm()
    print()
