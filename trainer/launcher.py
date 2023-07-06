import os
import subprocess

import torch
import torch.distributed

from configs.main_config import SystemConfig
from data.data_parser import parse_load_nr_data
from data.shm_helper import serialize_shm_info, deserialize_shm_info
from trainer.ddp_helper import (
    find_free_network_port,
    basic_subprocess_cmd,
    local_rank,
    main_address,
    NCCL_MASTER_ADDR_ENV,
    NCCL_MASTER_PORT_ENV,
    NCCL_LOCAL_RANK_ENV,
    NCCL_WORLD_SIZE_ENV
)
from trainer.trainer import Trainer


def launch(config: SystemConfig) -> None:
    # 1. parse & load data into shm
    # 2. init trainers and subprocess
    #    2.1 subprocesses will go to 1. again, thus need env to tell them they are not process 0!
    #    2.2 subprocesses need shm info from process 0, let's pass it through args
    #    2.3 not using mp since I hate to make everything pickleable
    # 3. begin train

    # 1. parse & load data into shm
    rank = local_rank()
    if rank == 0:
        shm_data_writer = parse_load_nr_data(config.data.path, white_background=config.data.white_background)
        shm_info = shm_data_writer.get_shm_info()
    else:
        assert config.serialized_shm_info is not None, "subprocesses need shm_info passed through args"
        shm_info = deserialize_shm_info(config.serialized_shm_info)

    # 2. init trainers and subprocess
    if rank == 0 and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            cmd = basic_subprocess_cmd()
            cmd += ["--config.serialized-shm-info", serialize_shm_info(shm_info)]

            # DDP environment variables
            os.environ[NCCL_MASTER_ADDR_ENV] = main_address()
            os.environ[NCCL_MASTER_PORT_ENV] = str(find_free_network_port())

            # allow the user to pass the node rank
            os.environ[NCCL_LOCAL_RANK_ENV] = str(0)
            os.environ[NCCL_WORLD_SIZE_ENV] = str(gpu_count)

            for i in range(1, gpu_count):
                env_copy = os.environ.copy()
                env_copy[NCCL_LOCAL_RANK_ENV] = str(i)
                subprocess.Popen(cmd, env=env_copy)
                print(f"Launched subprocess {i} of {gpu_count - 1}.")

    # 3. begin train
    trainer = Trainer(config, shm_info)
    trainer.run()

    # 4. exit gracefully
    trainer.release_shm()
    if rank == 0:
        shm_data_writer.release_shm()
