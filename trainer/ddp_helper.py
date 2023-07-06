import os
import socket
import sys
from typing import List


# Constant strings
NCCL_LOCAL_RANK_ENV = "RANK"
NCCL_WORLD_SIZE_ENV = "WORLD_SIZE"
NCCL_MASTER_ADDR_ENV = "MASTER_ADDR"
NCCL_MASTER_PORT_ENV = "MASTER_PORT"


def find_free_network_port() -> int:
    """
    Finds a free port on localhost.
    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    ref: https://github.com/Lightning-AI/lightning/blob/acb7ee223c9877631b1f2704b4f21a07798f8fe2/src/lightning/fabric/plugins/environments/lightning.py#L91
    :return: port number
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def basic_subprocess_cmd() -> List[str]:
    """
    Returns the command to run the current script in a subprocess.
    ref: https://github.com/Lightning-AI/lightning/blob/acb7ee223c9877631b1f2704b4f21a07798f8fe2/src/lightning/fabric/strategies/launchers/subprocess_script.py#L136
    :return: list of args
    """
    import __main__  # local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    if __main__.__spec__ is None:  # pragma: no-cover
        return [sys.executable, os.path.abspath(sys.argv[0])] + sys.argv[1:]
    else:
        return [sys.executable, "-m", __main__.__spec__.name] + sys.argv[1:]


def local_rank() -> int:
    return int(os.environ.get(NCCL_LOCAL_RANK_ENV, 0))


def main_address() -> str:
    return os.environ.get(NCCL_MASTER_ADDR_ENV, "127.0.0.1")
