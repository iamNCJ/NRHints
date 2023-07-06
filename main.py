from configs import get_config
from trainer import launch


if __name__ == '__main__':
    config = get_config().config
    launch(config)
