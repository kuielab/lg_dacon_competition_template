import hydra
import dotenv
from omegaconf import OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info
from src.train import train

dotenv.load_dotenv(override=True)


@hydra.main(config_path='configs/', config_name='train.yaml')
def main(config):
    rank_zero_info(OmegaConf.to_yaml(cfg=config))

    return train(config)


if __name__ == '__main__':
    main()
