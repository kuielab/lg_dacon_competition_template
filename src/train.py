from typing import List
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src import utils
from src.dataloader.lg_dataloader import Datasets


def train(config: DictConfig):
    ## data
    datasets: Datasets = hydra.utils.instantiate(config.dataloader.datasets)
    train_loader, valid_loader = datasets.get_dataloaders()

    if "seed" in config:
        pl.seed_everything(config.seed)

    ## model
    model = hydra.utils.instantiate(config.model)

    ## callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    ## logger
    logger: List[pl.LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))
        if any([isinstance(l, WandbLogger) for l in logger]):
            utils.wandb_login(key=config.wandb_api_key)

    ## trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    utils.log_hyperparameters(
        config=config,
        model=model,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    for l in [l for l in logger if isinstance(l, WandbLogger)]:
        l.watch(model=model, log='all', log_freq=25)

    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=valid_loader)

    utils.finish(
        logger=logger,
    )
    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
