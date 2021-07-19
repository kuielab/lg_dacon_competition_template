import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from src.model.utils.psnr import PSNR


class SimpleUNet(pl.LightningModule):
    def __init__(self, lr, loss, start_channels, growth_rate, num_layers, kernel_size, num_groups, img_shape: list or tuple = (256, 256)):
        super(SimpleUNet, self).__init__()

        self.lr = lr
        self.loss = loss
        self.start_channels = start_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.num_groups = num_groups

        num_channels = [3] + [start_channels * (growth_rate ** i) for i in range(num_layers)]
        encoders_ds = [
            (nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride=(1, 1), padding=kernel_size // 2,
                          groups=num_groups if input_channel % num_groups + output_channel % num_groups == 0 else 1),
                nn.LeakyReLU()),
             nn.Sequential(
                 nn.Conv2d(output_channel, output_channel, kernel_size=(2, 2), stride=(2, 2)),
                 nn.LeakyReLU(),
                 nn.BatchNorm2d(output_channel)
             ))
            for input_channel, output_channel
            in zip(num_channels[:-1], num_channels[1:])
        ]

        self.encoders = nn.ModuleList([encoder for encoder, ds in encoders_ds])
        self.downsamplers = nn.ModuleList([ds for encoder, ds in encoders_ds])

        decoders_us = [
            (nn.Sequential(
                nn.Conv2d(2*input_channel, output_channel, kernel_size, stride=(1, 1), padding=kernel_size // 2,
                          groups=num_groups if 2*input_channel % num_groups + output_channel % num_groups == 0 else 1),
                nn.LeakyReLU()),
             nn.Sequential(
                 nn.ConvTranspose2d(input_channel, input_channel, kernel_size=(2, 2), stride=(2, 2)),
                 nn.LeakyReLU(),
                 nn.BatchNorm2d(input_channel)
             ))
            for output_channel, input_channel
            in zip(reversed(num_channels[:-1]), reversed(num_channels[1:]))
        ]

        self.decoders = nn.ModuleList([decoder for decoder, us in decoders_us])
        self.upsamplers = nn.ModuleList([us for decoder, us in decoders_us])
        self.img_shape = img_shape

        if loss == 'l1_loss':
            self.loss = F.l1_loss
        elif loss == 'mse_loss':
            self.loss = F.mse_loss
        else:
            raise NotImplementedError('check loss function name')
        self.save_hyperparameters()

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param)

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr = self.lr)
        return optimizer

    def forward(self, x):
        block_output = []
        for layer, ds in zip(self.encoders, self.downsamplers):
            x = layer(x)
            block_output.append(x)
            x = ds(x)

        for skip, layer, us in zip(reversed(block_output), self.decoders, self.upsamplers):
            x = us(x)
            x = layer(torch.cat([skip, x], dim=-3))

        return x.sigmoid()

    def training_step(self, batch, batch_idx):
        input, target = batch
        input, target = input.float(), target.float()
        target_hat = self.forward(input)
        loss = self.loss(target_hat, target)

        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        input, target = input.float(), target.float()
        _, c, w, h = input.size()
        w_stride, h_stride = self.img_shape
        w_pad = int(np.ceil(w/w_stride) * w_stride - w)//2
        h_pad = int(np.ceil(h/h_stride) * h_stride - h)//2

        # padding
        padded_input = F.pad(input, (h_pad, h_pad, w_pad, w_pad))
        _, _, padded_w, padded_h = padded_input.size()
        # target_hat
        target_hat = torch.zeros_like(padded_input)

        # forward
        for i in range(padded_w//w_stride):
            for j in range(padded_h//h_stride):
                step_hat = self.forward(padded_input[..., i*w_stride: (i+1)*w_stride, j*h_stride: (j+1)*h_stride])
                target_hat[..., i*w_stride: (i+1)*w_stride, j*h_stride: (j+1)*h_stride] = step_hat

        # crop
        target_hat = target_hat[:, :, w_pad:w+w_pad, h_pad:h+h_pad]

        loss = self.loss(target_hat, target)
        self.log('valid/loss', loss, on_epoch=True)
        self.log('valid/PSNR', PSNR(target_hat, target), on_epoch = True)

        if batch_idx < 4:
            r = (input[0].detach().cpu().numpy(),
                 target[0].detach().cpu().numpy(),
                 target_hat[0].detach().cpu().numpy())
            return r

    def validation_epoch_end(self, outputs):
        for i, batch in enumerate(outputs[:4]):
            x, y, y_hat = batch
            x = rearrange(x, 'c h w -> h w c')
            y = rearrange(y, 'c h w -> h w c')
            y_hat = rearrange(y_hat, 'c h w -> h w c')
            for logger in self.logger:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log(
                        {'x_{}'.format(i): wandb.Image(x),
                         'y_{}'.format(i): wandb.Image(y),
                         'y_hat_{}'.format(i): wandb.Image(y_hat)}
                    )

