import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
from models.model import Model
from models.diffusion import Model as UNet
from models.model_ema import ModelEma
from functions import get_optimizer
from functions.losses import end2end_loss, layer_loss, layer_loss_v2
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from score.both import get_inception_and_fid_score
from torchvision.utils import make_grid, save_image
import random
from tqdm import trange
import torch.utils.tensorboard as tb
import copy
import lightning as L

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(L.LightningModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
    
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.register_buffer('betas', betas)
        self.num_timesteps = betas.shape[0]

        skip = self.num_timesteps // self.config.diffusion.num_block
        self.seq = range(0, self.num_timesteps, skip)
        at = (1-self.betas).cumprod(dim=0)
        if self.config.training.train_type == "end2end":
            t = np.array(list(reversed(self.seq[1:])))
            self.register_buffer('loss_weight', 1/at[t].view(-1, 1, 1, 1))
        
        self.model = Model(self.config, self.betas, self.seq)
        self.ema = ModelEma(self.model, decay=self.config.model.ema_rate)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config, self.model.parameters())
        return optimizer
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.config.model.ema:
            self.ema.update(self.model)
                    
        if self.global_step % self.config.training.sample_freq == 0:
            # path = os.path.join(self.args.log_path, '%d.png' % step)
            # self.sample_image(ema.module, path)
            path2 = os.path.join(self.args.log_path, '%d_model.png' % self.global_step)
            self.sample_image(self.model, path2)
    
    def training_step(self, batch, batch_idx):
        
        seq = self.seq[1:]
        seq_next = self.seq[:-1]

        if hasattr(self.config.training, "layer"):
            train_layer = self.config.training.layer
            seq = [self.seq[i] for i in train_layer]
            seq_next = [self.seq[i-1] for i in train_layer]
            
        x, y = batch
        x = x.to(self.device)
        x = data_transform(self.config, x)
        x_T = torch.randn_like(x)

        at = (1-self.betas).cumprod(dim=0)[self.seq[-1]].view(-1, 1, 1, 1)
        true_x = at.sqrt() * x + (1-at).sqrt() * x_T
        true_xs = [true_x]
        for i, j in zip(reversed(self.seq[1:]), reversed(self.seq[:-1])):
            at = (1-self.betas).cumprod(dim=0)[i].view(-1, 1, 1, 1)
            at_1 = (1-self.betas).cumprod(dim=0)[j].view(-1, 1, 1, 1)
            true_x = at_1.sqrt() * x + (1 - at_1).sqrt() * (true_x - at.sqrt() * x) / (1-at).sqrt()
            true_xs.append(true_x)
            
        x = true_xs[0]
        true_x_seq = true_xs[:-1]
        true_x_seq_next = true_xs[1:]  
                                
        losses = []

        if self.config.training.use_true_x:
            outputs = self.model(x, true_x_seq)                       
        else:
            outputs = self.model(x)
        
        h = x
        for k, (t, t_next) in enumerate(zip(reversed(seq), reversed(seq_next))):
            loss = outputs[k] - true_x_seq_next[k]
        
            if self.config.training.use_adv_loss:
                at = (1-self.betas).cumprod(dim=0)[t].view(-1, 1, 1, 1)
                at_1 = (1-self.betas).cumprod(dim=0)[t_next].view(-1, 1, 1, 1)
            
                
                coeff = (1-at_1).sqrt() - (at_1/at).sqrt() * (1-at).sqrt()
                loss = (loss - (at_1/at).sqrt()*(h-true_x_seq[k]))
                h = outputs[k]

            loss = loss.square().sum((1,2,3)).mean(dim=0)
            
            self.log(f"layer{t}/loss", loss)      
            logging.info(
                f"epoch: {self.current_epoch} layer: {t} step: {batch_idx}, loss: {loss.item()}"
            )
                
            losses.append(loss)
        if self.config.training.use_loss_weight:
            losses = [weight*loss for weight, loss in zip(self.loss_weight, losses)]
        loss_sum = torch.stack(losses)               
        loss_sum = loss_sum.sum()
        
        return loss_sum    
    
    def sample(self):
        model, ema = self.create_model()
        if not self.args.model:
            model = ema.module
        model.eval()
        self.sample_image(model, os.path.join(self.args.image_folder, f"sample.png"))

    def sample_image(self, model, path):
        
        config = self.config
        
        x = torch.randn(
            config.sampling.batch_size,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            model.eval()
            x = model.sample(x)
            model.train()
            
        x = inverse_data_transform(config, x)
        save_image(x, path, nrow=16)

    def sample2(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        
        model, ema = self.create_model()
        t_idx = len(self.seq) // 2
        t, t_next = self.seq[t_idx], self.seq[t_idx - 1]
        with torch.no_grad():
            a = (1-self.betas).cumprod(dim=0)[t].view(-1, 1, 1, 1)
            x0, _ = next(iter(train_loader))
            x0 = x0.to(self.device)
            x0 = data_transform(self.config, x0)

            e = torch.randn_like(x0)
            x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
            # e_pred = model[t](x, t, t_next)
            # x0_pred = (x - e_pred * (1.0 - a).sqrt()) / a.sqrt()
            # print(((1.0 - a).sqrt()).item())
            # print((e_pred - e).square().sum(dim=(1,2,3)).mean(dim=0).item())
            
            for i, j in zip(reversed(self.seq[1:t_idx]), reversed(self.seq[:-1-t_idx])):        
                x = model.models[str(i)].sample(x, i, j)                
            x0_pred = x
            # x0_pred = model.sample(x) # 噪声的影响超过了图片的影响
            print((x0_pred - x0).square().sum(dim=(1,2,3)).mean(dim=0).item())
            x0_pred = inverse_data_transform(config, x0_pred)
            x0 = inverse_data_transform(config, x0)
            x = inverse_data_transform(config, x)
            save_image(x0_pred, os.path.join(self.args.image_folder, f"x0_pred.png"), nrow=16)
            save_image(x0, os.path.join(self.args.image_folder, f"x0.png"), nrow=16)
            # save_image(x, os.path.join(self.args.image_folder, f"x.png"), nrow=16)

    def fid(self):
        model, ema = self.create_model()
        if not self.args.model:
            model = ema.module
        model.eval()
        config = self.config.eval
        with torch.no_grad():
            images = []
            desc = "generating images"
            for i in trange(0, config.num_images, config.batch_size, desc=desc):
                batch_size = min(config.batch_size, config.num_images - i)
                x_T = torch.randn((batch_size, 3, self.config.data.image_size, self.config.data.image_size), device=self.device)
                batch_images = model.sample(x_T).cpu()
                images.append((batch_images + 1) / 2)
            images = torch.cat(images, dim=0).numpy()
    
        (IS, IS_std), FID = get_inception_and_fid_score(
            images, config.fid_cache, num_images=config.num_images,
            use_torch=config.fid_use_torch, verbose=True)
        
        model_name = "Model" if self.args.model else "Model(EMA)"
        print(f"{model_name}: IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))

    def fid2(self):
        _, ema = self.create_model()
        model = ema.module
        model.eval()
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        last_n = (total_n_samples - img_id) % config.sampling.batch_size
        batchs = [config.sampling.batch_size] * n_rounds
        if last_n != 0:
            batchs += [last_n]
        with torch.no_grad():
            for n in tqdm.tqdm(
                batchs, desc="Generating image samples for FID evaluation."
            ): 
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = model.sample(x)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def loss2(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model, ema = self.create_model()
       
        t = 0
        skip = self.num_timesteps // config.diffusion.num_block
        seq = range(0, self.num_timesteps, skip)
        t_index = 0
        for i, (x, y) in enumerate(train_loader):
            if t_index >= len(seq):
                break
            t = seq[t_index]
            x = x.to(self.device)
            x = data_transform(self.config, x)
            x_T = torch.randn_like(x)
            loss = layer_loss(model, x, t, x_T, self.betas)
            print(f"layer {t_index} loss: {loss.item()}")
            t_index += 1

    def loss(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model, ema = self.create_model()

        skip = self.num_timesteps // config.diffusion.num_block
        seq = range(0, self.num_timesteps, skip)
        for i, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            x = data_transform(self.config, x)
            x_T = torch.randn_like(x)
            at = (1-self.betas).cumprod(dim=0)[self.seq[-1]].view(-1, 1, 1, 1)
            true_x = at.sqrt() * x + (1-at).sqrt() * x_T
            true_xs = [true_x]
            for i, j in zip(reversed(self.seq[1:]), reversed(self.seq[:-1])):
                at = (1-self.betas).cumprod(dim=0)[i].view(-1, 1, 1, 1)
                at_1 = (1-self.betas).cumprod(dim=0)[j].view(-1, 1, 1, 1)
                true_x = at_1.sqrt() * x + (1 - at_1).sqrt() * (true_x - at.sqrt() * x) / (1-at).sqrt()
                true_xs.append(true_x)
                
            x = true_xs[0]
            true_x_seq = true_xs[:-1]
            true_x_seq_next = true_xs[1:]  
            
            for k, (t, t_next) in enumerate(zip(reversed(seq[1:]), reversed(seq[:-1]))):    
                x = model[str(t)].sample(x, t, t_next)
                out = model[str(t)].sample(true_x_seq[k], t, t_next)
                loss1 = (out - true_x_seq_next[k]).square().sum((1,2,3)).mean(dim=0)
                loss2 = (x - true_x_seq_next[k]).square().sum((1,2,3)).mean(dim=0)
                print(f"layer {t} loss: {loss1.item()} loss2: {loss2.item()}")
                
            break