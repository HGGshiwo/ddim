import os
import logging

import numpy as np
import torch

from models.model import Model
from models.model_ema import ModelEma
from functions import get_optimizer
from datasets import data_transform, inverse_data_transform
from score.both import get_inception_and_fid_score
from torchvision.utils import save_image
from tqdm import trange
import lightning as L

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
        self.save_hyperparameters()
        self.args = args
        self.config = config
        
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.register_buffer("betas", torch.from_numpy(betas).float())
        
        self.num_timesteps = self.betas.shape[0]

        skip = self.num_timesteps // self.config.diffusion.num_block
        self.seq = range(0, self.num_timesteps, skip)
        at = (1-self.betas).cumprod(dim=0)
        
        t = np.array(list(reversed(self.seq[1:])))
        self.register_buffer('loss_weight', 1/at[t].view(-1, 1, 1, 1))
        
        self.model = Model(self.config, self.betas, self.seq)
        self.ema = ModelEma(self.model, decay=self.config.model.ema_rate)
    
        if self.local_rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) 
            logging.info(f"param: {total_params}")

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config, self.model.parameters())
        return optimizer
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.config.model.ema:
            self.ema.update(self.model)
        
        if self.local_rank == 0:    
            if self.global_step % self.config.training.sample_freq == 0:
                path2 = os.path.join(self.args.log_path, '%d_model.png' % self.global_step)
                self.sample_image(self.model, path2)
            

    def training_step(self, batch, batch_idx):
        seq = self.seq[1:]
        seq_next = self.seq[:-1]
        
        if self.args.train:
        
            if hasattr(self.config.training, "layer"):
                train_layer = self.config.training.layer
                seq = [self.seq[i] for i in train_layer]
                seq_next = [self.seq[i-1] for i in train_layer]
        x, y = batch 
        x = data_transform(self.config, x)
        x_T = torch.randn_like(x)

        at = (1-self.betas).cumprod(dim=0)[self.seq[-1]].view(-1, 1, 1, 1)
        true_x = at.sqrt() * x + (1-at).sqrt() * x_T
        true_xs = [true_x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
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
            
            # 只记录0号的损失
            if self.local_rank == 0:
                self.log(f"layer{t}/loss", loss)
                logging.info(
                    f"epoch: {self.current_epoch} layer: {t} step: {self.global_step}, loss: {loss.item()}"
                )
                
            losses.append(loss)
        
        if self.config.training.use_loss_weight:
            losses = [weight*loss for weight, loss in zip(self.loss_weight, losses)]
        loss_sum = torch.stack(losses)               
        loss_sum = loss_sum.sum()
        
        return loss_sum

    
    def sample(self):
        model = self.model
        if not self.args.model:
            model = self.ema.module
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

    def fid(self):
        model = self.model
        if not self.args.model:
            model = self.ema.module
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
        