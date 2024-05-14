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
from torchmetrics.metric import Metric
from score.inception import InceptionV3
from score.fid import calculate_frechet_distance, torch_cov

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class FidMetrics(Metric):
    def __init__(self, fid_cache, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.add_state("fid_acts", default=[], dist_reduce_fx="cat")

        f = np.load(fid_cache)
        m2, s2 = f['mu'][:], f['sigma'][:]
        f.close()

        self.m2 = m2.astype(np.float32)
        self.s2 = s2.astype(np.float32)
        block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
        self.model = [InceptionV3([block_idx1, block_idx2])]
        self.model[0].eval()

    def update(self, batch_images):
        batch_images = (batch_images + 1) / 2
        
        pred = self.model[0](batch_images)
        self.fid_acts.append(pred[0].view(-1, 2048))
        pass
    
    def compute(self):
        self.model[0].cpu()
        m1 = torch.mean(self.fid_acts, axis=0).cpu().numpy()
        s1 = torch_cov(self.fid_acts, rowvar=False).cpu().numpy()

        fid_score = calculate_frechet_distance(m1, s1, self.m2, self.s2, use_torch=False)
        super().reset() # 需要调用父类的reset才可以
        return fid_score    
    
    def reset_model(self):
        self.model[0].to(self.device)

class Diffusion(L.LightningModule):
    def __init__(self, args, config):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.args = args
        self.config = config
        self.fid_metric = FidMetrics(fid_cache=self.config.eval.fid_cache)

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
        print(f"train: {torch.cuda.memory_allocated()}")

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
            
            if self.local_rank == 0:
                self.log(f"layer{t}/loss", loss)
                # logging.info(
                #     f"epoch: {self.current_epoch} layer: {t} step: {self.global_step}, loss: {loss.item()}"
                # )
                
            losses.append(loss)
        
        if self.config.training.use_loss_weight:
            losses = [weight*loss for weight, loss in zip(self.loss_weight, losses)]
        loss_sum = torch.stack(losses)               
        loss_sum = loss_sum.sum()
        
        return loss_sum
    
    def on_validation_epoch_start(self):
        print(f'on_validation_epoch_start: {torch.cuda.memory_allocated()}')
        self.fid_metric.reset_model()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.ema.module.sample(x)
        self.fid_metric.update(out)
        print(f"validation_batch: {torch.cuda.memory_allocated()}")

    def on_validation_epoch_end(self):
        print(f"on_validation_epoch_end: {torch.cuda.memory_allocated()}")
        FID = self.fid_metric.compute()
        if self.local_rank == 0:    
            if self.args.train:
                self.log("fid", FID)
            else:
                self.val_fid = FID
        print(f'on_validation_epoch_end2: {torch.cuda.memory_allocated()}')
                
    def on_validation_end(self):
        if self.local_rank == 0:
            if not self.args.train:
                logging.info(f"Model(EMA): FID:{self.val_fid:7.3f}")

    def sample(self):
        model = self.model
        if not self.args.model:
            model = self.ema.module
        model.eval()
        self.sample_image(model, os.path.join(self.args.image_folder, f"sample.png"))

    def sample_image(self, model, path):
        
        config = self.config
        
        x = torch.randn(
            config.eval.batch_size,
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
