import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.model import Model
from models.diffusion import Model as UNet
from models.model_ema import ModelEma
from functions import get_optimizer
from functions.losses import end2end_loss, layer_loss
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from score.both import get_inception_and_fid_score
from torchvision.utils import make_grid, save_image
import random
from tqdm import trange
import torch.utils.tensorboard as tb


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


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        skip = self.num_timesteps // self.config.diffusion.num_block
        self.seq = range(0, self.num_timesteps, skip)

    def create_model(self):
        if self.config.use_pretrained:
            model = UNet(self.config, self.betas, self.seq) 
            states = torch.load(
                os.path.join("exp", f"model-790000.ckpt"),
                map_location=self.config.device,
            )
            model.load_state_dict(states, strict=True)
            model = model.to(self.device)
            ema = ModelEma(model, decay=self.config.model.ema_rate)
        else:
            model = Model(self.config, self.betas, self.seq)
            if not self.args.train:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
                model.load_state_dict(states[0], strict=True)
            
            model = model.to(self.device)
            if self.config.model.ema:
                ema = ModelEma(model, decay=self.config.model.ema_rate)
                if not self.args.train:
                    ema.load_state_dict(states[-1])
            else:
                ema = None
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
        logging.info(f"param: {total_params}")

        return model, ema

    def train(self):
        args, config = self.args, self.config
        tb_path = os.path.join(args.exp, "tensorboard", args.doc)
        tb_logger = tb.SummaryWriter(log_dir=tb_path)
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model, ema = self.create_model()
        optimizer = get_optimizer(self.config, model.parameters())

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema.load_state_dict(states[4])

        t_index = 0

        for epoch in range(start_epoch, self.config.training.n_epochs):
            
            model.train()  
            
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                x_T = torch.randn_like(x)
                
                if self.config.training.train_type == "end2end":
                    loss = end2end_loss(model, x, x_T)                
                else:
                    t_index = t_index % len(self.seq)
                    t = self.seq[t_index]
                    loss = layer_loss(model, x, t, x_T, self.betas)
                
                if self.config.training.train_type == "layer":
                    tb_logger.add_scalar(f"layer{t}/loss", loss, global_step=step)
                    logging.info(
                        f"epoch: {epoch} layer: {t} step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                    )
                else:
                    tb_logger.add_scalar(f"loss", loss, global_step=step)
                    logging.info(
                        f"epoch: {epoch} step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                    )
                t_index += 1

                optimizer.zero_grad()
                loss.backward()

                if self.config.training.train_type == "end2end":
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model[t].parameters(), config.optim.grad_clip
                    )
     
                optimizer.step()

                if self.config.model.ema:
                    if self.config.training.train_type == "layer":
                        ema.update(model, t)
                    else:
                        ema.update(model)
                
                if step % self.config.training.sample_freq == 0:
                    model.eval()
                    path = os.path.join(self.args.log_path, '%d.png' % step)
                    self.sample_image(ema.module, path)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema.state_dict())

                    # torch.save(
                    #     states,
                    #     os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    # )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
    
    def sample(self):
        model, _ = self.create_model()
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
            x = model.sample(x)

        x = inverse_data_transform(config, x)
        save_image(x, path, nrow=16)
            
    def fid(self):
        _, ema = self.create_model()
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
        
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))

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
       
        t = 0
        skip = self.num_timesteps // config.diffusion.num_block
        seq = range(0, self.num_timesteps, skip)
        t_index = 0
        for i, (x, y) in enumerate(train_loader):
            if t_index >= len(seq):
                break
            t = seq[t_index] * torch.ones(x.size(0), device=self.device, dtype=torch.long)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            x_T = torch.randn_like(x)
            loss = layer_loss(model, x, t, x_T, self.betas)
            print(f"layer {t_index} loss: {loss.item()}")
            t_index += 1
            
