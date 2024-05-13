import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import datetime
import lightning as L
from datasets import get_dataset
import torch.utils.data as data
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)
torch.set_float32_matmul_precision('high')

is_zero_rank = (os.getenv("LOCAL_RANK", '0') == '0')

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default=None,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--fid", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="use model to test fid",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        type=str,
        help="gpu id for testing",
    )
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  
    args.train = not args.sample and not args.fid 
    
    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    new_config = dict2namespace(config)
    
    
    if args.doc is None:
        if is_zero_rank:
            args.doc = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            args.log_path = os.path.join(args.exp, "logs", args.doc)
        else:
            args.log_path = None
    else:
        args.log_path = os.path.join(args.exp, "logs", args.doc)
    
    if args.train:
        if is_zero_rank: 
            if not args.resume_training:
                os.makedirs(args.log_path)

                with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                    yaml.dump(config, f, default_flow_style=False)

            # setup logger
            level = getattr(logging, args.verbose.upper(), None)
            if not isinstance(level, int):
                raise ValueError("level {} not supported".format(args.verbose))

            handler1 = logging.StreamHandler()
            handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
            formatter = logging.Formatter(
                "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
            )
            handler1.setFormatter(formatter)
            handler2.setFormatter(formatter)
            logger = logging.getLogger()
            logger.addHandler(handler1)
            logger.addHandler(handler2)
            logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    if args.sample:
        os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
        args.image_folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.image_folder = os.path.join(
            args.exp, "image_samples", args.image_folder
        )
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class RandnDataset(data.Dataset):
    def __init__(self, num_images):
        self.num_images = num_images

    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), 0

if __name__ == "__main__":
    args, config = parse_args_and_config()
    if is_zero_rank:
        logging.info("Writing log file to {}".format(args.log_path))
    
    try:
        if args.train:
            runner = Diffusion(args, config)
        else:
            runner = Diffusion.load_from_checkpoint(f"{args.log_path}/ckpt.pth", args=args, config=config)
            
        if args.sample:
            runner.sample() 
            exit()

        else: # fid or train
            test_loader = data.DataLoader(
                RandnDataset(config.eval.num_images), 
                batch_size=config.eval.batch_size, 
                num_workers=config.data.num_workers
            )
        callbacks = []
        if args.train:    
            dataset, _ = get_dataset(args, config)
            train_loader = data.DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
            )

            if is_zero_rank:
                tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.exp, name="tensorboard", version=args.doc)
            else:
                tb_logger = False 
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=args.log_path, 
                filename="ckpt", 
                every_n_train_steps=config.training.snapshot_freq
            )
            checkpoint_callback.FILE_EXTENSION = '.pth'
            callbacks.append(checkpoint_callback)
        else:
            tb_logger = False

        trainer = L.Trainer(
            accelerator="gpu", 
            devices="auto", 
            default_root_dir=args.log_path, 
            max_epochs=config.training.n_epochs,
            strategy='ddp_find_unused_parameters_true',
            # enable_progress_bar=not args.train,
            logger=tb_logger,
            gradient_clip_val=config.optim.grad_clip,
            callbacks=callbacks,
            log_every_n_steps=1,
            val_check_interval=config.training.fid_freq,
            check_val_every_n_epoch=None,
            num_sanity_val_steps=0,
        )
        
        if args.resume_training or args.fid:
            ckpt = f"{args.log_path}/ckpt.pth"
        else:
            ckpt = None

        if args.train:
            trainer.fit(model=runner, train_dataloaders=train_loader, ckpt_path=ckpt, 
                        val_dataloaders=test_loader)
        else:
            trainer.validate(model=runner, dataloaders=test_loader)

    except Exception:
        logging.error(traceback.format_exc())
