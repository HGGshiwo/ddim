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

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


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
    parser.add_argument("--loss", action="store_true", help="Whether to test the model")
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
    
    if args.doc is None:
        args.doc = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
 
    new_config = dict2namespace(config)

    args.train = not args.sample and not args.loss and not args.fid 
    if args.train:
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
        
        if args.fid and hasattr(new_config.sampling, "ckpt"):
            if not os.path.exists(args.log_path):
                os.makedirs(args.log_path)
                with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                    yaml.dump(config, f, default_flow_style=False)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    new_config.device = device

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


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    
    try:
        if args.train and not args.resume_training:
            runner = Diffusion(args, config)
        else:
            runner = Diffusion.load_from_checkpoint()
            
        if args.sample:
            runner.sample2()
        elif args.loss:
            runner.loss()
        elif args.fid:
            runner.fid()
        else:
            dataset, _ = get_dataset(args, config)
            train_loader = data.DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
            )
            
            tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.exp, name="tensorboard", version=args.doc)
            
            trainer = L.Trainer(
                accelerator="gpu", 
                devices="auto", 
                default_root_dir=args.log_path, 
                max_epochs=config.training.n_epochs,
                strategy='ddp_find_unused_parameters_true',
                enable_progress_bar=False,
                logger=tb_logger,
            )
            trainer.fit(model=runner, train_dataloaders=train_loader)
            runner.train()
            
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
