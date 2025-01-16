import torch
import numpy as np
import random
import wandb
import os
from models import encoders, decoders
from models.autoencoder import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB__SERVICE_WAIT"] = "300"


def setup_wandb(args, seed):
    """
    Setups wandb interface

    Args:
        args: Command line arguments from train_model.py
        seed: random seed used in experiments
    """
    # replace with your wandb key
    key = "YOUR_KEY"
    if key == "YOUR_KEY":
        raise ValueError("Please replace this value with your wandb key.")
    wandb.login(key=key)
    wandb_init_dict = {
            "data": args.data,
            "encoder": args.encoder,
            "decoder": args.decoder,
            "num_slots": args.num_slots,
            "slot_dim": args.slot_dim,
            "num_dec_layers_trans": args.num_dec_layers,
            "sigma": args.sigma,
            "alpha": args.alpha,
            "beta": args.beta,
            "warmup_lr_steps": args.warmup_lr_steps,
            "warmup_alpha_steps": args.warmup_alpha_steps,
            "lr_decay_rate": args.decay_rate,
            "lr_decay_steps": args.decay_steps,
            "eval_iter": args.eval_iter,
            "load_iter": args.load_iter,
            "load_seed": args.load_seed,
            "learning_rate": args.lr,
            "batch_size": args.batch_size}

    wandb.init(
        project="runs_" + args.data,
        name="seed_" + str(seed),
        config=wandb_init_dict)


def set_seed(seed):
    """
    Fixes random seed

    Args:
        seed: random seed
    """

    if seed == 0:
        seed = random.randint(0, 10000)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def setup_direcs(args, seed, load=False):
    """
    Setups directory to save model logs

    Args:
        args: Command line arguments from train_model.py

    Returns:
        directory for model logs as a string
    """

    if load:
        load_iter = 0
    else:
        load_iter = args.load_iter

    model_dir = "model_checkpoints/" + args.data + "_seed_" + str(seed) + "_alpha_" + str(args.alpha) + "_reg_step_" + str(load_iter) + "/"

    if load:
        return model_dir

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    return model_dir


def get_model(args):
    """
    Creates autoencoder model based on command line arguments in train_model.py

    Args:
        args: Command line arguments from train_model.py

    Returns:
        PyTorch model
    """

    if args.beta > 0:
        vae = True
        slot_dim = args.slot_dim * 2
    else:
        vae = False
        slot_dim = args.slot_dim

    if args.data == "clevr":
        resolution = (128, 128)
        chan_dim = 64
    elif args.data == "clevrtex":
        resolution = (16, 16)
        chan_dim = 64
    else:
        resolution = (64, 64)
        chan_dim = 32

    # get encoder
    encoder = encoders.SlotEncoder(resolution=resolution,
                                   num_slots=args.num_slots,
                                   slot_dim=slot_dim,
                                   chan_dim=chan_dim,
                                   enc_type=args.encoder,
                                   data=args.data).to(device)

    # get decoder
    if args.decoder == "spatial-broadcast":
        decoder = decoders.SpatialBroadcastDecoder(slot_dim=args.slot_dim,
                                                   resolution=resolution,
                                                   chan_dim=32).to(device)
    elif args.decoder == "transformer":
        decoder = decoders.TransformerDecoder(num_slots=args.num_slots,
                                              slot_dim=args.slot_dim,
                                              im_shape=resolution,
                                              proj_dim=args.proj_dim,
                                              query_dim=args.query_dim,
                                              n_layers=args.num_dec_layers).to(device)
    else:
        raise ValueError("Please specify a valid decoder type.")

    # get autoencoder
    autoencoder = AutoEncoder(data=args.data,
                                         num_slots=args.num_slots,
                                         slot_dim=slot_dim,
                                         encoder=encoder,
                                         decoder=decoder,
                                         vae=vae).to(device)
    return autoencoder


class LrScheduler:
    """
    Implements a learning rate schedule with warmup up and decay.
    Code adapted from https://github.com/stelzner/srt/blob/main/train.py
    """

    def __init__(self, peak_lr, warmup_steps, decay_rate, decay_it):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_it = decay_it

    def get_current_lr(self, it):
        # lr warmup
        if it < self.warmup_steps:  # Warmup period
            return self.peak_lr * (it / self.warmup_steps)
        it_since_peak = it - self.warmup_steps

        # lr decay
        return self.peak_lr * (self.decay_rate ** (it_since_peak / self.decay_it))