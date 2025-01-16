import torch
import numpy as np
import argparse
import warnings
from models.encoders import ViTEncoder
from eval.eval_model import eval_model
from data.dataloader import get_dataloader
from utilities.setup_utils import set_seed, setup_wandb, get_model, LrScheduler, setup_direcs
import wandb
from eval.eval_metrics import get_recon_loss
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(args):
    # fix random seed
    seed = set_seed(args.seed)
    print("Seed: ", seed)

    # get directory for model checkpoints
    model_dir = setup_direcs(args, seed)

    # setup weights and biases
    if args.use_wandb == 1:
        setup_wandb(args, seed)

    # load data
    train_loader, _, _ = get_dataloader(args)

    # create model
    model = get_model(args)
    if args.data == "clevrtex":
        vis_encoder = ViTEncoder().to(device)
        vis_encoder.eval()
    else:
        vis_encoder = None

    print("Num Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # the first conditional is the lr scheduling used in the paper for sprites and clevr6, the second is for clevrtex
    alpha_scheduler = LrScheduler(args.alpha, args.warmup_alpha_steps, .2, 250000)
    if (args.alpha > 0 or args.beta > 0) and args.data != "clevrtex":
        reg = True
        lr_scheduler = LrScheduler(args.lr, args.warmup_lr_steps, args.decay_rate, args.decay_steps)
        lr_scheduler_2 = LrScheduler(1e-4, 30000, args.decay_rate, args.decay_steps)
    else:
        lr_scheduler = LrScheduler(args.lr, args.warmup_lr_steps, args.decay_rate, args.decay_steps)

    glob_it = 0
    # set global iteration to load iteration if we're loading a model
    if args.load_seed > 0:
        glob_it = args.load_iter

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_scheduler.get_current_lr(glob_it))

    # load model and optimizer if specified
    if args.load_seed > 0:
        load_dir = setup_direcs(args, args.load_seed, load=True)
        model.load_state_dict(torch.load(load_dir + "_iter_" + str(glob_it) + "_model_state_dict.pth"))
        optimizer.load_state_dict(torch.load(load_dir + "_iter_" + str(glob_it) + "_optimizer_state_dict.pth"))
    else:
        glob_it = 0

    # if evaluating a model on the test set, only run eval
    if args.test_model == 1:
        eval_model(args, model, validation=False, vis_encoder=vis_encoder)
        if args.use_wandb == 1:
            wandb.finish()
        return

    # train loop
    b_it = 0
    model.train()
    while glob_it < args.num_iters:
        b_it += 1
        glob_it += 1
        optimizer.zero_grad()

        # get data
        x, _ = next(iter(train_loader))
        x = x.to(device)

        # get reconstructed latents and observations
        if args.data == "clevrtex":
            x = vis_encoder(x).detach()

        zh, xh = model(x)

        # reconstruction loss
        loss = args.sigma * get_recon_loss(x, xh)

        # kl loss
        if args.beta > 0:
            beta = args.beta
            loss += beta * model.dkl

        # interaction loss
        if args.decoder == "transformer" and args.alpha > 0:
            interac_dec_cross_att = model.decoder.transformer.compute_interaction()
            if glob_it < args.warmup_alpha_steps:
                # get current value for alpha during warming phase
                alpha = alpha_scheduler.get_current_lr(glob_it)
            else:
                alpha = args.alpha
            loss += alpha * interac_dec_cross_att

        loss.backward()

        # for experiments on clevrtex, gradient clipping was used instead of lr decay scheme
        if args.data == "clevrtex":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # update learning rate
        new_lr = lr_scheduler.get_current_lr(glob_it)

        # lr scheduler for sprites and clevr6
        if args.data != "clevrtex":
            if reg and glob_it > args.warmup_lr_steps:
                if glob_it < 30000:
                    new_lr = args.lr
                if glob_it > 30000:
                    new_lr = lr_scheduler_2.get_current_lr(glob_it)

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        # eval and save model
        if glob_it % args.eval_iter == 0 or glob_it == args.num_iters:
            b_it = 0
            torch.save(model.state_dict(),
                       model_dir + "_iter_" + str(glob_it) + "_model_state_dict.pth")
            torch.save(optimizer.state_dict(),
                       model_dir + "_iter_" + str(glob_it) + "_optimizer_state_dict.pth")
            # run eval
            eval_model(args, model, vis_encoder=vis_encoder)
            model.train()

    if args.use_wandb == 1:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        type=str,
                        help="Specifies dataset to use"
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        help="Specifies the size of each training batch"
                        )
    parser.add_argument("--encoder",
                        type=str,
                        help="Specifies slot encoder architecture"
                        )
    parser.add_argument("--decoder",
                        type=str,
                        help="Specifies decoder architecture"
                        )
    parser.add_argument("--num_slots",
                        type=int,
                        help="Specifies number of slots in model"
                        )
    parser.add_argument("--slot_dim",
                        type=int,
                        help="Specifies dimension of each model slot"
                        )
    parser.add_argument("--beta",
                        type=float,
                        help="Specifies the weighting coefficient on the KL divergence loss"
                        )
    parser.add_argument("--alpha",
                        type=float,
                        help="Specifies the weighting coefficient on the interaction loss"
                        )
    parser.add_argument("--sigma",
                        type=float,
                        default=1.,
                        help="Specifies the weighting coefficient on the reconstruction loss"
                        )
    parser.add_argument("--query_dim",
                        type=int,
                        default="180",
                        help="Specifies the dimensionality of pixel query vectors, when using a Transformer decoder"
                        )
    parser.add_argument("--proj_dim",
                        type=int,
                        default="516",
                        help="Specifies the dimensionality slots are projected to before applying Transformer decoder"
                        )
    parser.add_argument("--num_dec_layers",
                        type=int,
                        default="2",
                        help="Specifies the number of layers to use in a Transformer decoder"
                        )
    parser.add_argument("--seed",
                        type=int,
                        default="0",
                        help="Fixes the random seed to given value if different than 0. Otherwise, a seed is sampled"
                        )
    parser.add_argument("--lr",
                        type=float,
                        default="5e-4",
                        help="Learning rate used during training"
                        )
    parser.add_argument("--warmup_lr_steps",
                        type=int,
                        default="10000",
                        help="Number of warmup steps for the learning rate"
                        )
    parser.add_argument("--warmup_alpha_steps",
                        type=int,
                        default="25000",
                        help="Number of warmup steps for the coefficient on the interaction loss"
                        )
    parser.add_argument("--decay_rate",
                        type=float,
                        default="0.1",
                        help="Decay rate for the learning rate"
                        )
    parser.add_argument("--decay_steps",
                        type=int,
                        default="500000",
                        help="Number of steps used to decay the learning rate"
                        )
    parser.add_argument("--num_iters",
                        type=int,
                        default="500000",
                        help="Number of training iterations"
                        )
    parser.add_argument("--eval_iter",
                        type=int,
                        default="10000",
                        help="Evaluation metrics computed and logged every number of iterations given by arg"
                        )
    parser.add_argument("--load_seed",
                        type=int,
                        default="-1",
                        help="If different than -1, then a model will be loaded and the seed given will be set"
                        )
    parser.add_argument("--load_iter",
                        type=int,
                        default="0",
                        help="Specifies the iteration that a model should be loaded from"
                        )
    parser.add_argument("--use_wandb",
                        type=int,
                        default="1",
                        help="If 1 is given then wandb will be used to log metrics; if 0 is given then it will not be"
                        )
    parser.add_argument("--test_model",
                        type=int,
                        default="0",
                        help="If 1 is given, then a loaded model is evaluated on the test set, opposed to being trained"
                        )
    args = parser.parse_args()

    if args.data == "sprites":
        args.num_slots = 5
        args.slot_dim = 32
        args.batch_size = 64
        args.encoder = "transformer"
        args.decoder = "transformer"
        args.alpha = .05
        args.beta = .05
        if args.alpha > 0 or args.beta > 0:
            args.sigma = 5.

    elif args.data == "clevr":
        args.num_slots = 7
        args.slot_dim = 64
        args.batch_size = 32
        args.encoder = "transformer"
        args.decoder = "transformer"
        args.alpha = .05
        args.beta = .05
        if args.alpha > 0 or args.beta > 0:
            args.sigma = 1.
        args.warmup_lr_steps = 2500

    elif args.data == "clevrtex":
        args.num_slots = 11
        args.slot_dim = 64
        args.batch_size = 32
        args.query_dim = 360
        args.encoder = "transformer"
        args.decoder = "transformer"
        args.alpha = .1
        args.beta = .1
        if args.alpha > 0 or args.beta > 0:
            args.sigma = .1

    train_model(args)
