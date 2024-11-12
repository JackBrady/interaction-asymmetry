import torch
import wandb
import torch.nn.functional as F
from data.dataloader import get_dataloader
from eval.eval_metrics import get_ari
from utils.vis_utils import vis_interactions, vis_reconstructions, vis_slot_pixel_attn_mask
from eval.eval_metrics import get_recon_loss
from eval.eval_metrics import compute_decoder_jacobian, get_jac_interaction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(args, model, validation=True):
    # eval step
    model.eval()
    eval_dict = dict()

    with torch.no_grad():
        num_points_vis = 5
        b_it = 0

        if validation:
            # get validation dataloader
            _, data_loader, _ = get_dataloader(args)
        else:
            # get test dataloader
            _, _, data_loader = get_dataloader(args)

        num_batches = len(data_loader)

        for x, masks in data_loader:
            x = x.to(device)
            zh, xh = model(x)

            # get recon_loss and dkl
            val_recon = (get_recon_loss(x, xh) / num_batches)
            eval_dict["val_recon"] = val_recon + eval_dict.get("val_recon", 0)

            if args.beta > 0:
                dkl = (model.dkl / num_batches)
                eval_dict["dkl"] = dkl + eval_dict.get("dkl", 0)

            # get interaction for transformer
            if args.decoder == "transformer":
                _, interac_dec_cross_att = model.decoder.transformer.compute_interaction()
                interac_dec_cross_att = (interac_dec_cross_att / num_batches)
                eval_dict["interac_dec_cross_att"] = interac_dec_cross_att + eval_dict.get("interac_dec_cross_att", 0)

            if b_it == 0:
                recon_images = vis_reconstructions(x[0:num_points_vis], xh[0:num_points_vis])
                eval_dict["recon_images"] = recon_images

                if args.decoder == "transformer":
                    interac_proj_l1, interac_proj_l2 = vis_interactions(args, model, num_points_vis)
                    eval_dict["interac_proj_l1"], eval_dict["interac_proj_l2"] = interac_proj_l1, interac_proj_l2

            # compute slot-wise jacobian normalized across slots
            if validation:
                num_points_jac = int(args.batch_size * .25)
            else:
                num_points_jac = zh.shape[0]

            slot_norm_jacobian = compute_decoder_jacobian(args, model, zh[0:num_points_jac], norm=True)

            masks = F.one_hot(
                torch.as_tensor(masks, dtype=torch.int64),
            ).squeeze(3).permute(0, 3, 1, 2)

            background_mask = masks.squeeze(2)[:, 1:masks.shape[1], :, :].sum(1).unsqueeze(1).repeat(1, args.num_slots, 1, 1)

            # zero out all background pixels for slot-wise jacobians
            slot_norm_jacobian = (slot_norm_jacobian * background_mask[0:num_points_jac])

            # compute ari for foreground pixels using normalized jacobian as mask
            dec_jac_ari = (get_ari(masks[0:num_points_jac], slot_norm_jacobian) / num_batches)
            eval_dict["dec_jac_ari"] = dec_jac_ari + eval_dict.get("dec_jac_ari", 0)

            # compute interaction for foreground pixels
            dec_jac_interaction = (get_jac_interaction(slot_norm_jacobian) / num_batches)
            eval_dict["dec_jac_interaction"] = dec_jac_interaction + eval_dict.get("dec_jac_interaction", 0)

            if b_it == 0:
                # visualize normalized jacobian and ground-truth mask
                eval_dict["slot_norm_jacobian"] = vis_slot_pixel_attn_mask(slot_norm_jacobian.permute(0, 2, 3, 1),
                                                                  args.num_slots,
                                                                  num_points_vis)

                eval_dict["ground_truth_mask"] = vis_slot_pixel_attn_mask(masks.squeeze(2).permute(0, 2, 3, 1),
                                                                 args.num_slots,
                                                                 num_points_vis)

            b_it += 1

    for key in eval_dict.keys():
        if torch.is_tensor(eval_dict[key]):
            eval_dict[key] = eval_dict[key].item()

    if args.use_wandb == 1:
        wandb.log(eval_dict)
