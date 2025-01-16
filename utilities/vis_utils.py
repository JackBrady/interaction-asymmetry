import matplotlib.pyplot as plt
import wandb


def vis_slot_pixel_attn_mask(attn_mask, num_slots, num_points_vis):
    # shape should be num_points, num_pixels, num_pixels, num_slots
    fig, axarr = plt.subplots(num_points_vis, num_slots, figsize=(50, 20), tight_layout=True)
    # for clevr6
    if num_slots == 7:
        fig.subplots_adjust(wspace=-0.905)
    # for clevrtex
    elif num_slots > 7:
        fig.subplots_adjust(wspace=-0.655)
    # for sprites
    else:
        fig.subplots_adjust(wspace=-0.953)

    for i in range(num_points_vis):
        for j in range(num_slots):
            at_m = attn_mask[i, :, :, j]
            cax = axarr[i, j].matshow(at_m.cpu())
            # cb = plt.colorbar(cax, ax=axarr[i, j], pad=0.008)
            cax.set_clim(0., 1.)
            axarr[i, j].axis("off")

    return fig


def vis_reconstructions(x, xh, data):
    # vis reconstructed images
    samp_ims = []
    for i in range(len(x)):
        samp_ims.append(wandb.Image(x[i].unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy(), caption="Original_" + str(i)))
        if data != "clevrtex":
            samp_ims.append(
                wandb.Image(xh[i].unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy(), caption="Reconstruction_" + str(i)))

    return samp_ims


def vis_interactions(args, model, num_points_vis):
    interac_proj_1, interac_proj_2 = 0, 0

    # vis attention matrices for decoder in transformer
    if args.num_dec_layers >= 1:
        interac_proj_1 = model.decoder.transformer.vis_interaction(num_points_vis, layer=1)
    if args.num_dec_layers >= 2:
        interac_proj_2 = model.decoder.transformer.vis_interaction(num_points_vis, layer=2)

    return interac_proj_1, interac_proj_2
