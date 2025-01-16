import torch
from sklearn.metrics import adjusted_rand_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_recon_loss(x, xh):
    if len(x.shape) != 4:
        xh = xh.flatten(2, 3)
        x = x.permute(0, 2, 1)
    return (x - xh).square().sum() / (x.shape[0])


def get_ari(true_mask, pred_mask, num_ignored_objects=1):
    """Computes the ARI score.

    Args:
        true_mask: tensor of shape [batch_size, n_objects, *] where values go from 0 to the number of objects.
        pred_mask:  tensor of shape [batch_size, n_objects, *] where values go from 0 to the number of objects.
        num_ignored_objects: number of objects (in ground-truth mask) to be ignored when computing ARI.

    Returns:
        a vector of ARI scores, of shape [batch_size, ].
    """
    true_mask = true_mask.cpu().argmax(dim=1, keepdim=True)
    pred_mask = pred_mask.cpu().argmax(dim=1, keepdim=True)

    true_mask = true_mask.flatten(1)
    pred_mask = pred_mask.flatten(1)

    not_bg = true_mask >= num_ignored_objects

    result = []
    batch_size = len(true_mask)
    for i in range(batch_size):
        ari_value = adjusted_rand_score(
            true_mask[i][not_bg[i]], pred_mask[i][not_bg[i]]
        )
        result.append(ari_value)
    return torch.tensor(result).mean()


def compute_jac_iter_full(zh, model, im_size):
    num_samp = zh.shape[0]

    # first conditional is for clevrtex
    if im_size == 16:
        chan_dim = 768
        model.decoder.pixel_increment = 2
    else:
        chan_dim = 3
        model.decoder.pixel_increment = 500
    with torch.no_grad():
        jac_full = None
        while model.decoder.pixel_loop < im_size * im_size:
            pixel_gradients = torch.vmap(torch.func.jacfwd(model.decoder.iter_jac_comp))(zh).squeeze(1).flatten(1, 2).flatten(2, 3)
            lat_dim = pixel_gradients.shape[-1]
            num_pixels = pixel_gradients.shape[1] // chan_dim
            pixel_gradients = pixel_gradients.view(num_samp, num_pixels, chan_dim, lat_dim)

            if jac_full is None:
                jac_full = pixel_gradients
            else:
                jac_full = torch.cat((jac_full, pixel_gradients), dim=1)

            model.decoder.pixel_loop += model.decoder.pixel_increment

        model.decoder.pixel_loop = 0
        model.decoder.pixel_increment = 0

    return jac_full.permute(0, 2, 1, 3)


def compute_decoder_jacobian(args, model, zh, norm=False):
    with torch.no_grad():
        bs = zh.shape[0]
        im_size = 64
        chan_dim = 3
        if args.data == "clevr":
            im_size = 128
        elif args.data == "clevrtex":
            im_size = 16
            chan_dim = 768

        for i in range(bs):
            if im_size == 64 or args.decoder != "transformer":
                # compute jacobian all at once
                jac = torch.vmap(torch.func.jacfwd(model.decoder))(zh[i].unsqueeze(0).flatten(1)).cpu()
            else:
                # compute jacobian iteratively
                jac = compute_jac_iter_full(zh[i].unsqueeze(0), model, im_size).cpu()

            if i == 0:
                jac_full = jac
            else:
                jac_full = torch.cat((jac_full, jac), 0)

        jac_full = jac_full.view(bs, chan_dim, im_size, im_size, args.num_slots, args.slot_dim).abs().sum(5)
        # shape: num_points x num_color_channels x im_size x im_size x num_slots

        # getting slot wise jacobians summed over all color channels for a pixel
        jac_full = jac_full.sum(1).permute(0, 3, 1, 2)
        # shape: num_points x num_slots x im_size x im_size

        if norm:
            slot_pixel_grads = jac_full.flatten(2)
            # normalize pixel gradients by taking sum over slots and dividing
            sum_pixel_grads = torch.sum(slot_pixel_grads, 1).unsqueeze(1).repeat(1, args.num_slots, 1)
            norm_pixel_grads = slot_pixel_grads / sum_pixel_grads
            jac_full = norm_pixel_grads.view(bs, args.num_slots, im_size, im_size)

    return jac_full


def get_jac_interaction(slot_norm_jacobian):
    num_slots = slot_norm_jacobian.shape[1]
    norm_pixel_grads = slot_norm_jacobian.flatten(2)
    # shape: num_points x num_slots x num_pixels

    # first take max over slots for each pixel, then mean across all pixels.
    max_norm_pixel_grads = torch.max(norm_pixel_grads, 1)[0].flatten()
    obj_ind = torch.nonzero(max_norm_pixel_grads).flatten()
    mean_score = max_norm_pixel_grads[obj_ind].mean()

    # normalize these scores between 0 and 1 to get final score. upper bound is 1. lower bound is 1/num_slots
    jac_interaction_score = (mean_score - (1 / num_slots)) / (1 - (1 / num_slots))

    return jac_interaction_score
