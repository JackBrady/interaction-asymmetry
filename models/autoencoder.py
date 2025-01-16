import torch
import torch.distributions as dists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(torch.nn.Module):
    def __init__(self, data, num_slots, slot_dim, encoder, decoder, vae):
        super().__init__()
        self.data = data
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae
        self.dkl = torch.Tensor([0.]).to(device)

    def sample_zh(self, enc_out):
        """
        Code adapted from: https://github.com/addtt/object-centric-library/blob/main/models/baseline_vae/model.py
        """
        if len(enc_out.shape) == 2:
            enc_out = enc_out.view(enc_out.shape[0], self.num_slots, self.slot_dim)

        bs, num_slots, slot_dim = enc_out.shape

        mu = enc_out[:, :, 0:slot_dim // 2].flatten(1)
        sig_2 = .5 * enc_out[:, :, slot_dim // 2:slot_dim].exp().flatten(1)

        prior_dist = dists.Normal(0.0, 1.0)
        latent_normal = dists.Normal(mu, sig_2)
        dkl = dists.kl_divergence(latent_normal, prior_dist).sum(dim=1).mean()
        zh = latent_normal.rsample()
        zh = zh.reshape(bs, num_slots, slot_dim // 2)
        return zh, dkl

    def forward(self, x):
        # encode
        zh = self.encoder(x)

        if self.vae:
            zh, self.dkl = self.sample_zh(zh)

        # decode
        xh = self.decoder(zh)

        return zh, xh
