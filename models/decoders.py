from torch import nn
import torch
import torch.nn.functional as F
from models.positional_encodings import SoftPositionEmbed, PositionalEncoding, compute_grid_indices
from models.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpatialBroadcastDecoder(nn.Module):
    """
    Spatial broadcast decoder from Slot Attention: https://arxiv.org/abs/2006.15055
    Code adapted from: https://github.com/evelinehong/slot-attention-pytorch/blob/master/model.py
    """

    def __init__(self, slot_dim, resolution, chan_dim):
        super().__init__()
        self.slot_dim = slot_dim
        self.chan_dim = chan_dim
        self.conv1 = nn.ConvTranspose2d(slot_dim, self.chan_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(self.chan_dim, self.chan_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(self.chan_dim, self.chan_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(self.chan_dim, self.chan_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(self.chan_dim, self.chan_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(self.chan_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(slot_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        if len(x.shape) != 3:
            x = x.reshape(1, int(x.shape[0] / self.slot_dim), self.slot_dim)
        bs = x.shape[0]
        x = x.reshape((-1, x.shape[-1])).unsqueeze(1).unsqueeze(2)
        x = x.repeat((1, 8, 8, 1))
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = F.gelu(x)
        x = self.conv5(x)
        x = F.gelu(x)
        x = self.conv6(x)
        x = x[:, :, :self.resolution[0], :self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        xs, masks = x.reshape(bs, -1, x.shape[1], x.shape[2], x.shape[3]).split([3, 1], dim=-1)
        masks = nn.Softmax(dim=1)(masks)
        xs = xs * masks
        x = torch.sum(xs, dim=1).permute(0, 3, 1, 2)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_slots, slot_dim, im_shape, proj_dim, query_dim, n_layers):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.proj_dim = proj_dim
        self.im_shape = im_shape
        self.n_layers = n_layers

        if im_shape == (16, 16):
            self.fin_mlp_hid_dim = 768
            self.out_channels = 768
            self.query_mult = 4
        else:
            self.fin_mlp_hid_dim = query_dim
            self.out_channels = 3
            self.query_mult = 2

        pos_enc = PositionalEncoding()

        self.enc_coords = pos_enc(compute_grid_indices(im_shape)).to(device)

        self.init_mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.proj_dim*4),
            nn.GELU(),
            nn.Linear(self.proj_dim*4, self.proj_dim))

        self.query_mlp = nn.Sequential(
            nn.Linear(self.enc_coords.shape[2], query_dim*self.query_mult),
            nn.GELU(),
            nn.Linear(query_dim*self.query_mult, query_dim))

        self.transformer = Transformer(d_model=self.proj_dim,
                                            d_q=query_dim,
                                            d_ff=query_dim * 2,
                                            n_heads=12,
                                            n_layers=n_layers,
                                            n_slots=self.num_slots,
                                            selfatt=False,
                                            crossatt=True)

        # first conditional is the output mlp used for clevrtex
        if im_shape == (16, 16):
            self.output_mlp = nn.Sequential(
                nn.Linear(query_dim, self.fin_mlp_hid_dim * 4),
                nn.GELU(),
                nn.Linear(self.fin_mlp_hid_dim * 4, self.out_channels))
        else:
            self.output_mlp = nn.Sequential(
                    nn.Linear(query_dim, self.fin_mlp_hid_dim),
                    nn.GELU(),
                    nn.Linear(self.fin_mlp_hid_dim, self.fin_mlp_hid_dim),
                    nn.GELU(),
                    nn.Linear(self.fin_mlp_hid_dim, self.out_channels))

        self.pixel_loop = 0
        self.pixel_increment = 0

    def forward(self, zh):
        if len(zh.shape) == 1:
            zh = zh.view(1, self.num_slots, self.slot_dim)

        bs = zh.shape[0]

        # map slots to higher dim
        zh_proj = self.init_mlp(zh)

        # map to output space
        queries = self.query_mlp(self.enc_coords).repeat(bs, 1, 1)
        xh = self.transformer(queries, zh_proj)
        xh = self.output_mlp(xh)

        # reshape to original image shape: batch_size x output_channels x im_height x im_width
        xh = xh.reshape(bs, self.im_shape[0], self.im_shape[1], self.out_channels).permute(0, 3, 1, 2)
        return xh

    # used to compute jacobian of decoder iteratively in blocks of pixels
    def iter_jac_comp(self, zh):
        zh = self.init_mlp(zh).unsqueeze(0)
        queries = self.query_mlp(self.enc_coords).repeat(1, 1, 1)

        start = self.pixel_loop
        if (queries.shape[1] - start) < self.pixel_increment:
            end = (queries.shape[1] - start) + start
        else:
            end = start + self.pixel_increment

        xh = self.transformer(queries[:, start:end, :], zh)
        xh = self.output_mlp(xh)
        return xh
