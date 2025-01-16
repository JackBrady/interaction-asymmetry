from torch import nn
import torch
import torch.nn.functional as F
from models.slot_attention import SlotAttention
from models.positional_encodings import SoftPositionEmbed
from models.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNBackbone(nn.Module):
    def __init__(self, chan_dim, slot_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, chan_dim, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(chan_dim, chan_dim, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(chan_dim, chan_dim, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(chan_dim, slot_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = F.gelu(x)
        return x


class ViTEncoder(nn.Module):
    """
    Code adapted from: https://github.com/gorkaydemir/DINOSAUR/blob/main/models/model.py
    """
    def __init__(self):
        super().__init__()

        self.token_num = (128 * 128) // (8**2)
        self.model = self.load_model()

    def load_model(self):
        model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        for p in model.parameters():
            p.requires_grad = False

        return model

    @torch.no_grad()
    def forward(self, frames):
        # :arg frames:  (B, 3, H, W)
        #
        # :return x:  (B, token, 768)

        B = frames.shape[0]

        self.model.eval()
        x = self.model.prepare_tokens(frames)

        for blk in self.model.blocks:
            x = blk(x)
        x = x[:, 1:]

        assert x.shape[0] == B
        assert x.shape[1] == self.token_num
        assert x.shape[2] == 768

        return x


class SlotEncoder(nn.Module):
    def __init__(self, resolution, num_slots, slot_dim, chan_dim, enc_type, data):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.chan_dim = chan_dim
        self.enc_type = enc_type
        self.data = data

        self.cnn_backbone = CNNBackbone(self.chan_dim, self.slot_dim)

        if data == "clevrtex":
            self.mlp_inp_dim = 768
        else:
            self.mlp_inp_dim = self.slot_dim

        self.slot_mlp = nn.Sequential(nn.Linear(self.mlp_inp_dim, self.mlp_inp_dim),
                              nn.GELU(),
                              nn.Linear(self.mlp_inp_dim, self.slot_dim))

        self.encoder_pos = SoftPositionEmbed(self.slot_dim, resolution)

        if self.enc_type == "transformer":
            self.queries = nn.Parameter(torch.rand(1, self.num_slots, self.slot_dim))
            self.transformer = Transformer(d_model=self.slot_dim,
                                                d_q=self.slot_dim,
                                                d_ff=self.slot_dim * 2,
                                                n_heads=4,
                                                n_layers=5,
                                                n_slots=self.num_slots,
                                                selfatt=True,
                                                crossatt=True)

        elif self.enc_type == "slot-attention":
            self.slot_attention = SlotAttention(
                num_slots=self.num_slots,
                dim=self.slot_dim,
                iters=3,
                eps=1e-8,
                hidden_dim=64)

        else:
            raise ValueError("Please specify a valid encoder type.")

    def forward(self, x):
        bs = x.shape[0]
        if self.data != "clevrtex":
            x = self.cnn_backbone(x)
            x = x.permute(0, 2, 3, 1)
            x = self.encoder_pos(x)
            x = torch.flatten(x, 1, 2)

        x = nn.LayerNorm(x.shape[1:], device=device)(x)
        x = self.slot_mlp(x)

        if self.enc_type == "slot-attention":
            return self.slot_attention(x)

        elif self.enc_type == "transformer":
            queries = self.queries.repeat(bs, 1, 1)
            return self.transformer(queries, x)
