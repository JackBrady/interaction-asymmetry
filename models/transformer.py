import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from einops import rearrange
from utilities.vis_utils import vis_slot_pixel_attn_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, z):
        return self.w_2(F.gelu(self.w_1(z)))


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_q, selfatt):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_q = d_q
        self.h = n_heads
        self.scale = d_q ** -0.5
        self.selfatt = selfatt
        if selfatt:
            self.to_qkv = nn.Linear(d_model, d_model * 3)
        else:
            self.to_q = nn.Linear(d_q, d_model)
            self.to_kv = nn.Linear(d_model, d_model*2)

        self.fin_proj = nn.Linear(d_model, d_q)
        self.attn = None

    def forward(self, x, z=None):
        if self.selfatt:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            assert z is not None
            query = self.to_q(x)
            key, value = self.to_kv(z).chunk(2, dim=-1)
            qkv = (query, key, value)

        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.h), qkv)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        p_attn = F.softmax(scores, dim=-1)
        self.attn = p_attn

        out = torch.matmul(p_attn, value)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.fin_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_q, d_ff, selfatt, crossatt):
        super(TransformerLayer, self).__init__()

        self.selfatt = selfatt
        self.crossatt = crossatt

        if self.crossatt:
            self.cross_attn = MultiHeadAttention(n_heads, d_model, d_q, selfatt=False)
            self.norm_att_cross_q = torch.nn.LayerNorm(d_q)

        if self.selfatt:
            self.self_attn = MultiHeadAttention(n_heads, d_model, d_model, selfatt=True)
            self.norm_att_self_q = torch.nn.LayerNorm(d_q)

        self.feed_forward = FeedForward(d_q, d_ff)

        self.norm_att_kv = torch.nn.LayerNorm(d_model)
        self.norm_ff = torch.nn.LayerNorm(d_q)

    def forward(self, x, z=None):
        # if you give z this means that you have cross attention where query is x otherwise self-att
        # regardless you always add x
        if self.crossatt:
            assert z is not None
            x = self.cross_attn(self.norm_att_cross_q(x), self.norm_att_kv(z)) + x

        if self.selfatt:
            x = self.self_attn(self.norm_att_self_q(x)) + x

        return self.feed_forward(self.norm_ff(x)) + x


class Transformer(nn.Module):
    def __init__(self, d_model, d_q, d_ff, n_heads, n_layers, n_slots, selfatt, crossatt):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.layers = nn.ModuleList([])
        self.n_heads = n_heads
        self.n_slots = n_slots
        self.selfatt = selfatt
        self.crossatt = crossatt
        self.fin_norm = torch.nn.LayerNorm(d_q)
        for _ in range(n_layers):
            self.layers.append(TransformerLayer(d_model, n_heads, d_q, d_ff, self.selfatt, self.crossatt))
        self.reset_parameters()

    def forward(self, x, z=None):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0]//self.d_model, self.d_model)

        if z is not None:
            if len(z.shape) == 1:
                z = z.view(1, z.shape[0]//self.d_model, self.d_model)

        for layer in self.layers:
            x = layer(x=x, z=z)

        return self.fin_norm(x)

    def cross_att_interaction(self, weights):
        if weights.shape[3] != self.n_slots:
            weights = weights.permute(0, 1, 3, 2)

        bs, n_head, n_queries, n_slots = weights.shape
        # sum over attention heads such that we ensure consistent pixel slot assignments across heads
        weights = weights.sum(1).permute(0, 2, 1)  # batch_size x num_slots x num_pixels

        # Compute sum products of each pixels attention weight for each slot with all other slots.
        interaction = 0
        for i in range(n_slots):
            for j in range(i, n_slots - 1):
                interaction += (weights[:, i] * weights[:, j + 1])

        # take mean over pixels and mean over pixels. Then take the mean of this quantity for the batch
        interaction = interaction.mean(1).mean()

        return interaction

    def compute_interaction(self):
        weights_cross = 0

        for i in range(self.n_layers):
            if self.layers[i].crossatt:
                weights_cross += (self.layers[i].cross_attn.attn / self.n_layers)

        interac_cross_att = self.cross_att_interaction(weights_cross)

        return interac_cross_att

    def vis_interaction(self, num_points, layer=None):
        weights = self.layers[layer-1].cross_attn.attn.sum(1) / (self.n_heads)
        num_slots = weights.shape[2]
        num_pixels = int(math.sqrt(weights.shape[1]))
        weights = weights.view(weights.shape[0], num_pixels, num_pixels, num_slots)
        fig = vis_slot_pixel_attn_mask(weights, num_slots, num_points)

        return fig

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)