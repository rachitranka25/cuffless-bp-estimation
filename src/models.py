"""Deep baselines for BP regression from 10 s ECG+PPG segments.

Two architectures, matching what the spec asks for in month 4 and what the
PulseDB literature reports: a plain 1D CNN and a 1D ResNet.

Both predict SBP and DBP jointly from two output units rather than training a
separate network per target. The published work usually trains separately; joint
training halves the compute for the same data and lets the shared trunk see both
targets, and the two are strongly correlated anyway. Worth noting as a deviation
if the numbers ever need to line up exactly with a specific paper.

Input  (B, 2, 1250)  channel 0 = ECG, 1 = PPG, min-max scaled to [0, 1]
Output (B, 2)        standardised SBP, DBP — the trainer un-standardises
"""

import os

import torch
import torch.nn as nn

_SAFE_ATTENTION = os.environ.get("BP_SAFE_ATTENTION", "0") == "1"


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, stride=1, pool=2, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool) if pool > 1 else nn.Identity(),
            nn.Dropout(drop) if drop else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Straightforward convolutional stack — the standard first baseline."""

    def __init__(self, in_ch=2, width=32, drop=0.1, n_out=2):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            ConvBlock(in_ch, w, 7, pool=2),
            ConvBlock(w, w * 2, 5, pool=2, drop=drop),
            ConvBlock(w * 2, w * 4, 5, pool=2, drop=drop),
            ConvBlock(w * 4, w * 8, 3, pool=2, drop=drop),
            ConvBlock(w * 8, w * 8, 3, pool=2, drop=drop),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(w * 8, 128), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.head(self.features(x))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, k=7, stride=1, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(drop) if drop else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.short = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                    nn.BatchNorm1d(cout)))

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.short(x))


class ResNet1D(nn.Module):
    """1D ResNet — the strong deep baseline reported across PulseDB papers."""

    def __init__(self, in_ch=2, width=32, blocks=(2, 2, 2, 2), drop=0.1, n_out=2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, w, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(w), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        stages, cin = [], w
        for i, n in enumerate(blocks):
            cout = w * (2 ** i)
            for j in range(n):
                stages.append(ResBlock(cin, cout, stride=2 if (j == 0 and i > 0) else 1,
                                       drop=drop))
                cin = cout
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(drop), nn.Linear(cin, n_out),
        )

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


# --------------------------------------------------------------------------- #
# Transformer, following the published DMT design
# --------------------------------------------------------------------------- #

class FiLM(nn.Module):
    """Feature-wise linear modulation: scale and shift a sublayer's output.

    DMT's contribution is conditioning on the patient's age, sex and BMI, and
    doing it inside every block rather than concatenating the three numbers to
    the final feature vector. Each block gets its own gamma and beta produced
    from the demographic embedding, so the demographics can change how the
    signal is processed, not just shift the answer at the end.
    """

    def __init__(self, demo_dim, d_model):
        super().__init__()
        self.to_gamma_beta = nn.Linear(demo_dim, 2 * d_model)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)      # starts as identity

    def forward(self, x, demo_emb):
        gamma, beta = self.to_gamma_beta(demo_emb).chunk(2, dim=-1)
        return x * (1 + gamma).unsqueeze(1) + beta.unsqueeze(1)


class SelfAttention(nn.Module):
    """Multi-head self-attention on top of scaled_dot_product_attention.

    Written out rather than using nn.MultiheadAttention because on this machine
    SDPA is about three times faster on CPU (10.9 ms vs 30.5 ms per call at
    batch 64, 125 tokens, 128 dims). Note that both are *slower* on MPS than on
    CPU for this shape — Apple's backend handles convolutions well and attention
    badly — so the transformer trains on CPU while the convnets use the GPU.
    """

    def __init__(self, d_model, n_heads, drop=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = drop

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, T, self.h, self.dh)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        p = self.drop if self.training else 0.0
        if _SAFE_ATTENTION:
            # SDPA picks a fused backend on its own, and on some Windows +
            # recent-GPU combinations that kernel faults with an illegal memory
            # access. Set BP_SAFE_ATTENTION=1 to fall back to plain matmuls,
            # which are slower but always correct.
            att = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
            att = torch.softmax(att, dim=-1)
            if p:
                att = torch.nn.functional.dropout(att, p)
            o = att @ v
        else:
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=p)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class DMTBlock(nn.Module):
    def __init__(self, d_model, n_heads, demo_dim, mlp_ratio=4, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, drop=drop)
        self.film1 = FiLM(demo_dim, d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d_model * mlp_ratio, d_model), nn.Dropout(drop))
        self.film2 = FiLM(demo_dim, d_model)

    def forward(self, x, demo_emb):
        x = x + self.film1(self.attn(self.n1(x)), demo_emb)
        x = x + self.film2(self.mlp(self.n2(x)), demo_emb)
        return x


class DMT(nn.Module):
    """Transformer over PPG patches, conditioned on demographics.

    Follows arXiv:2606.11125: patch size 10 with stride 10 turns a 1250-sample
    clip into 125 tokens, then 6 blocks of 8-head self-attention with FiLM
    conditioning on age, sex and BMI.

    One deliberate deviation: one network with two outputs instead of separate
    SBP and DBP networks, which halves compute for the same data.

    `shape_head=True` adds DMT's other reported piece — arXiv:2606.11125 §III.D
    trains an auxiliary head that *classifies* each clip as normotensive- or
    hypertensive-like pulse morphology (not a regression head — corrected from
    an earlier version of this code that predicted raw morphology features
    instead), combined via Kendall-style learnable multi-task uncertainty
    weights (`log_sigma_bp`, `log_sigma_shape` — see Eq. 9 of the paper): two
    scalar parameters, not a per-sample heteroscedastic loss, which is both
    truer to the paper and far more stable to train (a per-sample version of
    this collapsed during an earlier trial run). The exact rule used to derive
    the classification *label* is still a best-effort stand-in — the paper
    cites external, unrestated work for its "morphology score" — see
    `train._shape_labels` for what this project uses instead.
    `forward` still returns exactly the (B, n_out) tensor everything
    downstream (eval, personalization, domain shift) already expects; the
    shape logits are stashed on `self.last_shape_logits` for the training loop.
    """

    def __init__(self, in_ch=1, d_model=128, depth=6, n_heads=8, patch=10,
                 demo_dim=3, demo_hidden=64, drop=0.1, n_out=2, seq_len=1250,
                 shape_head=False):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_ch, d_model, patch, stride=patch)
        n_tok = seq_len // patch
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.demo_enc = nn.Sequential(
            nn.Linear(demo_dim, demo_hidden), nn.GELU(),
            nn.Linear(demo_hidden, demo_hidden), nn.GELU())
        self.blocks = nn.ModuleList([
            DMTBlock(d_model, n_heads, demo_hidden, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d_model, n_out))
        self.has_shape_head = shape_head
        self.shape_head = nn.Linear(d_model, 2) if shape_head else None
        # Eq. 9's sigma_bp / sigma_shape, reparametrised as log(sigma) so
        # optimisation is unconstrained; loss uses exp(-log_sigma) + log_sigma,
        # algebraically identical to the paper's (1/sigma) + log(sigma).
        self.log_sigma_bp = nn.Parameter(torch.zeros(1)) if shape_head else None
        self.log_sigma_shape = nn.Parameter(torch.zeros(1)) if shape_head else None
        self.last_shape_logits = None

    def forward(self, x, demo=None):
        if demo is None:                       # allow running without demographics
            demo = x.new_zeros(x.size(0), self.demo_enc[0].in_features)
        e = self.demo_enc(demo)
        h = self.patch_embed(x).transpose(1, 2) + self.pos
        for blk in self.blocks:
            h = blk(h, e)
        pooled = self.norm(h).mean(dim=1)
        if self.has_shape_head:
            self.last_shape_logits = self.shape_head(pooled)
        return self.head(pooled)


def build(name, **kw):
    return {"cnn": CNN1D, "resnet": ResNet1D, "dmt": DMT}[name](**kw)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
