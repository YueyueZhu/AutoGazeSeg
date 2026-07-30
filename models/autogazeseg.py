from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep
from torch import Tensor


def _as_text_tokens(text_embedding: Tensor) -> Tensor:
    """Normalize supported text-embedding layouts to ``[B, N, D]``."""
    if text_embedding.dim() == 2:
        return text_embedding.unsqueeze(1)
    if text_embedding.dim() == 3:
        return text_embedding
    if text_embedding.dim() == 4:
        if text_embedding.shape[1] == 1:
            return text_embedding.squeeze(1)
        return rearrange(text_embedding, "b m n d -> b (m n) d")
    raise ValueError(f"Unsupported text_embedding shape: {tuple(text_embedding.shape)}")


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class CrossAttention(nn.Module):
    """Cross-attention used to inject textual guidance into image features."""

    def __init__(self, dim: int, beta: float = 2.35, gate_init: float = -2.0):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim * 2)
        self.v_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.beta = beta
        self.eps = 1e-6

    def forward(self, query: Tensor, context: Tensor, sample: Optional[bool] = None) -> Tensor:
        if sample is None:
            sample = self.training

        _, _, channels = query.shape
        q = self.q_proj(self.norm_q(query))

        k_out = self.k_proj(context)
        k_mu, k_logvar = k_out[..., :channels], k_out[..., channels:]
        k_mu = self.norm_k(k_mu)
        k_var = F.softplus(k_logvar) + self.eps

        v_out = self.v_proj(context)
        v_mu, v_logvar = v_out[..., :channels], v_out[..., channels:]
        v_mu = self.norm_v(v_mu)
        v_var = F.softplus(v_logvar) + self.eps

        mean_scores = torch.matmul(q, k_mu.transpose(1, 2)) / math.sqrt(channels)
        var_penalty = torch.matmul(q.pow(2), k_var.transpose(1, 2)) / channels
        scores = mean_scores - self.beta * torch.sqrt(torch.clamp(var_penalty, min=self.eps))
        attention = F.softmax(scores, dim=-1)

        if sample:
            noise = torch.randn_like(v_var)
            value = v_mu + torch.sqrt(v_var) * noise
        else:
            value = v_mu

        delta = torch.matmul(attention, value)
        delta = self.out_proj(delta)
        return query + torch.sigmoid(self.gate) * delta


class TwoWayCrossAttention(nn.Module):
    """Exchange information between image and text tokens in both directions."""

    def __init__(self, dim: int, beta: float = 2.35, gate_init: float = -2.0):
        super().__init__()
        self.img_to_txt = CrossAttention(dim, beta=beta, gate_init=gate_init)
        self.txt_to_img = CrossAttention(dim, beta=beta, gate_init=gate_init)

    def forward(self, image_tokens: Tensor, text_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        image_fused = self.img_to_txt(image_tokens, text_tokens)
        text_fused = self.txt_to_img(text_tokens, image_fused)
        return image_fused, text_fused


class CrossAttentionAdapter(nn.Module):
    """Project image and text tokens into a shared cross-attention space."""

    def __init__(
        self,
        in_channels_vis: int,
        in_channels_txt: int,
        adapter_channels: int = 256,
        beta: float = 2.35,
        gate_init: float = -2.0,
    ):
        super().__init__()
        self.proj_vis_down = nn.Linear(in_channels_vis, adapter_channels, bias=False)
        self.proj_txt_down = nn.Linear(in_channels_txt, adapter_channels, bias=False)
        self.proj_vis_up = nn.Linear(adapter_channels, in_channels_vis, bias=False)
        self.proj_txt_up = nn.Linear(adapter_channels, in_channels_txt, bias=False)
        self.two_way = TwoWayCrossAttention(adapter_channels, beta=beta, gate_init=gate_init)

    def forward(self, visual_tokens: Tensor, text_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        visual_low = self.proj_vis_down(visual_tokens)
        text_low = self.proj_txt_down(text_tokens)
        visual_fused, text_fused = self.two_way(visual_low, text_low)
        visual_delta = self.proj_vis_up(visual_fused - visual_low)
        text_delta = self.proj_txt_up(text_fused - text_low)
        return visual_delta, text_delta


class CrossAttentionFusion2D(nn.Module):
    """Fuse a two-dimensional U-Net feature map with textual guidance."""

    def __init__(
        self,
        image_channels: int,
        text_dim: int,
        adapter_channels: int = 256,
        beta: float = 2.35,
        gate_init: float = -2.0,
        dropout: float = 0.0,
        max_text_tokens: int = 128,
    ):
        super().__init__()
        self.max_text_tokens = max_text_tokens
        self.adapter = CrossAttentionAdapter(
            in_channels_vis=image_channels,
            in_channels_txt=text_dim,
            adapter_channels=adapter_channels,
            beta=beta,
            gate_init=gate_init,
        )
        self.norm = nn.GroupNorm(_valid_group_count(image_channels), image_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def _pool_text_tokens(self, text_tokens: Tensor) -> Tensor:
        if self.max_text_tokens <= 0 or text_tokens.shape[1] <= self.max_text_tokens:
            return text_tokens
        text_tokens = text_tokens.transpose(1, 2)
        text_tokens = F.adaptive_avg_pool1d(text_tokens, self.max_text_tokens)
        return text_tokens.transpose(1, 2)

    def forward(self, image_feature: Tensor, text_tokens: Tensor) -> Tensor:
        _, _, height, width = image_feature.shape
        text_tokens = self._pool_text_tokens(text_tokens)
        visual_tokens = rearrange(image_feature, "b c h w -> b (h w) c")
        visual_delta, _ = self.adapter(visual_tokens, text_tokens)
        visual_delta = rearrange(
            visual_delta,
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )
        return self.norm(image_feature + self.dropout(visual_delta))


class TextFeatureModulation(nn.Module):
    def __init__(self, text_dim: int, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

    def forward(self, feature: Tensor, text_tokens: Tensor) -> Tensor:
        pooled_text = text_tokens.mean(dim=1)
        gamma, beta = self.proj(pooled_text).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        return feature * (1.0 + 0.1 * gamma) + 0.1 * beta


class AutoGazeSeg(nn.Module):
    """One text-guided U-Net branch of the dual-branch AutoGazeSeg framework."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        feat_dim: int = 128,
        text_embedding_dim: int = 2560,
        unet_features: Sequence[int] = (64, 128, 256, 512, 1024, 128),
        adapter_channels: int = 256,
        beta: float = 2.35,
        gate_init: float = -2.0,
        fusion_dropout: float = 0.0,
        max_text_tokens: int = 128,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.text_embedding_dim = text_embedding_dim
        features = ensure_tuple_rep(unet_features, 6)

        self.encoder = BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feat_dim,
            features=features,
            norm=("group", {"num_groups": 4}),
        )

        self.fuse_x4 = CrossAttentionFusion2D(
            image_channels=features[4],
            text_dim=text_embedding_dim,
            adapter_channels=adapter_channels,
            beta=beta,
            gate_init=gate_init,
            dropout=fusion_dropout,
            max_text_tokens=max_text_tokens,
        )
        self.fuse_u4 = CrossAttentionFusion2D(
            image_channels=features[3],
            text_dim=text_embedding_dim,
            adapter_channels=adapter_channels,
            beta=beta,
            gate_init=gate_init,
            dropout=fusion_dropout,
            max_text_tokens=max_text_tokens,
        )
        self.fuse_u3 = CrossAttentionFusion2D(
            image_channels=features[2],
            text_dim=text_embedding_dim,
            adapter_channels=adapter_channels,
            beta=beta,
            gate_init=gate_init,
            dropout=fusion_dropout,
            max_text_tokens=max_text_tokens,
        )
        self.fuse_u2 = CrossAttentionFusion2D(
            image_channels=features[1],
            text_dim=text_embedding_dim,
            adapter_channels=adapter_channels,
            beta=beta,
            gate_init=gate_init,
            dropout=fusion_dropout,
            max_text_tokens=max_text_tokens,
        )

        self.final_film = TextFeatureModulation(text_embedding_dim, feat_dim)
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
        )
        self.classifier = nn.Conv2d(feat_dim, out_channels, kernel_size=1)

    def forward(self, image: Tensor, text_embedding: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        text_tokens = _as_text_tokens(text_embedding).to(dtype=image.dtype, device=image.device)

        x0 = self.encoder.conv_0(image)
        x1 = self.encoder.down_1(x0)
        x2 = self.encoder.down_2(x1)
        x3 = self.encoder.down_3(x2)
        x4 = self.encoder.down_4(x3)

        x4 = self.fuse_x4(x4, text_tokens)

        u4 = self.encoder.upcat_4(x4, x3)
        u4 = self.fuse_u4(u4, text_tokens)

        u3 = self.encoder.upcat_3(u4, x2)
        u3 = self.fuse_u3(u3, text_tokens)

        u2 = self.encoder.upcat_2(u3, x1)
        u2 = self.fuse_u2(u2, text_tokens)

        u1 = self.encoder.upcat_1(u2, x0)
        feature = self.encoder.final_conv(u1)
        feature = self.final_film(feature, text_tokens)

        head_output = self.head(feature)
        logits = self.classifier(head_output)

        return {
            "logits": logits,
            "feature": feature,
            "decoder_features": {
                "x1": u1,
                "x2": u2,
                "x3": u3,
                "x4": u4,
                "bottleneck": x4,
            },
        }


class TwoConv(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__()
        conv_0 = Convolution(
            spatial_dims,
            in_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        conv_1 = Convolution(
            spatial_dims,
            out_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Union[nn.Module, str, None] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        is_pad: bool = True,
    ):
        super().__init__()
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            cat_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns * 2, out_chns, act, norm, bias, dropout)
        self.is_pad = is_pad

    def forward(self, x: Tensor, skip: Optional[Tensor]) -> Tensor:
        upsampled = self.upsample(x)
        if skip is not None and torch.jit.isinstance(skip, Tensor):
            if self.is_pad:
                dimensions = len(x.shape) - 2
                padding = [0] * (dimensions * 2)
                for index in range(dimensions):
                    if skip.shape[-index - 1] != upsampled.shape[-index - 1]:
                        padding[index * 2 + 1] = 1
                upsampled = F.pad(upsampled, padding, "replicate")
            return self.convs(torch.cat([skip, upsampled], dim=1))
        return self.convs(upsampled)


class BasicUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        feature_channels = ensure_tuple_rep(features, 6)

        self.conv_0 = TwoConv(
            spatial_dims,
            in_channels,
            feature_channels[0],
            act,
            norm,
            bias,
            dropout,
        )
        self.down_1 = Down(
            spatial_dims,
            feature_channels[0],
            feature_channels[1],
            act,
            norm,
            bias,
            dropout,
        )
        self.down_2 = Down(
            spatial_dims,
            feature_channels[1],
            feature_channels[2],
            act,
            norm,
            bias,
            dropout,
        )
        self.down_3 = Down(
            spatial_dims,
            feature_channels[2],
            feature_channels[3],
            act,
            norm,
            bias,
            dropout,
        )
        self.down_4 = Down(
            spatial_dims,
            feature_channels[3],
            feature_channels[4],
            act,
            norm,
            bias,
            dropout,
        )

        self.upcat_4 = UpCat(
            spatial_dims,
            feature_channels[4],
            feature_channels[3],
            feature_channels[3],
            act,
            norm,
            bias,
            dropout,
            upsample,
        )
        self.upcat_3 = UpCat(
            spatial_dims,
            feature_channels[3],
            feature_channels[2],
            feature_channels[2],
            act,
            norm,
            bias,
            dropout,
            upsample,
        )
        self.upcat_2 = UpCat(
            spatial_dims,
            feature_channels[2],
            feature_channels[1],
            feature_channels[1],
            act,
            norm,
            bias,
            dropout,
            upsample,
        )
        self.upcat_1 = UpCat(
            spatial_dims,
            feature_channels[1],
            feature_channels[0],
            feature_channels[5],
            act,
            norm,
            bias,
            dropout,
            upsample,
            halves=False,
        )
        self.final_conv = Conv["conv", spatial_dims](
            feature_channels[5],
            out_channels,
            kernel_size=1,
        )

    def forward(self, image: Tensor) -> Tensor:
        x0 = self.conv_0(image)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)
        return self.final_conv(u1)


__all__ = ["AutoGazeSeg"]
