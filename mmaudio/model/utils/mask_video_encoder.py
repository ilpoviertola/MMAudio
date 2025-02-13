from functools import partial
import logging

import torch
from torch import nn
from timm.layers import trunc_normal_
from einops import rearrange

from mmaudio.ext.synchformer.vit_helper import PatchEmbed3D, DividedSpaceTimeBlock
from mmaudio.ext.synchformer.motionformer import SpatialTransformerEncoderLayer

log = logging.getLogger()


class MaskVideoEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        temporal_resolution: int = 8,
        in_chans: int = 1,
        patch_size: int = 16,
        z_block_size: int = 2,
        embed_dim: int = 768,
        flatten: bool = True,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        qkv_bias: bool = True,
        factorize_space_time: bool = True,
        aggregate_space: bool = False,
    ) -> None:
        super().__init__()
        self.temporal_resolution = temporal_resolution
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.factorize_space_time = factorize_space_time

        self.patch_3d = PatchEmbed3D(
            img_size=img_size,
            temporal_resolution=temporal_resolution,
            in_chans=in_chans,
            patch_size=patch_size,
            z_block_size=z_block_size,
            embed_dim=embed_dim,
            flatten=flatten,
        )
        self.patch_pe = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.temp_pe = nn.Parameter(torch.zeros(1, temporal_resolution, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                DividedSpaceTimeBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        # TODO: Add spatial aggregation module here
        self.final_norm = norm_layer(embed_dim)

        self.aggregate_space = aggregate_space
        if aggregate_space:
            self.factorize_space_time = True
            self.aggregation_mod = SpatialTransformerEncoderLayer(  # type: ignore
                d_model=768,
                nhead=8,
                activation=nn.GELU(),
                batch_first=True,
                dim_feedforward=int(mlp_ratio * 768),
                dropout=0.0,
                layer_norm_eps=1e-6,
                norm_first=True,
            )

    def _restore_spatio_temp_dims(
        self, feats: torch.Tensor, orig_shape: tuple
    ) -> torch.Tensor:
        """
        feats are of shape (B*S, T, D) where T = 1 + (224 // 16) * (224 // 16) * 8
        Our goal is to make them of shape (B*S, t, h, w, D) where h, w are the spatial dimensions.
        From `self.patch_embed_3d`, it follows that we could reshape feats with:
            `feats.transpose(1, 2).view(B*S, D, t, h, w)`
        """
        B, S, C, T, H, W = orig_shape
        D = self.embed_dim

        # num patches in each dimension
        t = T // self.patch_3d.z_block_size
        h = self.patch_3d.height
        w = self.patch_3d.width

        feats = feats.permute(0, 2, 1)  # (B*S, D, T)
        feats = feats.view(B * S, D, t, h, w)  # (B*S, D, t, h, w)
        return feats

    def _embed_masks(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        B, S, T, C, H, W = x.shape
        x = x.view(B * S, T, C, H, W).permute(0, 2, 1, 3, 4)
        log
        x = self.patch_3d(x)
        cls_tokens = self.cls_token.expand(B * S, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_pos_embed = self.patch_pe[:, :1]
        tile_pos_embed = self.patch_pe[:, 1:].repeat(1, self.temporal_resolution, 1)
        tile_temporal_embed = self.temp_pe.repeat_interleave(self.num_patches, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat((cls_pos_embed, total_pos_embed), dim=1)
        x += total_pos_embed

        for blk in self.blocks:
            x = blk(x, seq_len=self.num_patches, num_frames=self.temporal_resolution)
        x = self.final_norm(x[:, 1:])  # cut off CLS token

        if self.factorize_space_time:
            x = self._restore_spatio_temp_dims(x, (B, S, C, T, H, W))
        return x.view(B, S, *x.shape[1:])

    def forward(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        """
        Embeds mask video.

        Arguments:
            x (torch.Tensor): mask video to embed

        Returns:
            torch.Tensor: embedded mask video
        """
        b, t, c, h, w = x.shape
        assert c == 1 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size : i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self._embed_masks(x[i : i + batch_size]))
        x = torch.cat(outputs, dim=0)[:, 0, ...]

        if self.aggregate_space:
            x = self.aggregation_mod(x)
            x = rearrange(x, "(b s) t d -> b (s t) d", b=b)
        return x
