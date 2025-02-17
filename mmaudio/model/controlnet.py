import logging
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from einops import rearrange

from mmaudio.model.transformer_layers import MMDitSingleBlock

log = logging.getLogger()


@dataclass
class ControlNetSchema:
    """Here we can:
    1. Keep sync_f in extended_c and add mask_f to extended_c.
    2. Add mask_f to the latent and keep sync_f in extended_c.
    3. Remove sync_f from extended_c and add mask_f to extended_c.
    """

    add_mask_f_to_latent: bool
    use_extended_c: bool
    use_global_c: bool
    add_mask_f_to_extended_c: bool
    add_mask_f_to_global_c: bool
    use_preprocess_conv: bool

    def __post_init__(self):
        assert (
            self.add_mask_f_to_latent
            or self.add_mask_f_to_extended_c
            or self.add_mask_f_to_global_c
        ), f"{self.add_mask_f_to_latent=} {self.add_mask_f_to_extended_c=} {self.add_mask_f_to_global_c=}"

        assert (
            self.use_extended_c or self.use_global_c
        ), f"{self.use_extended_c=} {self.use_global_c=}"

        if self.add_mask_f_to_extended_c:
            assert self.use_extended_c, f"{self.use_extended_c=}"

        if self.add_mask_f_to_global_c:
            assert self.use_global_c, f"{self.use_global_c=}"


@dataclass
class ControlNetAggregationSchema:
    """Few approaches here:
    1. Use a control net for all the joint blocks and not fused blocks.
    2. Use a control net for half of the both block types.
    3. Use a control net strictly for half of the all blocks
        (not depending on the block type).
    We can do it only for latent blocks of joint part.

    TODO
        We might also want to first mask the whole condition and then gradually
        lower the mask?
    """

    for_joint_blocks: bool
    for_fused_blocks: bool
    for_latent: bool
    sum_pre_dit_block: bool
    sum_post_dit_block: bool

    def __post_init__(self):
        assert (
            self.for_joint_blocks or self.for_fused_blocks
        ), f"{self.for_joint_blocks=} {self.for_fused_blocks=}"
        if self.for_fused_blocks:
            assert self.for_latent, f"{self.for_latent=}"
        assert (
            self.sum_pre_dit_block or self.sum_post_dit_block
        ), f"{self.sum_pre_dit_block=} {self.sum_post_dit_block=}"


class DiTControlNetForLatentBlock(nn.Module):

    def __init__(
        self,
        *,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        add_mask_f_to_latent: bool,
        add_mask_f_to_extended_c: bool,
        add_mask_f_to_global_c: bool,
        use_prepocess_conv: bool,
        use_extended_c: bool,
        use_global_c: bool,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                MMDitSingleBlock(
                    dim=hidden_dim,
                    nhead=num_heads,
                    mlp_ratio=mlp_ratio,
                    pre_only=False,
                    kernel_size=3,
                    padding=1,
                )
                for _ in range(depth)
            ]
        )
        self.cfg: ControlNetSchema = ControlNetSchema(
            add_mask_f_to_latent=add_mask_f_to_latent,
            add_mask_f_to_extended_c=add_mask_f_to_extended_c,
            add_mask_f_to_global_c=add_mask_f_to_global_c,
            use_preprocess_conv=use_prepocess_conv,
            use_extended_c=use_extended_c,
            use_global_c=use_global_c,
        )

        # TODO: Why is this done in:
        #  https://github.com/EmilianPostolache/stable-audio-controlnet/blob/master/main/controlnet/controlnet.py#L222
        if self.cfg.use_preprocess_conv:
            self.preprocess_conv = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)
            nn.init.zeros_(self.preprocess_conv.weight)
        self.conv_in = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)
        nn.init.zeros_(self.conv_in.weight)
        self.conv_outs = nn.ModuleList(
            [nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False) for _ in range(depth)]
        )
        for conv_out in self.conv_outs:
            nn.init.zeros_(conv_out.weight)

    def add_mask_f_to_latent(
        self, latent: torch.Tensor, mask_f: torch.Tensor
    ) -> torch.Tensor:
        """Original way of using ControlNet."""
        latent = rearrange(latent, "b t c -> b c t")
        mask_f = rearrange(mask_f, "b t c -> b c t")
        if self.cfg.use_preprocess_conv:
            latent = self.preprocess_conv(latent) + latent
        controlnet_cond = self.conv_in(mask_f)
        latent = latent + controlnet_cond
        latent = rearrange(latent, "b c t -> b t c")
        return latent

    def add_mask_f_to_any_c(
        self, any_c: torch.Tensor, mask_f: torch.Tensor
    ) -> torch.Tensor:
        mask_f = rearrange(mask_f, "b t c -> b c t")
        any_c = rearrange(any_c, "b t c -> b c t")
        if self.cfg.use_preprocess_conv:
            any_c = self.preprocess_conv(any_c) + any_c
        controlnet_cond = self.conv_in(mask_f)
        any_c = any_c + controlnet_cond
        any_c = rearrange(any_c, "b c t -> b t c")
        return any_c

    def forward(
        self,
        latent: torch.Tensor,
        mask_f: torch.Tensor,
        extended_c: Optional[torch.Tensor],
        global_c: Optional[torch.Tensor],
        latent_rot: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.cfg.add_mask_f_to_latent:
            assert latent.shape == mask_f.shape, f"{latent.shape=} {mask_f.shape=}"
            latent = self.add_mask_f_to_latent(latent, mask_f)
            controlnet_c = extended_c if self.cfg.use_extended_c else global_c
        elif self.cfg.add_mask_f_to_extended_c:
            assert extended_c is not None, f"{extended_c=}"
            controlnet_c = self.add_mask_f_to_any_c(extended_c, mask_f)
        elif self.cfg.add_mask_f_to_global_c:
            assert global_c is not None, f"{global_c=}"
            controlnet_c = self.add_mask_f_to_any_c(global_c, mask_f)

        hidden_states = []
        for block in self.blocks:
            latent = block(latent, controlnet_c, latent_rot)
            hidden_states.append(latent)

        cn_hidden_states = []
        for i, conv_out in enumerate(self.conv_outs):
            h = rearrange(hidden_states[i], "b t c -> b c t")
            h = rearrange(conv_out(h), "b c t -> b t c")
            cn_hidden_states.append(h)
        return cn_hidden_states
