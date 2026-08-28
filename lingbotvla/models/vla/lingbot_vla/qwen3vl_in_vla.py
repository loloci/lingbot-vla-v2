import os

import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple

from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
import transformers.models.qwen3_vl.modeling_qwen3_vl as hf_qwen3vl
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
    Qwen3VLModel as _Qwen3VLModel,
    Qwen3VLTextModel as _Qwen3VLTextModel,
    Qwen3VLPreTrainedModel as _Qwen3VLPreTrainedModel,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    Qwen3VLVisionMLP,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)

# 视觉位置编码前导的两个缓存门。均默认关 ⇒ 与原字面量逐位等价。
# 背景与 bitwise 论证见 report/08_visual_pos_embed_cache/README.md §3-4。
_ROTPOS_CACHE = os.environ.get("LINGBOT_VISUAL_ROTPOS_CACHE", "0") == "1"
_POSEMB_CACHE = os.environ.get("LINGBOT_VISUAL_POSEMB_CACHE", "0") == "1"
_POSEMB_STRICT = os.environ.get("LINGBOT_VISUAL_POSEMB_CACHE_STRICT", "0") == "1"


def _qwen3vl_no_init_weights(self, module):
    return

_Qwen3VLPreTrainedModel._init_weights = _qwen3vl_no_init_weights
Qwen3VLPreTrainedModel = _Qwen3VLPreTrainedModel


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            out_fp32_atten = False
            if key_states.dtype == torch.float32:
                out_fp32_atten = True
                query_states = query_states.to(torch.bfloat16)
                key_states = key_states.to(torch.bfloat16)
                value_states = value_states.to(torch.bfloat16)
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
            if out_fp32_atten:
                attn_output = attn_output.to(torch.float32)
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: Optional[torch.Tensor] = None,
        start: Optional[int] = 0,
        end: Optional[int] = 0,
        compute_kqv: bool = False,
        output_atten: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        param_dtype = self.self_attn.q_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        if att_output is not None:
            att_output = att_output.to(param_dtype)

        if compute_kqv:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)
            query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
            key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
            return query_state, key_state, value_state

        if output_atten:
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])
            out_emb += hidden_states
            after_first_residual = out_emb.clone()
            out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)
            out_emb += after_first_residual
            return out_emb

        position_embeddings = kwargs.pop("position_embeddings", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if position_embeddings is not None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            return residual + hidden_states

        raise ValueError(
            f"Invalid operation compute_kqv={compute_kqv} and output_atten={output_atten} "
            "with Qwen3VLTextDecoderLayer in LingBot-VLA"
        )


class Qwen3VLTextModel(_Qwen3VLTextModel):
    def __init__(self, config: Qwen3VLTextConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3VLModel(_Qwen3VLModel):
    def __init__(self, config: Qwen3VLConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen3VLForConditionalGeneration(_Qwen3VLForConditionalGeneration, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


@torch.compiler.disable
def _build_posemb_plan(self, grid_thw):
    """只算 fast_pos_embed_interpolate 里每步不变的结构量。

    与 HF 原实现逐条同路（linspace 仍落 CPU、仍走 tolist/torch.tensor），
    差别只是 h/w/t 用 python int 而非 0-dim CUDA tensor（后者本来也是 __index__
    成同一个 int）⇒ idx/weight 逐位相同。
    """
    n = self.num_grid_per_side
    merge = self.config.spatial_merge_size
    idx_list, weight_list = [[] for _ in range(4)], [[] for _ in range(4)]
    ts, split, shapes = [], [], []
    for t, h, w in grid_thw.tolist():
        h_idxs = torch.linspace(0, n - 1, h)
        w_idxs = torch.linspace(0, n - 1, w)
        h_floor, w_floor = h_idxs.int(), w_idxs.int()
        h_ceil = (h_idxs.int() + 1).clip(max=n - 1)
        w_ceil = (w_idxs.int() + 1).clip(max=n - 1)
        dh, dw = h_idxs - h_floor, w_idxs - w_floor
        base_h, base_h_ceil = h_floor * n, h_ceil * n
        indices = [
            (base_h[None].T + w_floor[None]).flatten(),
            (base_h[None].T + w_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_floor[None]).flatten(),
            (base_h_ceil[None].T + w_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())
        ts.append(t)
        split.append(h * w)
        shapes.append((t, h // merge, merge, w // merge, merge, -1))
    dev = self.pos_embed.weight.device
    return {
        "key": tuple(grid_thw.shape),
        "grid_thw": grid_thw.clone(),
        "idx": torch.tensor(idx_list, dtype=torch.long, device=dev),
        "weight": torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=dev),
        "ts": ts,
        "split": split,
        "shapes": shapes,
    }


@torch.compiler.disable
def preprcess_grid_thw(self, grid_thw: torch.Tensor):
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len = int(torch.prod(grid_thw, dim=1).sum().item())
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return None, position_embeddings, cu_seqlens, split_sizes, max_seqlen


@torch.compiler.disable
def fast_pos_embed_interpolate_cached(self, grid_thw):
    """缓存结构量，把 gather-加权和留在线上每步跑。

    ⛔ 不缓存输出：pos_embed 是可训练的 nn.Embedding（freeze_vision_encoder=false）。
    """
    plan = getattr(self, "_lingbot_posemb_plan", None)
    if plan is None or plan["key"] != tuple(grid_thw.shape):
        plan = _build_posemb_plan(self, grid_thw)
        self._lingbot_posemb_plan = plan
        logger.info(f"LINGBOT_VISUAL_POSEMB_CACHE: plan built for grid_thw{plan['key']}")
    elif _POSEMB_STRICT and not torch.equal(grid_thw, plan["grid_thw"]):
        raise RuntimeError("grid_thw 内容变了但形状没变 —— 结构量缓存会过期")

    pos_embeds = self.pos_embed(plan["idx"]) * plan["weight"][:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    out = []
    for pos_embed, t, shape in zip(patch_pos_embeds.split(plan["split"]), plan["ts"], plan["shapes"]):
        pos_embed = pos_embed.repeat(t, 1)
        out.append(pos_embed.view(*shape).permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
    return torch.cat(out)


def forward_without_grid_thw(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor = None,
    pos_embeds: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)

    # pos_embeds 恒为 None（preprcess_grid_thw 的第一个返回值就是 None）⇒ 原条件恒真
    # ⇒ 外层 precompute_grid_thw 缓存的另外三个量从未生效。开门后只看那三个。
    need = position_embeddings is None or cu_seqlens is None or max_seqlen is None
    if not _ROTPOS_CACHE:
        need = need or pos_embeds is None
    if need:
        pos_embeds, position_embeddings, cu_seqlens, _, max_seqlen = self.preprcess_grid_thw(grid_thw)
    if pos_embeds is None:
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

    hidden_states = hidden_states + pos_embeds
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)
    return hidden_states, deepstack_feature_lists


def apply_lingbot_qwen3_vl_patch():
    logger.info_rank0("apply Qwen3-VL Lingbot patch")
    hf_qwen3vl.Qwen3VLPreTrainedModel = Qwen3VLPreTrainedModel
    hf_qwen3vl.Qwen3VLTextDecoderLayer = Qwen3VLTextDecoderLayer
    hf_qwen3vl.Qwen3VLTextModel = Qwen3VLTextModel
    hf_qwen3vl.Qwen3VLModel = Qwen3VLModel
    hf_qwen3vl.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    hf_qwen3vl.Qwen3VLVisionAttention = Qwen3VLVisionAttention
    hf_qwen3vl.Qwen3VLVisionBlock = Qwen3VLVisionBlock
    hf_qwen3vl.Qwen3VLVisionModel.forward = forward_without_grid_thw
    hf_qwen3vl.Qwen3VLVisionModel.preprcess_grid_thw = preprcess_grid_thw
    if _POSEMB_CACHE:
        hf_qwen3vl.Qwen3VLVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate_cached
        logger.info("LINGBOT_VISUAL_POSEMB_CACHE=1: fast_pos_embed_interpolate 只缓存结构量。")
    if _ROTPOS_CACHE:
        logger.info("LINGBOT_VISUAL_ROTPOS_CACHE=1: rot_pos_emb 复用外层 precompute_grid_thw 缓存。")
