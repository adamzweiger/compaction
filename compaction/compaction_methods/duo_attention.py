# compaction/compaction_methods/duo_attention.py
# implementation based on kvpress/presses/duo_attention_press.py (https://github.com/NVIDIA/kvpress/blob/dafafcb2968b2e52fd147539e1669a71e780ef56/kvpress/presses/duo_attention_press.py)
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
DuoAttention compaction method.

This adapts the open-source DuoAttention press from kvpress to our
compaction framework. Rather than modifying the KV tensors, it keeps
the original cache and records which heads should operate in streaming
mode so that the attention kernel can mask the middle of the article
during generation.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import requests

from .base import FullCacheCompactionAlgorithm
from ..query_generation import QueryConfig

# Mapping between HuggingFace model names and DuoAttention pattern subdirectories.
# The directory is expected to contain config.json and full_attention_heads.tsv
PATTERNS_DICT = {
    "meta-llama/Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10",
}

REMOTE_BASE_URL = "https://raw.githubusercontent.com/mit-han-lab/duo-attention/refs/heads/main/attn_patterns"

@lru_cache(maxsize=32)
def _load_pattern_files(model_name: str) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Load DuoAttention pattern config and scores from disk if available,
    otherwise fall back to downloading from the official repository.
    """
    rel_path = PATTERNS_DICT[model_name]
    base_url = f"{REMOTE_BASE_URL}/{rel_path}"
    config = requests.get(f"{base_url}/config.json", timeout=10).json()
    text = requests.get(f"{base_url}/full_attention_heads.tsv", timeout=10).text
    head_scores = np.loadtxt(StringIO(text), dtype=float, delimiter="\t")
    head_scores = np.clip(head_scores, 0.0, 1.0)
    return config, head_scores


@dataclass
class DuoAttentionCompaction(FullCacheCompactionAlgorithm):
    """
    DuoAttention: Hybrid attention with retrieval and streaming heads.

    Parameters
    ----------
    head_compaction_ratio : float
        Fraction of attention heads to convert to streaming heads.
    on_the_fly_scoring : bool
        Whether to compute attention patterns on the fly.
    """

    head_compaction_ratio: float = 0.0
    on_the_fly_scoring: bool = False
    config_name: Optional[str] = None

    compaction_ratio_: float = field(init=False, default=None)
    recent_size: int = field(init=False, default=None)
    sink_size: int = field(init=False, default=None)
    streaming_mask: Optional[torch.Tensor] = field(init=False, default=None)

    def name(self) -> str:
        return self.config_name or "duo_attention"

    def __post_init_from_model__(self, model):
        """
        Initialize sink_size, recent_size, and streaming_mask from a model.
        """
        # Always use sink_size=0 and recent_size=0 (full streaming, no windows)
        self.sink_size = 0
        self.recent_size = 0

        # Load head scores to determine which heads should be streaming
        if self.on_the_fly_scoring:
            raise Exception("On the fly not supported")
        else:
            _, head_scores = self.load_attention_pattern(model)
            # Ignore the sink_size and recent_size from config - we use 0, 0

        n_layers, n_kv_heads = head_scores.shape
        n_pruned = round(head_scores.size * max(0.0, min(1.0, self.head_compaction_ratio)))

        streaming_mask = torch.zeros((n_layers, n_kv_heads), dtype=torch.bool, device=model.device)
        if n_pruned > 0:
            flat_indices = np.argsort(head_scores, axis=None)[:n_pruned]
            rows, cols = np.unravel_index(flat_indices, head_scores.shape)
            streaming_mask[torch.from_numpy(rows).long(), torch.from_numpy(cols).long()] = True

        self.streaming_mask = streaming_mask

    def load_attention_pattern(self, model):
        """
        Load attention pattern from disk or the DuoAttention repository.
        """
        model_name = model.config.name_or_path
        config, head_scores = _load_pattern_files(model_name)

        # Sanity-check that the pattern matches model dimensions
        num_layers = len(model.model.layers)
        num_kv_heads = model.config.num_key_value_heads
        if head_scores.shape != (num_layers, num_kv_heads):
            raise ValueError(
                f"Pattern shape {head_scores.shape} does not match model "
                f"({num_layers} layers, {num_kv_heads} KV heads)."
            )
        return config, head_scores

    def compact_kv_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: object,
        tokenizer: object,
        formatted_context: str,
        compute_stats: bool = False,
        vllm_model: Optional[object] = None,
        verbose_logging: bool = False,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        num_layers = len(past_key_values)

        # DuoAttention does not support sliding window layers
        if sliding_layer_indices:
            raise NotImplementedError(
                f"DuoAttention does not support models with sliding window layers. "
                f"Found {len(sliding_layer_indices)} sliding layers: {sorted(sliding_layer_indices)}"
            )

        batch_size, num_kv_heads, seq_len, head_dim = past_key_values[0][0].shape

        if batch_size != 1:
            raise NotImplementedError("DuoAttention currently only supports batch_size=1")

        if compute_stats:
            print("Warning: compute_stats has no effect for DuoAttention (cache is unchanged).")

        # Compute head_compaction_ratio from target_size
        # target_size specifies how much of the article to compact
        # For DuoAttention, we interpret this as the fraction of heads to make streaming
        if indices is None:
            article_start = 0
            article_end = seq_len
            article_len = seq_len
        else:
            article_start = indices.start
            article_end = indices.stop
            article_len = max(0, article_end - article_start)

        # Handle partial compaction: compute sub-target size for article portion
        is_partial_compaction = indices is not None
        if is_partial_compaction:
            num_to_keep = seq_len - article_len
            sub_target_size = target_size - num_to_keep

            if sub_target_size <= 0:
                raise ValueError(
                    f"target_size ({target_size}) must be greater than the number of "
                    f"positions to keep ({num_to_keep}). Got sub_target_size = {sub_target_size}"
                )

            # head_compaction_ratio = fraction of article to remove
            # If sub_target_size = 0.1 * article_len, we want to compact 90% of heads
            compaction_ratio = 1.0 - (sub_target_size / article_len) if article_len > 0 else 0.0
        else:
            # Full compaction: target_size directly specifies compaction
            compaction_ratio = 1.0 - (target_size / seq_len) if seq_len > 0 else 0.0

        # Override the head_compaction_ratio with computed value
        self.head_compaction_ratio = max(0.0, min(1.0, compaction_ratio))

        print(f"\n{'='*60}")
        print(f"DuoAttention compaction")
        print(f"  Article range: [{article_start}, {article_end}) ({article_len} tokens)")
        print(f"  Target size: {target_size}")
        print(f"  Head compaction ratio: {self.head_compaction_ratio:.2%}")
        print(f"{'='*60}")

        # Initialize DuoAttention parameters for the current model.
        # This will set sink_size=0, recent_size=0, and compute streaming_mask
        self.__post_init_from_model__(model)

        print(f"  Sink size: {self.sink_size}, Recent size: {self.recent_size}")
        print(f"{'='*60}")

        if self.streaming_mask is None:
            raise RuntimeError("Streaming mask failed to initialize.")

        if self.streaming_mask.shape != (num_layers, num_kv_heads):
            raise ValueError(
                f"Streaming mask shape {self.streaming_mask.shape} does not match "
                f"({num_layers}, {num_kv_heads}) from the cache."
            )

        sink_keep = min(self.sink_size, article_len)
        remaining = max(article_len - sink_keep, 0)
        recent_keep = min(self.recent_size, remaining)
        middle_tokens = max(article_len - sink_keep - recent_keep, 0)

        streaming_fraction = self.streaming_mask.float().mean().item()
        effective_seq_len = seq_len - streaming_fraction * middle_tokens
        effective_seq_len = max(1.0, effective_seq_len)

        effective_article_tokens = max(0.0, article_len - streaming_fraction * middle_tokens)

        self.compaction_ratio_ = streaming_fraction * (1 - (sink_keep + recent_keep) / max(1, seq_len))

        stats = {
            "method": self.name(),
            "tensor_compacted_seq_len": seq_len,
            "effective_compacted_seq_len": effective_seq_len,
            "effective_article_tokens": effective_article_tokens,
            "tensor_article_tokens": float(article_len),
            "duo_attention": {
                "sink_size": sink_keep,
                "recent_size": recent_keep,
                "streaming_fraction": streaming_fraction,
                "middle_tokens": middle_tokens,
                "article_tokens": article_len,
                "article_start": article_start,
                "article_end": article_end,
            },
        }

        # Make stats compatible with QA evaluator expectations
        if compute_stats:
            stats["per_layer_head_metrics"] = {}
            stats["train_stats_time"] = 0.0

        if "query_generation" not in stats:
            stats["query_generation"] = {
                "query_generation_time": 0.0,
                "final_n_queries_per_kv_head": 0,
                "methods_used": {},
            }

        # Build compacted cache: zero out K/V and set beta=-inf for streaming heads on middle tokens
        compacted_cache = []
        device = past_key_values[0][0].device
        dtype = past_key_values[0][0].dtype

        # Compute the middle range (positions to zero out in streaming heads)
        middle_start = article_start + sink_keep
        middle_end = article_end - recent_keep

        for layer_idx, (layer_keys, layer_values) in enumerate(past_key_values):
            # Clone the full cache
            K = layer_keys.clone()  # (B, num_kv_heads, seq_len, head_dim)
            V = layer_values.clone()  # (B, num_kv_heads, seq_len, head_dim)
            beta = torch.zeros(batch_size, num_kv_heads, seq_len, dtype=dtype, device=device)

            # For each head, if it's a streaming head, zero out middle positions
            for head_idx in range(num_kv_heads):
                if self.streaming_mask[layer_idx, head_idx]:
                    # This is a streaming head - zero out the middle tokens
                    if middle_start < middle_end:
                        K[:, head_idx, middle_start:middle_end, :] = 0
                        V[:, head_idx, middle_start:middle_end, :] = 0
                        beta[:, head_idx, middle_start:middle_end] = float('-inf')

            compacted_cache.append((K, beta, V))

        return tuple(compacted_cache), stats
