# compaction/compaction_methods/chunked.py
"""
Chunked compaction wrapper.

This module implements chunked compaction for ultra-long contexts. Because attention
matching can select and compact an arbitrary subset of token indices, we can process
chunks separately and then stitch together the resulting KV caches.

We implement 2 approaches:
=============================

1. KV-Based Chunking (use_kv_based=True, default)
-------------------------------------------------
Prefill the full context once, extract each chunk's KV states, compact them, and
merge the results into the final cache.

Starting with the full sequence:
    <prefix><chunk_1><chunk_2>...<chunk_N><suffix>

For each chunk_i, we construct a KV cache for query generation:
    <prefix><chunk_i><suffix>

This is done entirely in KV space by slicing and concatenating tensors. Since these
KV states already have RoPE applied at their original positions, no RoPE correction
is needed. Self-study query tokens get RoPE positions starting at the original
sequence length. After compaction, we extract just the compacted chunk_i portion.

For sliding window layers, we always keep the original sliding window cache from
the full prefill - both for self-study query generation and in the final output.

Advantages:
- More theoretically correct: we are compacting KVs from the original sequence.
- No RoPE corrections needed (positions already correct)
- Sliding window handling is straightforward

Disadvantages:
- Somewhat complicated, especially to integrate with on-policy queries.

2. Text-Based Chunking (use_kv_based=False)
-------------------------------------------
Extract each chunk's text, prefill and compact it in isolation using local
positional indices starting at 0, then apply a RoPE phase shift to align the
compacted KV states with the chunk's original global offset before merging.

For each chunk, we:
1. Format the chunk text with its own chat template
2. Run prefill to get KV cache (positions start at 0)
3. Run query generation and compaction
4. Apply RoPE correction: rotate keys from chunk-local positions to original
   positions in the full context

RoPE Correction Math:
To shift a key from position p to position p', we apply a rotation by angle (p' - p):
    K_new = K_old * cos(p' - p) + rotate_half(K_old) * sin(p' - p)

For sliding window layers (which use a different rope_theta in models like Gemma3),
we apply the correction using the local RoPE embedding. The returned cache's sliding 
window layers are instantiated with the shifted last chunk's sliding window. The last chunk
includes the suffix tokens so sliding window layers see correct recent context.

Advantages:
- Memory efficient for very long contexts (only one chunk in memory at a time)
- Can process chunks that wouldn't fit in memory together

Disadvantages:
- Requires RoPE corrections (more complex)
- Sliding window handling requires careful position tracking
- Technically wrong: we are prefilling each chunk independently of the others.
"""
import torch
import time
from typing import Tuple, Dict, Optional, Any, List, Union, Callable

from .base import FullCacheCompactionAlgorithm
from .per_layer_head_on_policy import PerLayerHeadOnPolicyCompaction
from ..query_generation import QueryConfig
from ..chunking import ChunkingStrategy


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_to_cache(
    cache: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to cached keys.

    Parameters
    ----------
    cache : torch.Tensor
        Cached keys of shape (batch, num_heads, seq_len, head_dim)
    cos : torch.Tensor
        Cosine embeddings of shape (batch, seq_len, head_dim)
    sin : torch.Tensor
        Sine embeddings of shape (batch, seq_len, head_dim)

    Returns
    -------
    rotated_cache : torch.Tensor
        Cache with RoPE applied
    """
    # cos/sin: (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (cache * cos) + (rotate_half(cache) * sin)


def compute_rope_correction(
    model: Any,
    current_positions: torch.Tensor,
    target_positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    use_local_rope: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE correction factors to shift from current_positions to target_positions.

    To shift RoPE from position p to position p':
    K_new = apply_rotary(apply_inverse_rotary(K_old, cos_p, sin_p), cos_p', sin_p')

    Since apply_inverse_rotary uses -sin, and applying two rotations in sequence:
    K_new = K_old * (cos_p * cos_p' + sin_p * sin_p') + rotate_half(K_old) * (sin_p' * cos_p - cos_p' * sin_p)

    Which is equivalent to applying rotation with angle (p' - p):
    cos_diff = cos(p' - p), sin_diff = sin(p' - p)

    Parameters
    ----------
    model : Any
        Model with rotary_emb attribute
    current_positions : torch.Tensor
        Current position IDs (shape: seq_len or (batch, seq_len))
    target_positions : torch.Tensor
        Target position IDs (same shape as current_positions)
    device : torch.device
        Device for tensors
    dtype : torch.dtype
        Data type for tensors
    use_local_rope : bool
        If True, use the local RoPE embedding (rotary_emb_local) for models like Gemma3
        that have separate RoPE for sliding window layers. Default: False (use global RoPE).

    Returns
    -------
    cos_diff : torch.Tensor
        Cosine of position difference
    sin_diff : torch.Tensor
        Sine of position difference
    """
    # Get the rotary embedding module
    # For models with separate local RoPE (e.g., Gemma3), sliding window layers use
    # rotary_emb_local with a different rope_theta than the global rotary_emb
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model

    if use_local_rope and hasattr(base_model, 'rotary_emb_local'):
        rotary_emb = base_model.rotary_emb_local
    else:
        rotary_emb = base_model.rotary_emb

    # Ensure positions are 2D: (batch, seq_len)
    if current_positions.dim() == 1:
        current_positions = current_positions.unsqueeze(0)
    if target_positions.dim() == 1:
        target_positions = target_positions.unsqueeze(0)

    # Compute position difference
    position_diff = target_positions - current_positions

    # Create a dummy tensor with correct dtype for rotary_emb
    dummy = torch.zeros(1, 1, rotary_emb.inv_freq.shape[0] * 2, device=device, dtype=dtype)

    # Get cos/sin for the position difference
    cos_diff, sin_diff = rotary_emb(dummy, position_diff.to(device))

    return cos_diff.to(dtype), sin_diff.to(dtype)


class ChunkedCompaction(FullCacheCompactionAlgorithm):
    """
    Chunked compaction that processes articles in independent chunks.

    Each chunk is:
    1. Formatted with its own chat template
    2. Run through prefill to get KV cache
    3. Run through independent query generation
    4. Compacted using the specified compaction method

    The final cache concatenates all compacted chunks with the original
    chat template prefix and suffix.

    Supports any FullCacheCompactionAlgorithm.
    """

    def __init__(
        self,
        inner_compaction_method: Union[FullCacheCompactionAlgorithm, Callable[[], FullCacheCompactionAlgorithm]],
        chunking_strategy: ChunkingStrategy,
        chunk_system_prompt_template: str = "You are a helpful assistant. Below is a portion of {article_name}.",
        config_name: Optional[str] = None,
        use_kv_based: bool = True,
    ):
        """
        Initialize chunked compaction.

        Parameters
        ----------
        inner_compaction_method : FullCacheCompactionAlgorithm or callable
            The compaction method to use for each chunk. Can be:
            - A FullCacheCompactionAlgorithm instance (same instance used for all chunks)
            - A callable that returns a new FullCacheCompactionAlgorithm (creates fresh instance per chunk)
        chunking_strategy : ChunkingStrategy
            Strategy for splitting articles into chunks
        chunk_system_prompt_template : str
            System prompt template for each chunk. Use {article_name} placeholder.
            Ignored for gemma models (they don't use system prompts).
            Only used for text-based chunking (use_kv_based=False).
        config_name : str, optional
            Name of the configuration (used for logging)
        use_kv_based : bool
            If True, uses KV-based chunking where the full context is prefilled once
            and chunks are extracted from the KV cache. This avoids RoPE corrections
            but requires enough memory for the full prefill.
            If False, uses text-based chunking where each chunk is prefilled
            independently and RoPE corrections are applied. Default: True.
        """
        self.inner_compaction_method = inner_compaction_method
        self.chunking_strategy = chunking_strategy
        self.chunk_system_prompt_template = chunk_system_prompt_template
        self.config_name = config_name
        self.use_kv_based = use_kv_based

        # Determine if we have a factory or an instance
        self._is_factory = callable(inner_compaction_method) and not isinstance(inner_compaction_method, FullCacheCompactionAlgorithm)

    def _get_inner_method(self, chunk_idx: Optional[int] = None) -> FullCacheCompactionAlgorithm:
        """Get the inner compaction method, creating a new instance if factory."""
        if self._is_factory:
            method = self.inner_compaction_method()
            # Update config name if chunk_idx provided
            if chunk_idx is not None and hasattr(method, 'config_name') and self.config_name:
                method.config_name = f"{self.config_name}_chunk{chunk_idx}"
        else:
            method = self.inner_compaction_method

        # For text-based methods (like summarize), enable cache return mode
        # so they can be used as inner methods for chunked compaction
        if hasattr(method, 'return_cache') and not method.returns_cache():
            method.return_cache = True

        return method

    def name(self) -> str:
        """Return the config name if provided, otherwise a generated name."""
        if self.config_name:
            return self.config_name
        inner_name = self._get_inner_method().name()
        return f"chunked_{self.chunking_strategy.name}_{inner_name}"

    def requires_preextracted_cache(self) -> bool:
        """
        Return whether this method needs pre-extracted KV cache.

        For text-based chunking (use_kv_based=False): Returns False.
        Each chunk is processed independently with its own prefill to avoid OOM.

        For KV-based chunking (use_kv_based=True): Returns True.
        The full context must be prefilled once, then chunks are extracted
        from the KV cache.
        """
        return self.use_kv_based

    def compact_kv_cache(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        compute_stats: bool = False,
        verbose_logging: bool = False,
        vllm_model: Optional[Any] = None,
        article_text: Optional[str] = None,
        article_name: Optional[str] = None,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact the KV cache using chunked processing.

        Parameters
        ----------
        past_key_values : tuple of tuples, optional
            Original KV cache from model forward pass. Can be None if
            requires_preextracted_cache() returns False - in that case,
            the method extracts prefix/suffix caches via small prefills.
        target_size : int or float
            Target compaction ratio (0-1) applied per-chunk
        indices : range, optional
            Article indices in the original context (only used if past_key_values provided)
        query_config : QueryConfig
            Configuration for query generation
        model : Any
            Model instance
        tokenizer : Any
            Tokenizer
        formatted_context : str
            Formatted context string
        compute_stats : bool
            Whether to compute detailed statistics
        verbose_logging : bool
            Whether to save selected indices in stats
        vllm_model : Any, optional
            vLLM model for query generation
        article_text : str, optional
            Original article text (extracted from formatted_context if not provided)
        article_name : str, optional
            Article name/title for system prompt

        Returns
        -------
        compacted_cache : tuple of tuples
            Concatenated compacted cache with template wrapper
        stats : dict
            Statistics from all chunk compactions
        """
        # Dispatch to KV-based or text-based implementation
        if self.use_kv_based:
            return self._compact_kv_cache_kv_based(
                past_key_values=past_key_values,
                target_size=target_size,
                indices=indices,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=formatted_context,
                compute_stats=compute_stats,
                verbose_logging=verbose_logging,
                vllm_model=vllm_model,
                article_text=article_text,
                article_name=article_name,
                sliding_layer_indices=sliding_layer_indices,
            )
        else:
            return self._compact_kv_cache_text_based(
                past_key_values=past_key_values,
                target_size=target_size,
                indices=indices,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=formatted_context,
                compute_stats=compute_stats,
                verbose_logging=verbose_logging,
                vllm_model=vllm_model,
                article_text=article_text,
                article_name=article_name,
                sliding_layer_indices=sliding_layer_indices,
            )

    def _compact_kv_cache_text_based(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        compute_stats: bool = False,
        verbose_logging: bool = False,
        vllm_model: Optional[Any] = None,
        article_text: Optional[str] = None,
        article_name: Optional[str] = None,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """Text-based chunked compaction (original implementation)."""
        # Extract article text if not provided
        if article_text is None:
            article_text = self._extract_article_text(formatted_context, tokenizer)

        if article_name is None:
            article_name = "the document"

        # Determine device and dtype from model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # If past_key_values provided, extract info from it (legacy path)
        # Otherwise, we need to compute prefix cache ourselves
        if past_key_values is not None:
            num_layers = len(past_key_values)
            # Find a non-sliding layer to get the full sequence length
            # (sliding layers may have shorter seq_len due to window size)
            sliding_layer_indices = sliding_layer_indices or set()
            ref_layer_idx = 0
            for i in range(num_layers):
                if i not in sliding_layer_indices:
                    ref_layer_idx = i
                    break
            batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape

            # Get prefix/article boundaries from indices
            # The suffix text is included in the last chunk for correct RoPE, but we don't
            # extract a separate suffix cache
            if indices is not None:
                prefix_end = indices.start
                article_len = len(indices)  # indices.stop - indices.start
            else:
                prefix_end = 0
                article_len = seq_len

            # Extract prefix KV cache
            prefix_cache = self._extract_cache_slice(past_key_values, 0, prefix_end)

            # Extract suffix text from formatted_context for use with last chunk
            from evaluation.utils import detect_user_tags
            user_start_tag, user_end_tag = detect_user_tags(formatted_context)
            user_end_pos = formatted_context.find(user_end_tag)
            outer_suffix_text = formatted_context[user_end_pos:] if user_end_pos != -1 else ""
        else:
            # No pre-extracted cache - compute prefix cache ourselves
            # This is the memory-efficient path for long documents
            prefix_cache, prefix_end, seq_len, article_len, outer_suffix_text = \
                self._extract_prefix_cache(
                    model, tokenizer, formatted_context, article_text, device,
                    sliding_layer_indices=sliding_layer_indices
                )

            # Get num_layers from prefix_cache
            num_layers = len(prefix_cache)

        # Zero the sliding window layers of prefix_cache.
        # For sliding window layers, the prefix is outside the sliding window and irrelevant,
        # so we replace with empty tensors (size 0 in seq dim).
        if sliding_layer_indices:
            prefix_cache_list = []
            for layer_idx, (keys, values) in enumerate(prefix_cache):
                if layer_idx in sliding_layer_indices:
                    # Empty tensors with same shape except seq_len=0
                    empty_keys = keys.new_zeros(keys.shape[0], keys.shape[1], 0, keys.shape[3])
                    empty_values = values.new_zeros(values.shape[0], values.shape[1], 0, values.shape[3])
                    prefix_cache_list.append((empty_keys, empty_values))
                else:
                    prefix_cache_list.append((keys, values))
            prefix_cache = tuple(prefix_cache_list)

        # Target size should be a ratio (0-1) for per-chunk compaction
        if target_size >= 1:
            # Convert absolute target to ratio based on article length
            if article_len > 0:
                non_article_len = seq_len - article_len
                target_size = (target_size - non_article_len) / article_len
            else:
                target_size = 1.0

        target_size = min(1.0, target_size)

        print(f"\n{'='*60}")
        print(f"Chunked Compaction")
        print(f"{'='*60}")
        print(f"Chunking strategy: {self.chunking_strategy.name}")
        print(f"Inner method: {self._get_inner_method().name()}")
        print(f"Per-chunk compaction ratio: {target_size:.2%}")

        print(f"Prefix tokens: {prefix_end}")
        print(f"Suffix tokens: {seq_len - prefix_end - article_len}")
        print(f"Article tokens: {article_len}")

        # Chunk the article
        start_time = time.time()
        chunks = self.chunking_strategy.chunk(article_text, tokenizer)
        chunk_time = time.time() - start_time
        print(f"Chunked article into {len(chunks)} chunks in {chunk_time:.2f}s")

        # Process each chunk independently
        compacted_chunks = []
        chunk_rope_info = []  # Track RoPE correction info for each chunk
        chunk_stats = []
        total_original_tokens = 0
        total_compacted_tokens = 0
        total_effective_article_tokens = 0.0

        # Aggregate timing stats from inner compactions
        total_query_generation_time = 0.0
        total_inner_compaction_time = 0.0
        total_train_stats_time = 0.0

        # Store original chunk caches for test stats computation (only when compute_stats=True)
        original_chunk_caches = [] if compute_stats else None

        last_chunk_outer_suffix_len = 0  # Track suffix tokens included in last chunk

        # Get sliding window size from model config if available
        sliding_window = None
        if sliding_layer_indices:
            config = getattr(model, 'config', None)
            if config is not None:
                sliding_window = getattr(config, 'sliding_window', None)

        for chunk_idx, chunk in enumerate(chunks):
            is_last_chunk = (chunk_idx == len(chunks) - 1)

            print(f"\n--- Processing chunk {chunk_idx + 1}/{len(chunks)} ---")
            if chunk.metadata.get('note_id'):
                print(f"Note ID: {chunk.metadata['note_id']}")
            elif chunk.metadata.get('filename'):
                print(f"Filename: {chunk.metadata['filename']}")

            # Format chunk with its own system prompt (None for gemma models)
            if self._is_gemma_model(model):
                chunk_system_prompt = None
            else:
                chunk_system_prompt = self.chunk_system_prompt_template.format(article_name=article_name)

            # For the last chunk, include the outer suffix text so it gets correct attention context
            # and RoPE positions. The suffix will be extracted along with the compacted article.
            if is_last_chunk and outer_suffix_text:
                print(f"Last chunk: including outer suffix ({len(outer_suffix_text)} chars)")
                chunk_seq_len, chunk_past_key_values, chunk_article_indices, chunk_outer_suffix_len = \
                    self._extract_chunk_kv_cache(
                        model, tokenizer, chunk.text, chunk_system_prompt, device,
                        outer_suffix_text=outer_suffix_text
                    )
                last_chunk_outer_suffix_len = chunk_outer_suffix_len
            else:
                chunk_seq_len, chunk_past_key_values, chunk_article_indices, _ = \
                    self._extract_chunk_kv_cache(
                        model, tokenizer, chunk.text, chunk_system_prompt, device
                    )

            chunk_article_len = len(chunk_article_indices)
            total_original_tokens += chunk_article_len
            print(f"Chunk tokens: {chunk_seq_len} (article portion: {chunk_article_len})")

            # Compute chunk target size using the ratio
            chunk_target = max(1, int(chunk_article_len * target_size))
            chunk_full_target = chunk_target + (chunk_seq_len - chunk_article_len)
            print(f"Chunk target: {chunk_target} tokens ({target_size:.2%} of {chunk_article_len})")

            # Save original article-portion cache for test stats if requested
            if compute_stats:
                # Extract just the article portion of the original cache
                original_article_cache = self._extract_cache_slice(
                    chunk_past_key_values,
                    chunk_article_indices.start,
                    chunk_article_indices.stop,
                    is_compacted=False
                )
                original_chunk_caches.append(original_article_cache)

            # Get compaction method for this chunk
            chunk_compactor = self._get_inner_method(chunk_idx)

            # Time the inner compaction
            chunk_start_time = time.time()

            # Compact this chunk
            chunk_compacted, chunk_stat = chunk_compactor.compact_kv_cache(
                past_key_values=chunk_past_key_values,
                target_size=chunk_full_target,
                indices=chunk_article_indices,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=self._format_chunk_context(model, tokenizer, chunk.text, chunk_system_prompt),
                compute_stats=compute_stats,
                verbose_logging=verbose_logging,
                vllm_model=vllm_model,
                sliding_layer_indices=sliding_layer_indices,
            )

            chunk_total_time = time.time() - chunk_start_time

            # Extract timing from inner compaction stats
            chunk_query_gen_time = 0.0
            if 'query_generation' in chunk_stat and 'query_generation_time' in chunk_stat['query_generation']:
                chunk_query_gen_time = chunk_stat['query_generation']['query_generation_time']
            chunk_train_stats_time = chunk_stat.get('train_stats_time', 0.0)
            # Compaction time is total minus query gen and train stats
            chunk_compaction_time = chunk_total_time - chunk_query_gen_time - chunk_train_stats_time

            total_query_generation_time += chunk_query_gen_time
            total_inner_compaction_time += chunk_compaction_time
            total_train_stats_time += chunk_train_stats_time

            # Extract only the compacted article portion (remove chunk's prefix/suffix)
            # For the last chunk with outer suffix, we keep the outer suffix but remove chunk prefix
            chunk_prefix_len = chunk_article_indices.start
            chunk_suffix_len = chunk_seq_len - chunk_article_indices.stop if not (is_last_chunk and last_chunk_outer_suffix_len > 0) else 0

            # For nonuniform caches, different layers may have different compacted lengths
            # We need to handle slicing per-layer rather than using a single ref_layer_idx
            article_only_compacted_list = []
            total_compacted_article_len = 0
            num_global_layers_in_chunk = 0

            for layer_idx in range(len(chunk_compacted)):
                if layer_idx in sliding_layer_indices:
                    # Sliding layers: keep placeholder
                    article_only_compacted_list.append(chunk_compacted[layer_idx])
                    continue

                # Get this layer's compacted length
                layer_compacted_len = chunk_compacted[layer_idx][0].shape[2]

                # Compute extraction bounds for this layer
                if is_last_chunk and last_chunk_outer_suffix_len > 0:
                    # Last chunk: extract [compacted_article + outer_suffix], strip only chunk prefix
                    # The outer_suffix is at the end of the compacted chunk (it wasn't compacted)
                    layer_article_len = layer_compacted_len - chunk_prefix_len - last_chunk_outer_suffix_len
                    extraction_end = layer_compacted_len  # Include suffix
                else:
                    # Normal chunk: extract just compacted article, strip both prefix and suffix
                    layer_article_len = layer_compacted_len - chunk_prefix_len - chunk_suffix_len
                    extraction_end = chunk_prefix_len + layer_article_len

                # Slice this layer
                article_only_compacted_list.append((
                    chunk_compacted[layer_idx][0][:, :, chunk_prefix_len:extraction_end, :],
                    chunk_compacted[layer_idx][1][:, :, chunk_prefix_len:extraction_end],
                    chunk_compacted[layer_idx][2][:, :, chunk_prefix_len:extraction_end, :]
                ))

                total_compacted_article_len += layer_article_len
                num_global_layers_in_chunk += 1

            article_only_compacted = tuple(article_only_compacted_list)

            # Compute average compacted article length across global layers
            compacted_article_len = total_compacted_article_len / num_global_layers_in_chunk if num_global_layers_in_chunk > 0 else 0

            if is_last_chunk and last_chunk_outer_suffix_len > 0:
                print(f"Last chunk: extracting article ({compacted_article_len:.1f} tokens avg) + suffix ({last_chunk_outer_suffix_len} tokens)")

            total_compacted_tokens += compacted_article_len

            # Track effective article tokens from inner method (if available, otherwise use tensor len)
            chunk_effective_article_tokens = chunk_stat.get('effective_article_tokens', compacted_article_len)
            total_effective_article_tokens += chunk_effective_article_tokens

            # For the last chunk, extract sliding layer data for models with sliding window attention
            # Apply RoPE correction to shift from chunk-local positions to original context positions
            # The offset is the same as for global layers: original_start - chunk_local_start
            # We inject this data into article_only_compacted so it's part of the unified (C1, beta, C2) cache
            if is_last_chunk and sliding_layer_indices and sliding_window is not None:
                # Same offset as global layers: chunk_start_in_original - chunk_start_in_local
                original_article_start = prefix_end + chunk.start_token_idx
                rope_offset = original_article_start - chunk_prefix_len

                # Convert article_only_compacted to list for modification
                article_only_compacted_list = list(article_only_compacted)

                for layer_idx in sliding_layer_indices:
                    # Get the KV cache for this sliding layer from the chunk
                    # Note: sliding layers only store the last (sliding_window - 1) tokens
                    keys = chunk_past_key_values[layer_idx][0].clone()
                    values = chunk_past_key_values[layer_idx][1].clone()

                    sliding_cache_len = keys.shape[2]  # Actual tokens stored (up to sliding_window - 1)

                    # Compute RoPE correction for sliding layers:
                    # The sliding layer stores the last sliding_cache_len tokens of the chunk
                    # Current positions: [chunk_seq_len - sliding_cache_len, ..., chunk_seq_len - 1]
                    # Target positions: current + rope_offset (same offset as global layers)
                    # NOTE: Sliding layers use local RoPE (different rope_theta) in models like Gemma3
                    if rope_offset != 0:
                        current_start = chunk_seq_len - sliding_cache_len
                        target_start = current_start + rope_offset
                        current_positions = torch.arange(current_start, chunk_seq_len, device=device)
                        target_positions = torch.arange(target_start, target_start + sliding_cache_len, device=device)
                        cos_diff, sin_diff = compute_rope_correction(
                            model, current_positions, target_positions, device, dtype,
                            use_local_rope=True  # Sliding layers use local RoPE
                        )
                        keys = apply_rotary_pos_emb_to_cache(keys, cos_diff, sin_diff)

                    # Store sliding layer data in (C1=keys, beta=0, C2=values) format
                    beta = torch.zeros(keys.shape[0], keys.shape[1], keys.shape[2], device=device, dtype=dtype)
                    article_only_compacted_list[layer_idx] = (keys, beta, values)

                article_only_compacted = tuple(article_only_compacted_list)

                print(f"Injected sliding layer data from last chunk for {len(sliding_layer_indices)} layers "
                      f"(sliding_cache_len={sliding_cache_len}, rope_offset={rope_offset})")

            # RoPE Correction Strategy:
            # Both global and sliding layers are shifted by the same offset:
            #   rope_offset = original_article_start - chunk_local_start
            #                = (prefix_end + chunk.start_token_idx) - chunk_prefix_len
            # This shifts all keys from their chunk-local positions to their original positions
            # in the full context (outer_prefix + article + outer_suffix).
            # Note that sliding layers have a different rope theta.
            #
            # For global layers: applied in _concatenate_with_template
            # For sliding layers: applied above when extracting sliding layer (last chunk only)
            original_article_offset = prefix_end + chunk.start_token_idx  # Position in original full context

            # For last chunk, RoPE info covers both article and suffix portions
            if is_last_chunk and last_chunk_outer_suffix_len > 0:
                rope_length = compacted_article_len + last_chunk_outer_suffix_len
            else:
                rope_length = compacted_article_len

            chunk_rope_info.append({
                'chunk_local_start': chunk_prefix_len,  # RoPE positions used during chunk processing
                'original_article_start': original_article_offset,  # Original position in full context
                'length': rope_length,
                'includes_suffix': is_last_chunk and last_chunk_outer_suffix_len > 0,
                'suffix_len': last_chunk_outer_suffix_len if is_last_chunk else 0,
            })
            print(f"RoPE correction: chunk-local [{chunk_prefix_len}:{chunk_prefix_len + chunk_article_len}] "
                  f"-> original [{original_article_offset}:{original_article_offset + chunk_article_len}]")

            compacted_chunks.append(article_only_compacted)
            chunk_stat['chunk_idx'] = chunk_idx
            chunk_stat['chunk_original_tokens'] = chunk_article_len
            chunk_stat['chunk_compacted_tokens'] = compacted_article_len
            chunk_stat['chunk_timing'] = {
                'query_generation_time': chunk_query_gen_time,
                'compaction_time': chunk_compaction_time,
                'train_stats_time': chunk_train_stats_time,
                'total_time': chunk_total_time,
            }
            # Enhanced rope correction info with more context
            rope_info = chunk_rope_info[-1].copy()
            rope_info['rope_offset_applied'] = rope_info['original_article_start'] - rope_info['chunk_local_start']
            chunk_stat['rope_correction'] = rope_info
            chunk_stats.append(chunk_stat)

            # Free chunk cache to save memory
            del chunk_past_key_values
            del chunk_compacted

        # Concatenate all compacted chunks with prefix, applying RoPE corrections
        # The suffix is already included in the last chunk for correct RoPE/sliding window
        print(f"\n--- Concatenating {len(compacted_chunks)} compacted chunks with RoPE correction ---")
        combined_cache = self._concatenate_with_template(
            compacted_chunks, prefix_cache, device, dtype,
            chunk_rope_info=chunk_rope_info, model=model,
            sliding_layer_indices=sliding_layer_indices
        )

        # Compute average tensor length across all global (non-sliding) layers
        # For nonuniform caches, different layers can have different sequence lengths
        num_layers = len(combined_cache)
        num_global_layers = num_layers - len(sliding_layer_indices)
        total_tensor_len = 0
        for layer_idx in range(num_layers):
            if layer_idx not in sliding_layer_indices:
                total_tensor_len += combined_cache[layer_idx][0].shape[2]

        if num_global_layers > 0:
            avg_tensor_compacted_len = total_tensor_len / num_global_layers
        else:
            avg_tensor_compacted_len = 0

        print(f"Final average cache size: {avg_tensor_compacted_len:.1f} tokens across {num_global_layers} global layers")
        print(f"Total article tokens: {total_original_tokens} -> {total_compacted_tokens}")
        print(f"Aggregated timing: query_gen={total_query_generation_time:.2f}s, "
              f"compaction={total_inner_compaction_time:.2f}s, "
              f"train_stats={total_train_stats_time:.2f}s")

        # Compute overall stats
        # Effective article tokens accounts for -inf beta padding in non-uniform caches
        effective_article_tokens = total_effective_article_tokens
        num_kept = seq_len - article_len
        effective_compacted_seq_len = effective_article_tokens + num_kept

        # Tensor article tokens is the average tensor size of the article portion
        tensor_article_tokens = avg_tensor_compacted_len - num_kept

        all_stats = {
            'method': self.name(),
            'chunking_strategy': self.chunking_strategy.name,
            'inner_method': self._get_inner_method().name(),
            'num_chunks': len(chunks),
            'tensor_compacted_seq_len': avg_tensor_compacted_len,
            'effective_article_tokens': effective_article_tokens,
            'tensor_article_tokens': tensor_article_tokens,
            'effective_compacted_seq_len': effective_compacted_seq_len,
            'total_original_article_tokens': total_original_tokens,
            'total_compacted_article_tokens': total_compacted_tokens,
            'target_ratio': target_size,
            'chunk_stats': chunk_stats,
            # Empty per_layer_head_metrics for compatibility with evaluator
            # (chunked compaction doesn't provide per-layer-head metrics in the same format)
            'per_layer_head_metrics': {},
            # RoPE offset handling
            'rope_corrections': chunk_rope_info,
            # Aggregated timing from inner compactions (matches per_layer_head format)
            # Aggregate query_generation stats from chunks
            'query_generation': self._aggregate_query_generation_stats(chunk_stats, total_query_generation_time),
            'train_stats_time': total_train_stats_time,
            # Note: compaction_time here excludes query_gen and train_stats (just the OMP/algorithm time)
            'inner_compaction_time': total_inner_compaction_time,
            # Original chunk caches for test stats (only present when compute_stats=True)
            # Each entry is ((K, V), ...) for the article portion of that chunk
            '_original_chunk_caches': original_chunk_caches,
            # Compacted article-only chunks (needed for test stats computation)
            '_compacted_chunk_caches': compacted_chunks if compute_stats else None,
        }

        # Aggregate train stats from all chunks if available
        if compute_stats:
            self._aggregate_chunk_train_stats(all_stats, chunk_stats)

        return combined_cache, all_stats

    def _compact_kv_cache_kv_based(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        compute_stats: bool = False,
        verbose_logging: bool = False,
        vllm_model: Optional[Any] = None,
        article_text: Optional[str] = None,
        article_name: Optional[str] = None,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        KV-based chunked compaction.

        Extracts the full KV cache once, then slices it into chunks for compaction.
        This avoids RoPE corrections since chunk KV slices already have correct positions.

        For each chunk, we create two separate KV caches:
        1. chunk_kv_raw: Standard tuple format with just the chunk's KV states
           - Used for compaction (what we're compacting)
        2. chunk_kv_for_queries: CompactedPrefixCache with prefix + chunk + sliding layers
           - Used for query generation (provides correct context and RoPE positions)
           - Includes the chat template prefix so queries see the full context
           - Includes sliding window layers from the end of the original sequence
           - Uses original_seq_len to ensure new tokens get correct RoPE positions
        """
        # Extract article text if not provided
        if article_text is None:
            article_text = self._extract_article_text(formatted_context, tokenizer)

        if article_name is None:
            article_name = "the document"

        # Determine device and dtype from model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # past_key_values must be provided for KV-based chunking
        if past_key_values is None:
            raise ValueError(
                "KV-based chunked compaction requires pre-extracted KV cache. "
                "Set use_kv_based=False to use text-based chunking instead."
            )

        num_layers = len(past_key_values)
        sliding_layer_indices = sliding_layer_indices or set()

        # Find a non-sliding layer to get the full sequence length
        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape

        # Get prefix/article boundaries from indices
        if indices is not None:
            prefix_end = indices.start
            article_len = len(indices)
        else:
            prefix_end = 0
            article_len = seq_len

        # Extract prefix KV cache and compute suffix boundaries
        prefix_cache = self._extract_cache_slice(past_key_values, 0, prefix_end)
        article_end = prefix_end + article_len  # End of article in global sequence

        # Get sliding window size if available
        sliding_window = None
        if sliding_layer_indices:
            config = getattr(model, 'config', None)
            if config is not None:
                sliding_window = getattr(config, 'sliding_window', None)

        # Target size should be a ratio (0-1) for per-chunk compaction
        if target_size >= 1:
            # Convert absolute target to ratio based on article length
            if article_len > 0:
                non_article_len = seq_len - article_len
                target_size = (target_size - non_article_len) / article_len
            else:
                target_size = 1.0

        target_size = min(1.0, target_size)

        print(f"\n{'='*60}")
        print(f"KV-Based Chunked Compaction")
        print(f"{'='*60}")
        print(f"Chunking strategy: {self.chunking_strategy.name}")
        print(f"Inner method: {self._get_inner_method().name()}")
        print(f"Per-chunk compaction ratio: {target_size:.2%}")
        print(f"Prefix tokens: {prefix_end}")
        print(f"Article tokens: {article_len}")
        print(f"Full sequence length: {seq_len}")

        # Chunk the article text
        start_time = time.time()
        chunks = self.chunking_strategy.chunk(article_text, tokenizer)
        chunk_time = time.time() - start_time
        print(f"Chunked article into {len(chunks)} chunks in {chunk_time:.2f}s")

        # Process each chunk
        compacted_chunks = []
        chunk_stats = []
        total_original_tokens = 0
        total_compacted_tokens = 0
        total_effective_article_tokens = 0.0

        # Aggregate timing stats
        total_query_generation_time = 0.0
        total_inner_compaction_time = 0.0
        total_train_stats_time = 0.0

        # Store original chunk caches for test stats computation (only when compute_stats=True)
        original_chunk_caches = [] if compute_stats else None

        # Format chunk system prompt (None for gemma models)
        if self._is_gemma_model(model):
            chunk_system_prompt = None
        else:
            chunk_system_prompt = self.chunk_system_prompt_template.format(article_name=article_name)

        for chunk_idx, chunk in enumerate(chunks):
            print(f"\n--- Processing chunk {chunk_idx + 1}/{len(chunks)} ---")
            if chunk.metadata.get('note_id'):
                print(f"Note ID: {chunk.metadata['note_id']}")
            elif chunk.metadata.get('filename'):
                print(f"Filename: {chunk.metadata['filename']}")

            # Compute chunk boundaries in the full KV cache
            chunk_num_tokens = chunk.end_token_idx - chunk.start_token_idx
            chunk_start_global = prefix_end + chunk.start_token_idx
            chunk_end_global = chunk_start_global + chunk_num_tokens

            # Extract chunk KV slice from full cache (for compaction)
            chunk_kv_raw = self._extract_cache_slice(past_key_values, chunk_start_global, chunk_end_global)

            # Build KV cache for query generation with prefix + chunk + suffix + sliding layers
            # This ensures queries get correct RoPE positions via CompactedPrefixCache
            chunk_kv_for_queries_tuples = []
            for layer_idx in range(num_layers):
                if layer_idx in sliding_layer_indices:
                    # Use full original sliding layer (already at end of sequence)
                    keys = past_key_values[layer_idx][0]
                    values = past_key_values[layer_idx][1]
                else:
                    # Concatenate: [prefix before article] + [chunk] + [suffix after article]
                    # Prefix is the chat template before article starts (same for all chunks!)
                    prefix_keys = prefix_cache[layer_idx][0]
                    chunk_keys = chunk_kv_raw[layer_idx][0]
                    # Suffix is everything after the article ends (not after this chunk!)
                    suffix_keys = past_key_values[layer_idx][0][:, :, article_end:, :]
                    keys = torch.cat([prefix_keys, chunk_keys, suffix_keys], dim=2)

                    prefix_values = prefix_cache[layer_idx][1]
                    chunk_values = chunk_kv_raw[layer_idx][1]
                    suffix_values = past_key_values[layer_idx][1][:, :, article_end:, :]
                    values = torch.cat([prefix_values, chunk_values, suffix_values], dim=2)

                beta = torch.zeros(keys.shape[0], keys.shape[1], keys.shape[2],
                                  device=device, dtype=dtype)
                chunk_kv_for_queries_tuples.append((keys, beta, values))

            # Wrap in CompactedPrefixCache with full original_seq_len
            # This ensures new tokens (self-study queries) get correct RoPE positions
            from models.cache import CompactedPrefixCache
            chunk_kv_for_queries = CompactedPrefixCache(
                compacted_cache=tuple(chunk_kv_for_queries_tuples),
                original_seq_len=seq_len,  # Full original length for correct RoPE
                sliding_layer_indices=sliding_layer_indices,
                sliding_window=sliding_window,
            )

            chunk_article_len = chunk_num_tokens
            total_original_tokens += chunk_article_len
            print(f"Chunk tokens: {chunk_article_len}")

            # Compute chunk target size using the ratio
            chunk_target = max(1, int(chunk_article_len * target_size))
            print(f"Chunk target: {chunk_target} tokens ({target_size:.2%} of {chunk_article_len})")

            # Save original article cache for test stats if requested
            if compute_stats:
                original_article_cache = chunk_kv_raw  # Already extracted article portion
                original_chunk_caches.append(original_article_cache)

            # Get compaction method for this chunk
            chunk_compactor = self._get_inner_method(chunk_idx)

            # Time the inner compaction
            chunk_start_time = time.time()

            # Compact this chunk
            # Use chunk_kv_raw for compaction (standard format, just the chunk)
            # Use chunk_kv_for_queries for query generation (CompactedPrefixCache with prefix)
            # The article boundaries in chunk_kv_for_queries are [prefix_end, prefix_end + chunk_article_len)
            compact_kwargs = dict(
                past_key_values=chunk_kv_raw,
                target_size=chunk_target,
                indices=range(0, chunk_article_len),  # Entire chunk is "article"
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=self._format_chunk_context(model, tokenizer, chunk.text, chunk_system_prompt),
                compute_stats=compute_stats,
                verbose_logging=verbose_logging,
                vllm_model=vllm_model,
                sliding_layer_indices=sliding_layer_indices,
                past_key_values_for_queries=chunk_kv_for_queries,
            )
            # Only on-policy methods use query_cache_article_boundaries
            if isinstance(chunk_compactor, PerLayerHeadOnPolicyCompaction):
                compact_kwargs['query_cache_article_boundaries'] = (prefix_end, prefix_end + chunk_article_len)
            chunk_compacted, chunk_stat = chunk_compactor.compact_kv_cache(**compact_kwargs)

            chunk_total_time = time.time() - chunk_start_time

            # Extract timing from inner compaction stats
            chunk_query_gen_time = 0.0
            if 'query_generation' in chunk_stat and 'query_generation_time' in chunk_stat['query_generation']:
                chunk_query_gen_time = chunk_stat['query_generation']['query_generation_time']
            chunk_train_stats_time = chunk_stat.get('train_stats_time', 0.0)
            chunk_compaction_time = chunk_total_time - chunk_query_gen_time - chunk_train_stats_time

            total_query_generation_time += chunk_query_gen_time
            total_inner_compaction_time += chunk_compaction_time
            total_train_stats_time += chunk_train_stats_time

            # Extract compacted article portion (remove any prefix from compacted chunk)
            # For KV-based, we compacted with indices=range(0, chunk_article_len)
            # so the compacted cache is just the article portion
            compacted_article_len = 0
            num_global_layers = 0
            for layer_idx in range(num_layers):
                if layer_idx not in sliding_layer_indices:
                    compacted_article_len += chunk_compacted[layer_idx][0].shape[2]
                    num_global_layers += 1

            if num_global_layers > 0:
                compacted_article_len = compacted_article_len / num_global_layers

            total_compacted_tokens += compacted_article_len

            # Track effective article tokens from inner method
            chunk_effective_article_tokens = chunk_stat.get('effective_article_tokens', compacted_article_len)
            total_effective_article_tokens += chunk_effective_article_tokens

            compacted_chunks.append(chunk_compacted)
            chunk_stat['chunk_idx'] = chunk_idx
            chunk_stat['chunk_original_tokens'] = chunk_article_len
            chunk_stat['chunk_compacted_tokens'] = compacted_article_len
            chunk_stat['chunk_timing'] = {
                'query_generation_time': chunk_query_gen_time,
                'compaction_time': chunk_compaction_time,
                'train_stats_time': chunk_train_stats_time,
                'total_time': chunk_total_time,
            }
            chunk_stats.append(chunk_stat)

            # Free chunk cache to save memory
            del chunk_kv_for_queries
            del chunk_compacted

        # Concatenate all compacted chunks with prefix
        # No RoPE correction needed since chunks already have correct positions!
        print(f"\n--- Concatenating {len(compacted_chunks)} compacted chunks (no RoPE correction needed) ---")
        combined_cache = self._concatenate_chunks_kv_based(
            compacted_chunks, prefix_cache, device, dtype,
            sliding_layer_indices=sliding_layer_indices
        )

        # Compute average tensor length across all global layers
        num_layers = len(combined_cache)
        num_global_layers = num_layers - len(sliding_layer_indices)
        total_tensor_len = 0
        for layer_idx in range(num_layers):
            if layer_idx not in sliding_layer_indices:
                total_tensor_len += combined_cache[layer_idx][0].shape[2]

        if num_global_layers > 0:
            avg_tensor_compacted_len = total_tensor_len / num_global_layers
        else:
            avg_tensor_compacted_len = 0

        print(f"Final average cache size: {avg_tensor_compacted_len:.1f} tokens across {num_global_layers} global layers")
        print(f"Total article tokens: {total_original_tokens} -> {total_compacted_tokens}")
        print(f"Aggregated timing: query_gen={total_query_generation_time:.2f}s, "
              f"compaction={total_inner_compaction_time:.2f}s, "
              f"train_stats={total_train_stats_time:.2f}s")

        # Compute overall stats
        effective_article_tokens = total_effective_article_tokens
        num_kept = seq_len - article_len
        effective_compacted_seq_len = effective_article_tokens + num_kept
        tensor_article_tokens = avg_tensor_compacted_len - num_kept

        all_stats = {
            'method': self.name(),
            'chunking_strategy': self.chunking_strategy.name,
            'inner_method': self._get_inner_method().name(),
            'num_chunks': len(chunks),
            'tensor_compacted_seq_len': avg_tensor_compacted_len,
            'effective_article_tokens': effective_article_tokens,
            'tensor_article_tokens': tensor_article_tokens,
            'effective_compacted_seq_len': effective_compacted_seq_len,
            'total_original_article_tokens': total_original_tokens,
            'total_compacted_article_tokens': total_compacted_tokens,
            'target_ratio': target_size,
            'chunk_stats': chunk_stats,
            'per_layer_head_metrics': {},
            'use_kv_based': True,
            'query_generation': self._aggregate_query_generation_stats(chunk_stats, total_query_generation_time),
            'train_stats_time': total_train_stats_time,
            'inner_compaction_time': total_inner_compaction_time,
            '_original_chunk_caches': original_chunk_caches,
            '_compacted_chunk_caches': compacted_chunks if compute_stats else None,
        }

        # Aggregate train stats from all chunks if available
        if compute_stats:
            self._aggregate_chunk_train_stats(all_stats, chunk_stats)

        return combined_cache, all_stats

    def _concatenate_chunks_kv_based(
        self,
        compacted_chunks: List[Tuple],
        prefix_cache: Tuple,
        device: torch.device,
        dtype: torch.dtype,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple:
        """
        Concatenate compacted chunks with prefix for KV-based chunking.

        No RoPE correction needed since chunks already have correct positions.
        """
        if sliding_layer_indices is None:
            sliding_layer_indices = set()

        if not compacted_chunks:
            raise ValueError("No compacted chunks to concatenate")

        num_layers = len(compacted_chunks[0])
        combined = []

        for layer_idx in range(num_layers):
            # Prefix: convert (K, V) to (K, beta=0, V)
            K_prefix = prefix_cache[layer_idx][0]
            V_prefix = prefix_cache[layer_idx][1]
            beta_prefix = torch.zeros(
                K_prefix.shape[0], K_prefix.shape[1], K_prefix.shape[2],
                device=device, dtype=dtype
            )

            # Gather compacted chunks (already in (C1, beta, C2) format)
            C1_parts = [chunk[layer_idx][0] for chunk in compacted_chunks]
            beta_parts = [chunk[layer_idx][1] for chunk in compacted_chunks]
            C2_parts = [chunk[layer_idx][2] for chunk in compacted_chunks]

            # Concatenate: [prefix] + [chunks...]
            C1_combined = torch.cat([K_prefix] + C1_parts, dim=2)
            beta_combined = torch.cat([beta_prefix] + beta_parts, dim=2)
            C2_combined = torch.cat([V_prefix] + C2_parts, dim=2)

            combined.append((C1_combined, beta_combined, C2_combined))

        return tuple(combined)

    def _aggregate_query_generation_stats(self, chunk_stats: List[Dict], total_query_generation_time: float) -> Dict:
        """
        Aggregate query generation stats from all chunks.

        Returns a dict with the same structure as QueryGenerator's stats,
        with aggregated/averaged values across chunks.
        """
        chunks_with_qstats = [
            cs['query_generation'] for cs in chunk_stats
            if 'query_generation' in cs
        ]

        if not chunks_with_qstats:
            return {'query_generation_time': total_query_generation_time}

        # Sum up queries per KV head across chunks
        total_queries_per_kv_head = sum(
            qs.get('final_n_queries_per_kv_head', 0) for qs in chunks_with_qstats
        )

        # Aggregate methods_used - sum up queries from each method
        aggregated_methods = {}
        for qs in chunks_with_qstats:
            for method, method_info in qs.get('methods_used', {}).items():
                if method not in aggregated_methods:
                    aggregated_methods[method] = {
                        'n_queries_requested_per_kv_head': 0,
                        'n_queries_actual_per_kv_head': 0,
                        'fraction': 0.0,
                    }
                aggregated_methods[method]['n_queries_requested_per_kv_head'] += method_info.get('n_queries_requested_per_kv_head', 0)
                aggregated_methods[method]['n_queries_actual_per_kv_head'] += method_info.get('n_queries_actual_per_kv_head', 0)

        # Recompute fractions based on total
        if total_queries_per_kv_head > 0:
            for method in aggregated_methods:
                aggregated_methods[method]['fraction'] = (
                    aggregated_methods[method]['n_queries_actual_per_kv_head'] / total_queries_per_kv_head
                )

        return {
            'query_generation_time': total_query_generation_time,
            'final_n_queries_per_kv_head': total_queries_per_kv_head,
            'methods_used': aggregated_methods,
            'num_chunks': len(chunks_with_qstats),
        }

    def _aggregate_chunk_train_stats(self, all_stats: Dict, chunk_stats: List[Dict]) -> None:
        """
        Aggregate train stats across all chunks.

        Each chunk may have 'all_head_train_stats' from its inner compaction.
        We average these across chunks, weighted by number of compacted tokens.
        """
        chunks_with_train_stats = [
            (cs, cs.get('chunk_compacted_tokens', 1))
            for cs in chunk_stats
            if 'all_head_train_stats' in cs
        ]

        if not chunks_with_train_stats:
            return

        # Get metric keys from first chunk
        first_stats = chunks_with_train_stats[0][0]['all_head_train_stats']
        metric_keys = [k for k in first_stats.keys() if k != 'eval_queries_per_kv_head']

        aggregated = {}
        total_weight = sum(weight for _, weight in chunks_with_train_stats)

        for key in metric_keys:
            weighted_sum = 0.0
            for cs, weight in chunks_with_train_stats:
                val = cs['all_head_train_stats'].get(key)
                if val is not None:
                    weighted_sum += val * weight
            aggregated[key] = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Preserve eval_queries_per_kv_head from first chunk
        aggregated['eval_queries_per_kv_head'] = first_stats.get('eval_queries_per_kv_head')
        aggregated['num_chunks_with_stats'] = len(chunks_with_train_stats)

        all_stats['all_head_train_stats'] = aggregated

    def _extract_article_text(self, formatted_context: str, tokenizer) -> str:
        """Extract raw article text from formatted chat context."""
        from evaluation.utils import detect_user_tags

        user_start_tag, user_end_tag = detect_user_tags(formatted_context)

        # Find start of article (after user start tag and newline)
        user_start_pos = formatted_context.find(user_start_tag)
        if user_start_pos == -1:
            raise ValueError(f"Could not find '{user_start_tag}' in formatted context")

        article_start = formatted_context.find('\n', user_start_pos + len(user_start_tag))
        if article_start == -1:
            raise ValueError("Could not find newline after user start tag")
        article_start += 1

        # Find end of article (before user end tag)
        user_end_pos = formatted_context.find(user_end_tag, article_start)
        if user_end_pos == -1:
            # No end tag found, use rest of string
            article_end = len(formatted_context)
        else:
            article_end = user_end_pos

        return formatted_context[article_start:article_end].rstrip()

    def _is_gemma_model(self, model: Any) -> bool:
        """Check if the model is a gemma model (which doesn't use system prompts)."""
        model_name = getattr(model.config, 'name_or_path', '') or ''
        return "gemma" in model_name.lower()

    def _format_chunk_context(self, model, tokenizer, chunk_text: str, system_prompt: Optional[str]) -> str:
        """Format a chunk with its own chat template."""
        # Gemma models don't use system prompts
        if system_prompt is None or self._is_gemma_model(model):
            context_messages = [
                {"role": "user", "content": chunk_text}
            ]
        else:
            context_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text}
            ]
        context = tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        if context.startswith("<bos>"):
            context = context[len("<bos>"):]
        
        return context

    def _extract_prefix_cache(
        self,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        article_text: str,
        device: torch.device,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple, int, int, int, str]:
        """
        Extract KV cache for the prefix (non-article) portion.

        The suffix is NOT extracted here because the last chunk includes the suffix
        for correct RoPE positioning and sliding window attention.

        Parameters
        ----------
        model : Any
            Model instance
        tokenizer : Any
            Tokenizer
        formatted_context : str
            Full formatted context string
        article_text : str
            The article text within the context
        device : torch.device
            Device for tensors
        sliding_layer_indices : set, optional
            Set of layer indices that use sliding window attention

        Returns
        -------
        prefix_cache : tuple
            KV cache for prefix (system prompt, user tag) in (K, V) format.
            Sliding window layers have empty tensors (size 0 in seq dim).
        prefix_end : int
            Number of prefix tokens
        seq_len : int
            Total sequence length (what it would be if we processed full context)
        article_len : int
            Number of tokens in the article
        suffix_text : str
            The suffix text (for use with last chunk)
        """
        from evaluation.utils import detect_user_tags

        user_start_tag, user_end_tag = detect_user_tags(formatted_context)

        # Find article boundaries in the formatted context
        user_start_pos = formatted_context.find(user_start_tag)
        if user_start_pos == -1:
            raise ValueError(f"Could not find '{user_start_tag}' in formatted context")

        article_text_start = formatted_context.find('\n', user_start_pos + len(user_start_tag))
        if article_text_start == -1:
            raise ValueError("Could not find newline after user start tag")
        article_text_start += 1

        # Article ends where end tag begins (or end of string)
        user_end_pos = formatted_context.find(user_end_tag, article_text_start)
        if user_end_pos == -1:
            article_text_end = len(formatted_context)
            suffix_text = ""
        else:
            article_text_end = user_end_pos
            suffix_text = formatted_context[article_text_end:]

        # Extract prefix text
        prefix_text = formatted_context[:article_text_start]

        # Tokenize to get token counts
        prefix_tokens = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        prefix_end = prefix_tokens['input_ids'].shape[1]

        # Tokenize article to get article token count
        article_tokens = tokenizer(article_text, return_tensors="pt", add_special_tokens=False)
        article_len = article_tokens['input_ids'].shape[1]

        # Tokenize suffix to get suffix token count
        if suffix_text:
            suffix_tokens = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)
            suffix_len = suffix_tokens['input_ids'].shape[1]
        else:
            suffix_len = 0

        # Compute total seq_len
        seq_len = prefix_end + article_len + suffix_len

        # Run prefill for prefix (small, just system prompt + user tag)
        prefix_input_ids = prefix_tokens['input_ids'].to(device)
        with torch.no_grad():
            prefix_outputs = model(prefix_input_ids, use_cache=True)
        raw_prefix_cache = prefix_outputs.past_key_values

        return raw_prefix_cache, prefix_end, seq_len, article_len, suffix_text

    def _extract_chunk_kv_cache(
        self,
        model,
        tokenizer,
        chunk_text: str,
        system_prompt: Optional[str],
        device: str,
        outer_suffix_text: Optional[str] = None,
    ) -> Tuple[int, Tuple, range, int]:
        """
        Extract KV cache for a single chunk.

        Parameters
        ----------
        model : Any
            Model instance
        tokenizer : Any
            Tokenizer
        chunk_text : str
            The chunk's article text
        system_prompt : str, optional
            System prompt for this chunk. Ignored for gemma models.
        device : str
            Device for tensors
        outer_suffix_text : str, optional
            If provided, append this suffix text after the chunk content instead of
            using the normal chat template suffix. Used for the last chunk to include
            the original context's suffix tokens with correct attention context.

        Returns
        -------
        seq_len : int
            Total sequence length of the formatted chunk
        past_key_values : tuple
            KV cache from prefill
        article_indices : range
            Token indices of the article portion (excludes prefix and suffix)
        suffix_len : int
            Number of suffix tokens (0 for normal chunks, >0 when outer_suffix_text provided)
        """
        from evaluation.utils import detect_user_tags

        # Format chunk with its chat template (system prompt omitted for gemma)
        formatted_chunk = self._format_chunk_context(model, tokenizer, chunk_text, system_prompt)
        user_start_tag, user_end_tag = detect_user_tags(formatted_chunk)

        # Find article boundaries in chunk
        user_start_pos = formatted_chunk.find(user_start_tag)
        article_text_start = formatted_chunk.find('\n', user_start_pos + len(user_start_tag))
        if article_text_start == -1:
            raise ValueError("Could not find newline after user start tag in chunk")
        article_text_start += 1
        article_text_end = article_text_start + len(chunk_text)

        # If outer_suffix_text is provided, replace the chunk's suffix with it
        suffix_len = 0
        if outer_suffix_text is not None:
            # Find where the chunk's normal suffix starts (after article text)
            # and replace everything after article_text_end with outer_suffix_text
            formatted_chunk = formatted_chunk[:article_text_end] + outer_suffix_text
            suffix_len = len(tokenizer(outer_suffix_text, return_tensors="pt", add_special_tokens=False)['input_ids'][0])

        inputs = tokenizer(formatted_chunk, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs['input_ids'].to(device)
        seq_len = input_ids.shape[1]

        # Get token indices for article portion
        prefix_text = formatted_chunk[:article_text_start]
        prefix_tokens = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        article_start_idx = prefix_tokens['input_ids'].shape[1]

        article_with_prefix = formatted_chunk[:article_text_end]
        article_with_prefix_tokens = tokenizer(article_with_prefix, return_tensors="pt", add_special_tokens=False)
        article_end_idx = article_with_prefix_tokens['input_ids'].shape[1]

        article_indices = range(article_start_idx, article_end_idx)

        # Run prefill
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)

        return seq_len, outputs.past_key_values, article_indices, suffix_len

    def _extract_cache_slice(
        self,
        cache: Tuple,
        start: int,
        end: int,
        is_compacted: bool = False
    ) -> Tuple:
        """
        Extract a slice of the cache.

        Parameters
        ----------
        cache : tuple
            Either original cache ((K, V), ...) or compacted cache ((C1, beta, C2), ...)
        start : int
            Start index
        end : int
            End index
        is_compacted : bool
            Whether cache is in compacted format

        Returns
        -------
        sliced_cache : tuple
            Sliced cache in the same format
        """
        if start >= end:
            # Return empty cache
            if is_compacted:
                return tuple([
                    (
                        layer[0][:, :, 0:0, :],
                        layer[1][:, :, 0:0],
                        layer[2][:, :, 0:0, :]
                    )
                    for layer in cache
                ])
            else:
                return tuple([
                    (layer[0][:, :, 0:0, :], layer[1][:, :, 0:0, :])
                    for layer in cache
                ])

        if is_compacted:
            return tuple([
                (
                    layer[0][:, :, start:end, :],
                    layer[1][:, :, start:end],
                    layer[2][:, :, start:end, :]
                )
                for layer in cache
            ])
        else:
            return tuple([
                (layer[0][:, :, start:end, :], layer[1][:, :, start:end, :])
                for layer in cache
            ])

    def _concatenate_with_template(
        self,
        compacted_chunks: List[Tuple],
        prefix_cache: Tuple,
        device: torch.device,
        dtype: torch.dtype,
        chunk_rope_info: Optional[List[Dict]] = None,
        model: Optional[Any] = None,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple:
        """
        Concatenate compacted chunks with original template prefix.

        Structure: [prefix] + [chunk1] + [chunk2] + ... + [chunkN (includes suffix)]

        For prefix, use beta=0 (no attention bias). The suffix is included in the
        last chunk for correct RoPE positioning and sliding window attention.

        If chunk_rope_info and model are provided, applies RoPE correction to
        shift cached keys from their chunk-local positions to their target
        positions in the concatenated output.

        Parameters
        ----------
        compacted_chunks : List[Tuple]
            List of compacted chunks, each in (C1, beta, C2) format per layer
        prefix_cache : Tuple
            Original prefix cache in (K, V) format per layer
        device : torch.device
            Device for tensors
        dtype : torch.dtype
            Data type for tensors
        chunk_rope_info : List[Dict], optional
            RoPE correction info for each chunk. Each dict contains:
            - chunk_local_start: position where tokens were processed
            - original_article_start: position in original full context
            - length: number of tokens in this chunk
            - includes_suffix: whether chunk includes suffix
            - suffix_len: length of suffix
        model : Any, optional
            Model with rotary_emb for computing RoPE corrections
        sliding_layer_indices : set, optional
            Set of layer indices that use sliding window attention (skip RoPE correction for these)
        """
        if sliding_layer_indices is None:
            sliding_layer_indices = set()

        if not compacted_chunks:
            raise ValueError("No compacted chunks to concatenate")

        num_layers = len(compacted_chunks[0])
        apply_rope_correction = chunk_rope_info is not None and model is not None

        # Pre-compute RoPE corrections for all chunks if needed
        # Since all tokens in a chunk get the same offset shift, we only need to compute
        # cos/sin for the offset once (not per-position)
        rope_corrections = []
        if apply_rope_correction:
            for info in chunk_rope_info:
                # Each key in the compacted cache retains its original RoPE from chunk processing.
                # We apply a uniform shift of (original_article_start - chunk_local_start) to all keys,
                # which moves them from chunk-local positions to their original positions in the full article.
                chunk_len = info['length']
                if chunk_len > 0:
                    # Compute the offset - all tokens shift by the same amount
                    offset = info['original_article_start'] - info['chunk_local_start']
                    # Compute correction for this offset (using position 0 as reference)
                    cos_diff, sin_diff = compute_rope_correction(
                        model,
                        torch.tensor([0], device=device),
                        torch.tensor([offset], device=device),
                        device, dtype
                    )
                    rope_corrections.append((cos_diff, sin_diff))
                else:
                    rope_corrections.append(None)

        combined = []

        for layer_idx in range(num_layers):
            # Prefix: convert (K, V) to (K, beta=0, V)
            K_prefix = prefix_cache[layer_idx][0]
            V_prefix = prefix_cache[layer_idx][1]
            beta_prefix = torch.zeros(
                K_prefix.shape[0], K_prefix.shape[1], K_prefix.shape[2],
                device=device, dtype=dtype
            )

            # Gather and optionally correct compacted chunks (already in (C1, beta, C2) format)
            # Skip RoPE correction for sliding layers - their RoPE was already corrected with local
            # RoPE when the data was injected (sliding layers use a different rope_theta)
            C1_parts = []
            for chunk_idx, chunk in enumerate(compacted_chunks):
                C1 = chunk[layer_idx][0]  # Keys: (batch, num_heads, seq_len, head_dim)

                # Apply RoPE correction if available (skip for sliding layers - already corrected with local RoPE)
                if apply_rope_correction and rope_corrections[chunk_idx] is not None and layer_idx not in sliding_layer_indices:
                    cos_diff, sin_diff = rope_corrections[chunk_idx]
                    C1 = apply_rotary_pos_emb_to_cache(C1, cos_diff, sin_diff)

                C1_parts.append(C1)

            beta_parts = [chunk[layer_idx][1] for chunk in compacted_chunks]
            C2_parts = [chunk[layer_idx][2] for chunk in compacted_chunks]

            # Concatenate: [prefix] + [chunks...] (last chunk includes suffix)
            C1_combined = torch.cat([K_prefix] + C1_parts, dim=2)
            beta_combined = torch.cat([beta_prefix] + beta_parts, dim=2)
            C2_combined = torch.cat([V_prefix] + C2_parts, dim=2)

            combined.append((C1_combined, beta_combined, C2_combined))

        return tuple(combined)
