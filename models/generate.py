# models/generate.py
from typing import List, Tuple, Optional, Dict, Set, Any, TYPE_CHECKING

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import Cache

if TYPE_CHECKING:
    from vllm import LLM

Tensor = torch.Tensor

from .cache import CompactedPrefixCache


def get_generation_params(model) -> Dict[str, Optional[float]]:
    """
    Extract generation parameters from a model's generation_config.

    Returns the model's default temperature, top_k, and top_p if set,
    otherwise returns None for each parameter.

    Parameters
    ----------
    model : PreTrainedModel
        The model to extract generation parameters from

    Returns
    -------
    params : dict
        Dictionary with keys 'temperature', 'top_k', 'top_p', each either
        a value from the model's config or None if not meaningfully set.
    """
    gen_config = getattr(model, 'generation_config', None)

    temperature = None
    top_k = None
    top_p = None

    if gen_config is not None:
        if getattr(gen_config, 'do_sample', False):
            temp = getattr(gen_config, 'temperature', None)
            if temp is not None:
                temperature = temp

            tk = getattr(gen_config, 'top_k', None)
            if tk is not None and tk > 0:
                top_k = tk

            tp = getattr(gen_config, 'top_p', None)
            if tp is not None and tp < 1.0:
                top_p = tp

    return {
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
    }


@torch.inference_mode()
def chunked_prefill(
    model,
    input_ids: Tensor,
    past_key_values: Optional[Cache] = None,
    chunk_size: int = 4096,
    **model_kwargs,
) -> Any:
    """
    Process a prefill in chunks to avoid OOM on large attention masks.

    This function handles both CompactedPrefixCache and standard DynamicCache.
    The cache is updated in-place as each chunk is processed.

    Parameters
    ----------
    model : PreTrainedModel
        The language model
    input_ids : Tensor
        Input token IDs, shape (batch_size, seq_len)
    past_key_values : Cache, optional
        Existing KV cache (CompactedPrefixCache or DynamicCache)
    chunk_size : int
        Maximum number of tokens to process at once
    **model_kwargs
        Additional arguments passed to model.forward()

    Returns
    -------
    outputs
        The model output from the final chunk. The past_key_values cache
        is updated in-place with all processed tokens.
    """
    device = input_ids.device
    batch_size, input_len = input_ids.shape

    # Determine starting position from cache
    if past_key_values is not None:
        start_pos = past_key_values.get_seq_length()
    else:
        start_pos = 0

    outputs = None

    for chunk_start in range(0, input_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, input_len)
        chunk_input_ids = input_ids[:, chunk_start:chunk_end]

        # Compute cache_position for this chunk
        chunk_cache_position = torch.arange(
            start_pos + chunk_start,
            start_pos + chunk_end,
            device=device,
            dtype=torch.long,
        )

        # Build attention mask for current total KV length
        # The mask covers: [past_key_values] + [tokens processed so far] + [current chunk]
        current_kv_len = start_pos + chunk_end
        chunk_attention_mask = torch.ones(
            (batch_size, current_kv_len),
            device=device,
            dtype=torch.long,
        )

        outputs = model(
            input_ids=chunk_input_ids,
            past_key_values=past_key_values,
            cache_position=chunk_cache_position,
            attention_mask=chunk_attention_mask,
            use_cache=True,
            **model_kwargs,
        )

    return outputs


def get_sliding_layer_info(model) -> Tuple[Set[int], Optional[int]]:
    """
    Extract sliding layer indices and window size from a model's config.

    Parameters
    ----------
    model : PreTrainedModel
        The model to extract info from

    Returns
    -------
    sliding_layer_indices : set
        Set of layer indices that use sliding window attention
    sliding_window : int or None
        The sliding window size, or None if not applicable
    """
    sliding_layer_indices = set()
    sliding_window = None

    config = getattr(model, 'config', None)
    if config is not None:
        layer_types = getattr(config, 'layer_types', None)
        sliding_window = getattr(config, 'sliding_window', None)
        if layer_types is not None:
            for layer_idx, layer_type in enumerate(layer_types):
                if layer_type == "sliding_attention":
                    sliding_layer_indices.add(layer_idx)

    return sliding_layer_indices, sliding_window

@torch.inference_mode()
def generate_with_full_context(
    model,
    tokenizer: AutoTokenizer,
    full_prompt: str,
    max_new_tokens: int = 128,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> str:
    """
    Generate text using the full context (no cache reuse).

    This is the baseline generation that processes the full context + prompt
    from scratch each time.

    Parameters
    ----------
    model : Qwen3ForCausalLM | Llama3ForCausalLm
        The language model
    tokenizer : AutoTokenizer
        Tokenizer for the model
    full_prompt : str
        Pre-formatted full prompt (should include chat template formatting)
    max_new_tokens : int
        Maximum number of tokens to generate
    temperature : float, optional
        Sampling temperature. If None, uses model's default or 1.0.
    top_k : int, optional
        Top-k sampling parameter. If None, uses model's default or disabled.
    top_p : float, optional
        Top-p (nucleus) sampling parameter. If None, uses model's default or 1.0.

    Returns
    -------
    answer_text : str
        Generated answer text (only the generated tokens)
    """
    device = next(model.parameters()).device

    # Get model's default generation params if not specified
    gen_params = get_generation_params(model)
    if temperature is None:
        temperature = gen_params['temperature'] if gen_params['temperature'] is not None else 1.0
    if top_k is None:
        top_k = gen_params['top_k']  # None means disabled
    if top_p is None:
        top_p = gen_params['top_p'] if gen_params['top_p'] is not None else 1.0

    # Tokenize (add_special_tokens=False since full_prompt already has <bos> from chat template)
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    gen_only = outputs[:, inputs.input_ids.size(1):]  # slice by token length
    answer_text = tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()

    return answer_text


@torch.inference_mode()
def generate_with_full_context_batch(
    model,
    tokenizer: AutoTokenizer,
    full_prompts: List[str],
    max_new_tokens: int = 128,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> List[str]:
    """
    Generate text using the full context (no cache reuse) for multiple prompts in batch.

    Parameters
    ----------
    model : Qwen3ForCausalLM | Llama3ForCausalLm
        The language model
    tokenizer : AutoTokenizer
        Tokenizer for the model
    full_prompts : List[str]
        List of pre-formatted full prompts (should include chat template formatting)
    max_new_tokens : int
        Maximum number of tokens to generate
    temperature : float, optional
        Sampling temperature. If None, uses model's default or 1.0.
    top_k : int, optional
        Top-k sampling parameter. If None, uses model's default or disabled.
    top_p : float, optional
        Top-p (nucleus) sampling parameter. If None, uses model's default or 1.0.

    Returns
    -------
    answer_texts : List[str]
        List of generated answer texts (only the generated tokens)
    """
    device = next(model.parameters()).device

    # Get model's default generation params if not specified
    gen_params = get_generation_params(model)
    if temperature is None:
        temperature = gen_params['temperature'] if gen_params['temperature'] is not None else 1.0
    if top_k is None:
        top_k = gen_params['top_k']  # None means disabled
    if top_p is None:
        top_p = gen_params['top_p'] if gen_params['top_p'] is not None else 1.0

    # Save original padding side and set to left for decoder-only model
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    # Set pad token if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize with padding (add_special_tokens=False since prompts already have <bos>)
    inputs = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    ).to(device)

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    # Decode each sequence, extracting only the generated tokens
    answer_texts = []
    input_lengths = inputs.input_ids.shape[1]  # Length includes padding
    for i, output_ids in enumerate(outputs):
        # With left-padding, generated tokens start at position input_lengths
        # output_ids has shape [total_length] where total_length = input_lengths + generated_tokens
        gen_only = output_ids[input_lengths:]
        answer_text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        answer_texts.append(answer_text)

    return answer_texts


@torch.inference_mode()
def generate_with_compacted_cache(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    compacted_cache: Tuple[Tuple[Tensor, Tensor, Tensor], ...],
    max_new_tokens: int = 128,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    original_seq_len: Optional[int] = None,
    stop_strings: Optional[List[str]] = None,
    return_cache: bool = False,
):
    """
    Generator that:
      1) installs a CompactedPrefixCache(pref=C1,β,C2),
      2) batches prompt tokens for fast prefill,
      3) then generates new tokens with additive β biases over the compacted prefix.

    For models with sliding window layers, the sliding layer info is automatically
    extracted from the model config. The compacted_cache should have keys/values
    in C1/C2 for these layers (with beta=0).

    Parameters
    ----------
    stop_strings : list of str, optional
        List of strings that should stop generation when encountered.
    return_cache : bool, optional
        If True, also return the updated cache after generation.

    Returns
    -------
    answer_text : str
        Generated answer text (only the generated tokens, excluding prompt)
    num_generated_tokens : int
        Number of tokens generated
    stopped_early : bool
        True if generation stopped due to stop_strings or EOS (before max_new_tokens)
    updated_cache : tuple, optional
        If return_cache=True, the updated cache in (C1, beta, C2) format per layer.
        The cache includes the original compacted prefix plus all newly generated tokens.
        Note: beta for newly generated tokens is 0.
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Get model's default generation params if not specified
    gen_params = get_generation_params(model)
    if temperature is None:
        temperature = gen_params['temperature'] if gen_params['temperature'] is not None else 1.0
    if top_k is None:
        top_k = gen_params['top_k']  # None means disabled
    if top_p is None:
        top_p = gen_params['top_p'] if gen_params['top_p'] is not None else 1.0

    # Extract sliding layer info from model
    sliding_layer_indices, sliding_window = get_sliding_layer_info(model)

    # Move compacted cache tensors onto model device/dtype
    moved_layers = []
    for (C1, beta, C2) in compacted_cache:
        moved_layers.append((
            C1.to(device=device, dtype=model_dtype),
            beta.to(device=device, dtype=model_dtype),
            C2.to(device=device, dtype=model_dtype),
        ))

    cache = CompactedPrefixCache(
        tuple(moved_layers),
        original_seq_len=original_seq_len,
        sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
        sliding_window=sliding_window,
    )

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    past_seen_tokens = cache.get_seq_length()  # t
    attn_mask = enc.get("attention_mask", None).to(device)
    if attn_mask is None:
        # tokenizer didn't produce a mask; assume everything is real tokens
        attn_mask = torch.ones_like(input_ids, dtype=torch.long)
    batch_size = attn_mask.size(0)
    if past_seen_tokens > 0:
        prefix_mask = torch.ones(
            (batch_size, past_seen_tokens),
            device=attn_mask.device,
            dtype=attn_mask.dtype,
        )
        attention_mask = torch.cat([prefix_mask, attn_mask], dim=1)
    else:
        attention_mask = attn_mask
    input_len = input_ids.shape[1]
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + input_len,
        device=device,
        dtype=torch.long,
    )

    # Build generation kwargs
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": cache,
        "cache_position": cache_position,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }

    # Add stop strings if provided
    if stop_strings:
        generate_kwargs["stop_strings"] = stop_strings
        generate_kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    gen_only = outputs[:, input_ids.size(1):]
    num_generated_tokens = gen_only.shape[1]
    answer_text = tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()
    stopped_early = num_generated_tokens < max_new_tokens

    if return_cache:
        # Extract updated cache from CompactedPrefixCache
        # Each layer has keys, values (updated with new tokens) and beta (original prefix only)
        updated_cache = []
        for layer in cache.layers:
            # keys/values include original prefix + newly generated tokens
            # beta only covers the original prefix length, so we need to extend it with zeros
            keys = layer.keys
            values = layer.values
            if hasattr(layer, 'beta'):
                # CompactedPrefixLayer - extend beta with zeros for new tokens
                original_beta = layer.beta
                base_len = layer.base_len
                current_len = keys.shape[2]
                if current_len > base_len:
                    # Extend beta with zeros for newly generated tokens
                    new_tokens = current_len - base_len
                    zeros = torch.zeros(
                        original_beta.shape[0], original_beta.shape[1], new_tokens,
                        dtype=original_beta.dtype, device=original_beta.device
                    )
                    beta = torch.cat([original_beta, zeros], dim=2)
                else:
                    beta = original_beta
            else:
                # DynamicSlidingWindowLayer - beta is all zeros
                beta = torch.zeros(
                    keys.shape[0], keys.shape[1], keys.shape[2],
                    dtype=keys.dtype, device=keys.device
                )
            updated_cache.append((keys, beta, values))
        return answer_text, num_generated_tokens, stopped_early, tuple(updated_cache)

    return answer_text, num_generated_tokens, stopped_early


@torch.inference_mode()
def generate_with_compacted_cache_batch(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    compacted_cache: Tuple[Tuple[Tensor, Tensor, Tensor], ...],
    max_new_tokens: int = 128,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    original_seq_len: Optional[int] = None,
) -> List[str]:
    """
    Batched generator using compacted cache + flex attention.

    This handles:
    1. Expanding the shared compacted prefix tensors to match the prompt batch size.
    2. Left-padding the prompts.
    3. Constructing the correct attention mask (Shared Prefix [1s] + Prompt Padding [0s] + Prompt [1s]).
    4. Generating using the standard model.generate loop with the custom cache.

    For models with sliding window layers, the sliding layer info is automatically
    extracted from the model config. The compacted_cache should have keys/values
    in C1/C2 for these layers (with beta=0).

    Parameters
    ----------
    original_seq_len : int, optional
        Original sequence length before compaction (used for RoPE offset).
        Each layer computes its own rope_base as: original_seq_len - compacted_prefix_len[layer]
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Get model's default generation params if not specified
    gen_params = get_generation_params(model)
    if temperature is None:
        temperature = gen_params['temperature'] if gen_params['temperature'] is not None else 1.0
    if top_k is None:
        top_k = gen_params['top_k']  # None means disabled
    if top_p is None:
        top_p = gen_params['top_p'] if gen_params['top_p'] is not None else 1.0

    # Extract sliding layer info from model
    sliding_layer_indices, sliding_window = get_sliding_layer_info(model)

    # Tokenize with left padding
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    input_attn_mask = enc["attention_mask"]
    
    batch_size = input_ids.shape[0]
    tokenizer.padding_side = original_padding_side

    pad_counts = (input_attn_mask == 0).sum(dim=1)  # how many pads

    # Prepare and Expand Cache for Batch Size
    # The cache tensors are (1, KV, t, D) or (KV, t, D). 
    # We need to ensure they are (B, KV, t, D) to match the input batch size.
    expanded_layers = []
    def expand_kv_tensor(t: Tensor) -> Tensor:
        """Ensure the KV tensor has a batch dim that matches batch_size."""
        t = t.to(device=device, dtype=model_dtype)
        if t.dim() == 3:
            # Missing batch dimension: (KV, t, D)
            t = t.unsqueeze(0)
        if t.shape[0] == batch_size:
            return t
        if t.shape[0] == 1 and batch_size > 1:
            return t.expand(batch_size, *t.shape[1:]).contiguous()
        return t

    def expand_beta_tensor(beta: Tensor) -> Tensor:
        """Ensure beta tensors broadcast correctly across the prompt batch."""
        beta = beta.to(device=device, dtype=model_dtype)
        if beta.dim() == 2:
            # Missing batch dimension: (KV, t)
            beta = beta.unsqueeze(0)
        if beta.shape[0] == batch_size:
            return beta
        if beta.shape[0] == 1 and batch_size > 1:
            return beta.expand(batch_size, *beta.shape[1:]).contiguous()
        return beta

    for (C1, beta, C2) in compacted_cache:
        expanded_layers.append((
            expand_kv_tensor(C1),
            expand_beta_tensor(beta),
            expand_kv_tensor(C2),
        ))

    # Initialize the cache with the expanded tensors
    cache = CompactedPrefixCache(
        tuple(expanded_layers),
        original_seq_len=original_seq_len,
        pad_counts=pad_counts,
        sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
        sliding_window=sliding_window,
    )
    past_seen_tokens = cache.get_seq_length() # Length of the compacted prefix (t)

    # Construct Attention Mask
    # Mask shape must be (B, prefix_len + prompt_len)
    # Prefix part is always 1 (attended to). Prompt part follows tokenizer mask (handles PAD).
    if past_seen_tokens > 0:
        prefix_mask = torch.ones(
            (batch_size, past_seen_tokens),
            device=device,
            dtype=input_attn_mask.dtype,
        )
        attention_mask = torch.cat([prefix_mask, input_attn_mask], dim=1)
    else:
        attention_mask = input_attn_mask

    # Define Cache Position & Per-example Position IDs
    # The input_ids implicitly continue *after* the compacted prefix.
    # cache_position indices: [t, t+1, t+2, ... t+L]
    input_len = input_ids.shape[1]
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + input_len,
        device=device,
        dtype=torch.long,
    )

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode
    # outputs shape: (B, input_len + generated_len)
    # We slice off the input tokens.
    answer_texts = []
    for i, output_ids in enumerate(outputs):
        gen_only = output_ids[input_len:]
        answer_text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        answer_texts.append(answer_text)

    return answer_texts


def generate_with_vllm_batch(
    vllm_model: "LLM",
    full_prompts: List[str],
    max_new_tokens: int = 128,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> List[str]:
    """
    Generate text using vLLM for efficient batch inference.

    This is the recommended method for generating with full context as vLLM
    handles memory more efficiently than HuggingFace for long sequences.
    vLLM automatically batches and schedules requests efficiently.

    Parameters
    ----------
    vllm_model : LLM
        Initialized vLLM model
    full_prompts : List[str]
        List of pre-formatted full prompts (should include chat template formatting)
    max_new_tokens : int
        Maximum number of tokens to generate
    temperature : float, optional
        Sampling temperature. If None, uses default (1.0).
    top_k : int, optional
        Top-k sampling parameter. If None, uses default (-1, disabled).
    top_p : float, optional
        Top-p (nucleus) sampling parameter. If None, uses default (1.0).

    Returns
    -------
    answer_texts : List[str]
        List of generated answer texts (only the generated tokens)
    """
    from vllm import SamplingParams

    # Apply defaults if not specified
    if temperature is None:
        temperature = 1.0
    if top_k is None:
        top_k = -1
    if top_p is None:
        top_p = 1.0

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    outputs = vllm_model.generate(full_prompts, sampling_params)

    # Extract generated text from each output
    answer_texts = [output.outputs[0].text.strip() for output in outputs]

    return answer_texts
