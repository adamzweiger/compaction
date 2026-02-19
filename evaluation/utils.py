# evaluation/utils.py
"""
Shared utilities for evaluation tasks.

This module contains common functionality used across different evaluation scripts:
- Model loading
- KV cache extraction
- Chat template formatting
- vLLM initialization and management
- Answer parsing utilities
"""
import re
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.generate import get_generation_params

QWEN_USER_START = "<|im_start|>user"
QWEN_USER_END = "<|im_end|>"
LLAMA_USER_START = "<|start_header_id|>user"
LLAMA_USER_END = "<|eot_id|>"
GEMMA_USER_START = "<start_of_turn>user"
GEMMA_USER_END = "<end_of_turn>"


def detect_user_tags(formatted_context: str) -> Tuple[str, str]:
    """
    Detect the user start/end tags used in the formatted chat context.

    Parameters
    ----------
    formatted_context : str
        Chat-formatted string produced by tokenizer.apply_chat_template.

    Returns
    -------
    tuple
        (user_start_tag, user_end_tag)
    """
    if LLAMA_USER_START in formatted_context:
        return LLAMA_USER_START, LLAMA_USER_END
    if QWEN_USER_START in formatted_context:
        return QWEN_USER_START, QWEN_USER_END
    if GEMMA_USER_START in formatted_context:
        return GEMMA_USER_START, GEMMA_USER_END
    raise ValueError("Could not detect user tags from formatted context.")


def compute_article_indices(
    tokenizer,
    formatted_context: str,
    article_text: str,
) -> range:
    """
    Compute the token indices of the article portion within a formatted context.

    This function finds where the article text appears within the chat-formatted
    context and returns the corresponding token index range.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer used to encode the context
    formatted_context : str
        Chat-formatted string produced by tokenizer.apply_chat_template
    article_text : str
        The original article text (before chat formatting)

    Returns
    -------
    range
        Token indices corresponding to the article portion (start_idx, end_idx)
    """
    # Detect user tags to find article boundaries
    user_start_tag, _ = detect_user_tags(formatted_context)

    # Find position after the user start tag
    user_start_pos = formatted_context.find(user_start_tag)
    if user_start_pos == -1:
        raise ValueError(f"Could not find '{user_start_tag}' tag in formatted context")

    # The article starts after the user tag and a newline
    article_text_start = formatted_context.find('\n', user_start_pos + len(user_start_tag))
    if article_text_start == -1:
        raise ValueError("Could not find newline after user start tag")
    article_text_start += 1  # Skip the newline itself

    # The article ends at article_text_start + len(article_text)
    article_text_end = article_text_start + len(article_text)

    # Tokenize prefix (everything before article) to find article start token
    prefix_text = formatted_context[:article_text_start]
    prefix_tokens = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    article_start_idx = prefix_tokens['input_ids'].shape[1]

    # Tokenize everything up to and including the article to find article end token
    article_with_prefix = formatted_context[:article_text_end]
    article_with_prefix_tokens = tokenizer(article_with_prefix, return_tensors="pt", add_special_tokens=False)
    article_end_idx = article_with_prefix_tokens['input_ids'].shape[1]

    return range(article_start_idx, article_end_idx)


def load_model_and_tokenizer(
    model_name: str,
    device: str = None,
    dtype: Optional[torch.dtype] = None,
    max_model_len: Optional[int] = None,
):
    """
    Load model and tokenizer with proper configuration.

    Parameters
    ----------
    model_name : str
        HuggingFace model name
    device : str, optional
        Device to use ('cpu' or 'cuda')
    dtype : torch.dtype, optional
        Data type for computations
    max_model_len : int, optional
        Maximum model context length. If specified, enables YaRN rope scaling
        to handle extended context lengths beyond the model's default.

    Returns
    -------
    model : PreTrainedModel
        Loaded model
    tokenizer : PreTrainedTokenizer
        Loaded tokenizer
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {
        "device_map": device if device == "cuda" else None,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa"
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    # Enable YaRN rope scaling for extended context if max_model_len is specified
    if model_name == "Qwen/Qwen3-4B" and max_model_len is not None:
        model_kwargs["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": max_model_len/32768,
            "original_max_position_embeddings": 32768,
        }
        print(f"Enabling YaRN rope scaling for long context support")

    # Use Qwen3ForCausalLM for qwen3 models, otherwise use AutoModelForCausalLM
    if "qwen" in model_name.lower():
        from models.qwen3 import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    elif "llama" in model_name.lower():
        from models.llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    elif "gemma" in model_name.lower():
        from models.gemma3 import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    else:
        print("Warning: Using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

    if device == "cpu":
        model = model.to(device)

    model.eval()
    print(f"Model loaded successfully")
    return model, tokenizer


def offload_model_to_cpu(model) -> None:
    """
    Offload a HuggingFace model from GPU to CPU to free GPU memory.

    Parameters
    ----------
    model : PreTrainedModel
        The model to offload
    """
    if model is not None:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def reload_model_to_gpu(model, device: str = "cuda") -> None:
    """
    Reload a HuggingFace model from CPU back to GPU.

    Parameters
    ----------
    model : PreTrainedModel
        The model to reload
    device : str
        Target device (default: 'cuda')
    """
    if model is not None:
        model.to(device)
        model.eval()


def initialize_vllm(
    model_name: str,
    gpu_memory_utilization: float = 0.5,
    max_model_len: Optional[int] = None,
):
    """
    Initialize vLLM model for query generation.

    Parameters
    ----------
    model_name : str
        HuggingFace model name to use with vLLM
    gpu_memory_utilization : float
        Fraction of GPU memory to use (default: 0.5)
    max_model_len : int, optional
        Maximum model context length. If None, uses model default.
        For long contexts, set to 131072 or higher.
        When set, also enables YaRN rope scaling for extended context.

    Returns
    -------
    vllm_model : LLM
        Initialized vLLM model in sleep mode
    """
    print("Initializing vLLM model (will be reused across all articles)...")

    # Set vLLM multiprocessing method to 'spawn' to avoid CUDA fork issues
    import os
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    try:
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "vLLM is required for this run."
        )

    # Build kwargs for LLM initialization
    llm_kwargs = {
        "model": model_name,
        "tokenizer": model_name,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_sleep_mode": True,
    }

    # Add max_model_len and rope_scaling if specified
    if model_name == "Qwen/Qwen3-4B" and max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
        # Enable YaRN rope scaling for extended context
        # This allows the model to handle longer sequences than its default
        llm_kwargs["hf_overrides"] = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": max_model_len/32768,
                "original_max_position_embeddings": 32768,
            }
        }
        print(f"Using extended context: max_model_len={max_model_len} with YaRN rope scaling")

    vllm_model = LLM(**llm_kwargs)
    # Put it to sleep immediately to free GPU memory
    vllm_model.sleep()
    print(f"vLLM initialized and put to sleep (model: {model_name})")

    return vllm_model


def extract_full_kv_cache(
    model,
    tokenizer,
    article_text: str,
    device: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[int, Tuple, range, str, int]:
    """
    Extract full KV cache from article text formatted with QA chat template.

    Parameters
    ----------
    model : PreTrainedModel
        The model to use for cache extraction
    tokenizer : PreTrainedTokenizer
        The tokenizer
    article_text : str
        The article text
    device : str
        Device to use
    system_prompt : str, optional
        System prompt to use for formatting. If None, uses default based on model_name.
    model_name : str, optional
        Model name to determine default system prompt (gemma uses empty prompt).

    Returns
    -------
    seq_len : int
        Sequence length of the formatted context
    past_key_values : tuple
        Full KV cache from the model
    article_indices : range
        Indices of the article portion (between <user> and </user> tags)
    formatted_context : str
        The formatted context string (article with chat template applied)
    original_token_length : int
        Original token length of the sequence
    """
    # Format article as context using QA-specific system prompt
    formatted_context = format_context(tokenizer, article_text, system_prompt, model_name)

    # Tokenize the full context
    # Use add_special_tokens=False since formatted_context already has <bos> from chat template
    inputs = tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
    original_token_length = inputs['input_ids'].shape[1]

    input_ids = inputs['input_ids'].to(device)
    seq_len = input_ids.shape[1]

    # Compute article token indices using the shared utility function
    article_indices = compute_article_indices(tokenizer, formatted_context, article_text)

    with torch.no_grad():
        outputs = model(
            input_ids,
            use_cache=True
        )
        # TODO: fix the below for models with sliding layers 
        # from models.generate import chunked_prefill
        # outputs = chunked_prefill(
        #     model,
        #     input_ids=input_ids,
        # )
    return seq_len, outputs.past_key_values, article_indices, formatted_context, original_token_length


def get_default_system_prompt(model_name: Optional[str] = None) -> str:
    """
    Get the default system prompt for a model.

    Gemma models don't have explicit system prompts.
    """
    if model_name and "gemma" in model_name.lower():
        return ""
    return "You are a helpful assistant. Answer a question based on the provided context."


def format_context(
    tokenizer,
    article: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Format article as context using QA-specific system prompt.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer
    article : str
        The article text
    system_prompt : str, optional
        System prompt to use. If None, uses default based on model_name.
    model_name : str, optional
        Model name to determine default system prompt (gemma uses none).

    Returns
    -------
    str
        Formatted context
    """
    if system_prompt is None:
        system_prompt = get_default_system_prompt(model_name)

    # For gemma models, omit the system message entirely
    if system_prompt == "":
        context_messages = [
            {"role": "user", "content": article}
        ]
    else:
        context_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": article}
        ]
    return tokenizer.apply_chat_template(
        context_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )


def format_question(
    tokenizer,
    question: str,
    options: Optional[List[str]] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Format question as prompt using chat template.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer
    question : str
        The question text
    options : list of str, optional
        Multiple choice options. If provided, formats with letters (A, B, C, D, E, etc.)
    model_name : str, optional
        Model name to determine prompt format (used to detect Llama models)

    Returns
    -------
    str
        Formatted prompt ready for generation
    """
    # Detect model type for special handling
    is_qwen = model_name and "qwen" in model_name.lower()
    is_gemma = model_name and "gemma" in model_name.lower()

    if options:
        # Format as multiple choice with A, B, C, D, E, etc.
        # Support up to 26 options (A-Z)
        num_options = len(options)
        option_labels = [chr(65 + i) for i in range(num_options)]  # 65 is 'A'
        formatted_options = '\n'.join([
            f"{label}. {option}"
            for label, option in zip(option_labels, options)
        ])

        # Generate instruction based on number of options
        if num_options <= 4:
            option_list = "A, B, C, or D"
        elif num_options == 5:
            option_list = "A, B, C, D, or E"
        else:
            option_list = ", ".join(option_labels[:-1]) + f", or {option_labels[-1]}"

        if is_qwen:
            if model_name == "Qwen/Qwen3-4B-Instruct-2507": # need to specify step by step
                instruction = f"Please think step by step very briefly and then, on a new line, respond with the letter ({option_list}) of the correct option. If you are not sure, still make a guess."
            else:
                instruction = f"Please think very briefly and then respond with **only** the letter ({option_list}) of the correct option. If you are not sure, still make a guess."
        elif is_gemma: # need to remove "briefly" so it actually thinks...
            instruction = f"Please think step by step and then, on a new line, respond with **only** the letter ({option_list}) of the correct option. If you are not sure, still make a guess."
        else:
            instruction = f"Please think step by step briefly and then, on a new line, respond with **only** the letter ({option_list}) of the correct option. If you are not sure, still make a guess."

        content = f"{question}\n\n{formatted_options}\n\n{instruction}"
    else:
        content = question

    question_messages = [
        {"role": "user", "content": content}
    ]

    result = tokenizer.apply_chat_template(
        question_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    # Strip the first <bos> token for Gemma models
    if is_gemma:
        result = result[len("<bos>"):]

    return result


def parse_model_choice(answer_text: str, max_options: int = 5) -> Optional[int]:
    """
    Parse the model's answer to extract the chosen option index.

    Handles thinking mode where answer may be in format:
    <think>reasoning...</think>
    A

    Also handles formats like:
    - "the answer is: D."
    - "the answer is D"

    Parameters
    ----------
    answer_text : str
        The generated answer text
    max_options : int
        Maximum number of options (default: 5 for A-E)

    Returns
    -------
    int or None
        The option index (1=A, 2=B, 3=C, 4=D, 5=E, ...) or None if cannot parse
    """
    has_open_think = '<think>' in answer_text
    has_close_think = '</think>' in answer_text

    # Incomplete thinking tag => can't parse yet
    if has_open_think and not has_close_think:
        return None

    if has_close_think:
        parts = answer_text.split('</think>')
        if len(parts) > 1:
            answer_text = parts[-1].strip()
    elif not has_open_think:
        # No <think> at all means take the last non-empty line
        lines = answer_text.splitlines()
        for line in reversed(lines):
            if line.strip():
                answer_text = line.strip()
                break

    # Try to extract from various answer formats:
    # - "the answer is: X" or "the answer is X"
    # - "Answer: X" or "Answer X"
    answer_lower = answer_text.lower()
    for pattern in ['answer:', 'answer is', 'correct option is']:
        if pattern in answer_lower:
            # Extract everything after the pattern
            parts = answer_lower.split(pattern)
            if len(parts) > 1:
                # Remove colons, periods, *, and whitespace
                candidate = parts[-1].strip().lstrip(':').strip().rstrip('.').strip()
                candidate = candidate.replace('*', '')
                if candidate and candidate[0] in tuple(chr(97 + i) for i in range(max_options)):  # a-e (or more)
                    return ord(candidate[0]) - ord('a') + 1

    answer_clean = answer_lower.split(".")[0].strip()
    answer_clean = answer_clean.replace('*', '')

    # Check that it matches valid options (a-e by default, or more)
    # Return 1-indexed to match dataset gold_label format
    valid_options = tuple(chr(97 + i) for i in range(max_options))  # a, b, c, d, e, ...
    if answer_clean in valid_options:
        return ord(answer_clean) - ord('a') + 1

    # Fallback: look for standalone capital letters (A, B, C, D, E) in the last line
    valid_upper = tuple(chr(65 + i) for i in range(max_options))  # A, B, C, D, E, ...
    # Search for standalone letter (word boundary or surrounded by non-letters)
    pattern = r'(?<![a-zA-Z])([' + ''.join(valid_upper) + r'])(?![a-zA-Z])'
    matches = re.findall(pattern, answer_text)
    if matches:
        # Take the last match (most likely to be the final answer)
        letter = matches[-1]
        return ord(letter) - ord('A') + 1

    return None


def compute_cache_memory_stats(
    original_cache: Tuple,
    compacted_cache: Tuple,
) -> Dict:
    """
    Compute memory statistics for original vs compacted cache.

    Parameters
    ----------
    original_cache : tuple
        Original KV cache
    compacted_cache : tuple
        Compacted cache with (C1, beta, C2) per layer. For sliding layers,
        C1/C2 contain keys/values and beta should be zeros.

    Returns
    -------
    stats : dict
        Memory statistics
    """
    original_cache_size = sum([k.numel() + v.numel() for k, v in original_cache])
    compacted_cache_size = sum([(c1.numel() + beta.numel() + c2.numel()) for c1, beta, c2 in compacted_cache])

    memory_reduction_pct = 100 * (1 - compacted_cache_size / original_cache_size)

    return {
        'original_cache_size': original_cache_size,
        'compacted_cache_size': compacted_cache_size,
        'memory_reduction_pct': memory_reduction_pct,
    }


def print_query_generation_stats(compaction_stats: Dict):
    """
    Print query generation statistics from compaction stats.

    Parameters
    ----------
    compaction_stats : dict
        Compaction statistics containing query_generation info
    """
    if 'query_generation' not in compaction_stats:
        return

    qstats = compaction_stats['query_generation']
    query_generation_time = qstats.get('query_generation_time', 0.0)
    final_n_queries = qstats.get('final_n_queries_per_kv_head')

    # Skip printing if no query generation was performed (e.g., no_context method)
    if final_n_queries is None:
        return

    print(f"\nQuery generation statistics:")
    print(f"  Queries per KV head: {final_n_queries}")
    print(f"  Query generation time: {query_generation_time:.2f}s")

    # Print stats for each method
    if 'methods_used' in qstats:
        for method, method_info in qstats['methods_used'].items():
            print(f"\n  {method}:")
            print(f"    Fraction: {method_info['fraction']:.1%}")
            print(f"    Queries requested: {method_info['n_queries_requested_per_kv_head']}")
            print(f"    Queries actual: {method_info['n_queries_actual_per_kv_head']}")

            # Print method-specific stats
            if method == 'self_study' and 'stats' in method_info:
                details = method_info['stats']
                print(f"    Conversations generated: {details.get('n_conversations', 0)}")
                print(f"    Tokens extracted: {details.get('n_self_study_tokens_extracted', 0)}")
                if 'n_self_study_tokens_subsampled' in details:
                    print(f"    Tokens subsampled to: {details.get('n_self_study_tokens_subsampled', 0)}")


def print_train_stats(compaction_stats: Dict):
    """
    Print aggregated train statistics from compaction stats.

    Parameters
    ----------
    compaction_stats : dict
        Compaction statistics containing all_head_train_stats
    """
    if 'all_head_train_stats' not in compaction_stats:
        return

    train_stats = compaction_stats['all_head_train_stats']
    print(f"\nAll-head train statistics (eval queries per KV head: {train_stats.get('eval_queries_per_kv_head', 'N/A')}):")

    # Mean of means
    print(f"  Mean of mean output MSE: {train_stats.get('mean_mean_output_mse', 0):.6e}")
    print(f"  Mean of mean cosine sim: {train_stats.get('mean_mean_output_cosine_sim', 0):.6f}")
    print(f"  Mean of mean sumexp rel error: {train_stats.get('mean_mean_sumexp_relative_error', 0):.6e}")

    # Max metrics
    print(f"\n  Max of max output MSE: {train_stats.get('max_max_output_mse', 0):.6e} "
          f"(mean of max: {train_stats.get('mean_max_output_mse', 0):.6e})")
    print(f"  Max of max relative L2 error: {train_stats.get('max_max_output_relative_l2_error', 0):.6e} "
          f"(mean of max: {train_stats.get('mean_max_output_relative_l2_error', 0):.6e})")
    print(f"  Min of min cosine sim: {train_stats.get('min_min_output_cosine_sim', 0):.6f} "
          f"(mean of min: {train_stats.get('mean_min_output_cosine_sim', 0):.6f})")
    print(f"  Max of max sumexp rel error: {train_stats.get('max_max_sumexp_relative_error', 0):.6e} "
          f"(mean of max: {train_stats.get('mean_max_sumexp_relative_error', 0):.6e})")

    # RMS metrics
    print(f"\n  RMS of RMS output MSE: {train_stats.get('rms_rms_output_mse', 0):.6e} "
          f"(mean of RMS: {train_stats.get('mean_rms_output_mse', 0):.6e})")
    print(f"  RMS of RMS relative L2 error: {train_stats.get('rms_rms_output_relative_l2_error', 0):.6e} "
          f"(mean of RMS: {train_stats.get('mean_rms_output_relative_l2_error', 0):.6e})")
    print(f"  RMS of RMS cosine sim: {train_stats.get('rms_rms_output_cosine_sim', 0):.6f} "
          f"(mean of RMS: {train_stats.get('mean_rms_output_cosine_sim', 0):.6f})")
    print(f"  RMS of RMS sumexp rel error: {train_stats.get('rms_rms_sumexp_relative_error', 0):.6e} "
          f"(mean of RMS: {train_stats.get('mean_rms_sumexp_relative_error', 0):.6e})")


def compute_all_head_stats(per_layer_head_metrics: Dict, eval_queries_per_kv_head: int) -> Dict:
    """
    Compute aggregated statistics across all heads from per-layer-head metrics.

    For each metric, computes:
    - mean of the mean values across heads
    - max of the max values across heads (for max_* metrics)
    - mean of the max values across heads (for max_* metrics)
    - rms of the rms values across heads (for rms_* metrics)
    - mean of the rms values across heads (for rms_* metrics)

    Parameters
    ----------
    per_layer_head_metrics : dict
        Per-layer-head metrics with keys like 'L0H0', 'L0H1', etc.
    eval_queries_per_kv_head : int
        Maximum number of queries used per KV head

    Returns
    -------
    all_head_stats : dict
        Aggregated statistics across all heads
    """
    import torch

    # Collect all values for each metric
    metrics_dict = {
        'mean_output_mse': [],
        'max_output_mse': [],
        'rms_output_mse': [],
        'mean_output_relative_l2_error': [],
        'max_output_relative_l2_error': [],
        'rms_output_relative_l2_error': [],
        'mean_output_cosine_sim': [],
        'min_output_cosine_sim': [],
        'rms_output_cosine_sim': [],
        'mean_sumexp_relative_error': [],
        'max_sumexp_relative_error': [],
        'rms_sumexp_relative_error': [],
    }

    for head_key, head_metrics in per_layer_head_metrics.items():
        for metric_key in metrics_dict.keys():
            if metric_key in head_metrics:
                metrics_dict[metric_key].append(head_metrics[metric_key])

    # Compute aggregated statistics
    all_head_stats = {
        'eval_queries_per_kv_head': eval_queries_per_kv_head,
    }

    # Mean metrics
    if metrics_dict['mean_output_mse']:
        all_head_stats['mean_mean_output_mse'] = sum(metrics_dict['mean_output_mse']) / len(metrics_dict['mean_output_mse'])
    if metrics_dict['mean_output_relative_l2_error']:
        all_head_stats['mean_mean_output_relative_l2_error'] = sum(metrics_dict['mean_output_relative_l2_error']) / len(metrics_dict['mean_output_relative_l2_error'])
    if metrics_dict['mean_output_cosine_sim']:
        all_head_stats['mean_mean_output_cosine_sim'] = sum(metrics_dict['mean_output_cosine_sim']) / len(metrics_dict['mean_output_cosine_sim'])
    if metrics_dict['mean_sumexp_relative_error']:
        all_head_stats['mean_mean_sumexp_relative_error'] = sum(metrics_dict['mean_sumexp_relative_error']) / len(metrics_dict['mean_sumexp_relative_error'])

    # Max metrics: compute max and mean
    if metrics_dict['max_output_mse']:
        max_tensor = torch.tensor(metrics_dict['max_output_mse'])
        all_head_stats['max_max_output_mse'] = max_tensor.max().item()
        all_head_stats['mean_max_output_mse'] = max_tensor.mean().item()

    if metrics_dict['max_output_relative_l2_error']:
        max_tensor = torch.tensor(metrics_dict['max_output_relative_l2_error'])
        all_head_stats['max_max_output_relative_l2_error'] = max_tensor.max().item()
        all_head_stats['mean_max_output_relative_l2_error'] = max_tensor.mean().item()

    if metrics_dict['min_output_cosine_sim']:
        min_tensor = torch.tensor(metrics_dict['min_output_cosine_sim'])
        all_head_stats['min_min_output_cosine_sim'] = min_tensor.min().item()
        all_head_stats['mean_min_output_cosine_sim'] = min_tensor.mean().item()

    if metrics_dict['max_sumexp_relative_error']:
        max_tensor = torch.tensor(metrics_dict['max_sumexp_relative_error'])
        all_head_stats['max_max_sumexp_relative_error'] = max_tensor.max().item()
        all_head_stats['mean_max_sumexp_relative_error'] = max_tensor.mean().item()

    # RMS metrics: compute RMS and mean
    if metrics_dict['rms_output_mse']:
        rms_tensor = torch.tensor(metrics_dict['rms_output_mse'])
        all_head_stats['rms_rms_output_mse'] = torch.sqrt(torch.mean(rms_tensor ** 2)).item()
        all_head_stats['mean_rms_output_mse'] = rms_tensor.mean().item()

    if metrics_dict['rms_output_relative_l2_error']:
        rms_tensor = torch.tensor(metrics_dict['rms_output_relative_l2_error'])
        all_head_stats['rms_rms_output_relative_l2_error'] = torch.sqrt(torch.mean(rms_tensor ** 2)).item()
        all_head_stats['mean_rms_output_relative_l2_error'] = rms_tensor.mean().item()

    if metrics_dict['rms_output_cosine_sim']:
        rms_tensor = torch.tensor(metrics_dict['rms_output_cosine_sim'])
        all_head_stats['rms_rms_output_cosine_sim'] = torch.sqrt(torch.mean(rms_tensor ** 2)).item()
        all_head_stats['mean_rms_output_cosine_sim'] = rms_tensor.mean().item()

    if metrics_dict['rms_sumexp_relative_error']:
        rms_tensor = torch.tensor(metrics_dict['rms_sumexp_relative_error'])
        all_head_stats['rms_rms_sumexp_relative_error'] = torch.sqrt(torch.mean(rms_tensor ** 2)).item()
        all_head_stats['mean_rms_sumexp_relative_error'] = rms_tensor.mean().item()

    return all_head_stats


def print_test_stats(compaction_stats: Dict):
    """
    Print aggregated test statistics from compaction stats.

    Parameters
    ----------
    compaction_stats : dict
        Compaction statistics containing all_head_test_stats
    """
    if 'all_head_test_stats' not in compaction_stats:
        return

    test_stats = compaction_stats['all_head_test_stats']
    print(f"\nAll-head test statistics (queries per KV head: {test_stats.get('eval_queries_per_kv_head', 'N/A')}):")

    # Mean of means
    print(f"  Mean of mean output MSE: {test_stats.get('mean_mean_output_mse', 0):.6e}")
    print(f"  Mean of mean cosine sim: {test_stats.get('mean_mean_output_cosine_sim', 0):.6f}")
    print(f"  Mean of mean sumexp rel error: {test_stats.get('mean_mean_sumexp_relative_error', 0):.6e}")

    # Max metrics
    print(f"\n  Max of max output MSE: {test_stats.get('max_max_output_mse', 0):.6e} "
          f"(mean of max: {test_stats.get('mean_max_output_mse', 0):.6e})")
    print(f"  Max of max relative L2 error: {test_stats.get('max_max_output_relative_l2_error', 0):.6e} "
          f"(mean of max: {test_stats.get('mean_max_output_relative_l2_error', 0):.6e})")
    print(f"  Min of min cosine sim: {test_stats.get('min_min_output_cosine_sim', 0):.6f} "
          f"(mean of min: {test_stats.get('mean_min_output_cosine_sim', 0):.6f})")
    print(f"  Max of max sumexp rel error: {test_stats.get('max_max_sumexp_relative_error', 0):.6e} "
          f"(mean of max: {test_stats.get('mean_max_sumexp_relative_error', 0):.6e})")

    # RMS metrics
    print(f"\n  RMS of RMS output MSE: {test_stats.get('rms_rms_output_mse', 0):.6e} "
          f"(mean of RMS: {test_stats.get('mean_rms_output_mse', 0):.6e})")
    print(f"  RMS of RMS relative L2 error: {test_stats.get('rms_rms_output_relative_l2_error', 0):.6e} "
          f"(mean of RMS: {test_stats.get('mean_rms_output_relative_l2_error', 0):.6e})")
    print(f"  RMS of RMS cosine sim: {test_stats.get('rms_rms_output_cosine_sim', 0):.6f} "
          f"(mean of RMS: {test_stats.get('mean_rms_output_cosine_sim', 0):.6f})")
    print(f"  RMS of RMS sumexp rel error: {test_stats.get('rms_rms_sumexp_relative_error', 0):.6e} "
          f"(mean of RMS: {test_stats.get('mean_rms_sumexp_relative_error', 0):.6e})")


def check_reference_answers_cached(
    article_id: str,
    questions: List[Dict],
    model_name: str,
    cache_dir: str = "data/reference_answers_cache",
) -> bool:
    """
    Check if reference answers are already cached for the given article and questions.

    Parameters
    ----------
    article_id : str
        Unique identifier for the article
    questions : List[Dict]
        List of questions, each with 'question_unique_id'
    model_name : str
        Model name (for cache file naming)
    cache_dir : str
        Directory where cached generations are stored

    Returns
    -------
    bool
        True if cache exists and contains matching questions, False otherwise
    """
    import json
    import hashlib
    from pathlib import Path

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return False

    # Extract question IDs and create hash (same logic as get_or_generate_reference_answers)
    question_ids = [q.get('question_unique_id', f'q_{i}') for i, q in enumerate(questions)]
    question_ids_str = '_'.join(sorted(question_ids))
    question_ids_hash = hashlib.md5(question_ids_str.encode()).hexdigest()[:16]

    # Extract model short name
    model_short_name = model_name.split('/')[-1]

    # Check if cache file exists
    cache_file = cache_path / f"{model_short_name}_{article_id}_{question_ids_hash}.json"
    if not cache_file.exists():
        return False

    # Verify cached questions match requested questions
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        cached_question_ids = [item['question_id'] for item in cache_data['generations']]
        return cached_question_ids == question_ids
    except (json.JSONDecodeError, KeyError):
        return False


def get_or_generate_reference_answers(
    article_id: str,
    model,
    tokenizer,
    formatted_context: str,
    questions: List[Dict],
    model_name: str,
    max_new_tokens: int = 4096,
    cache_dir: str = "data/reference_answers_cache",
    device: str = "cuda",
    vllm_model=None,
) -> List[Tuple[str, List[int], str]]:
    """
    Get or generate reference answers for an article's questions.

    This function generates answers to each question using the original (uncompacted) model
    and caches the results for later use in computing perplexity and extracting test queries.
    The cache file name includes the model name, article ID, and question IDs to ensure uniqueness.

    Parameters
    ----------
    article_id : str
        Unique identifier for the article
    model : PreTrainedModel
        The model to use for generation (used as fallback if vllm_model not provided)
    tokenizer : PreTrainedTokenizer
        The tokenizer
    formatted_context : str
        Formatted context string for the article
    questions : List[Dict]
        List of questions to answer, each with 'question_unique_id', 'question', 'options'
    model_name : str
        Model name (for cache file naming)
    max_new_tokens : int
        Maximum number of tokens to generate per question
    cache_dir : str
        Directory to store cached generations
    device : str
        Device to use
    vllm_model : LLM, optional
        vLLM model instance for efficient generation. If provided, uses vLLM instead of HuggingFace.

    Returns
    -------
    results : List[Tuple[str, List[int], str]]
        List of (question_id, token_ids, generated_text) tuples for each question
    """
    import json
    import hashlib
    from pathlib import Path

    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Extract question IDs and create a hash for the cache file
    question_ids = [q.get('question_unique_id', f'q_{i}') for i, q in enumerate(questions)]
    question_ids_str = '_'.join(sorted(question_ids))
    question_ids_hash = hashlib.md5(question_ids_str.encode()).hexdigest()[:16]

    # Extract model short name (e.g., "Qwen3-4B" from "Qwen/Qwen3-4B")
    model_short_name = model_name.split('/')[-1]

    # Create cache file path: {model_short_name}_{article_id}_{question_ids_hash}.json
    cache_file = cache_path / f"{model_short_name}_{article_id}_{question_ids_hash}.json"

    # Check if cache exists
    if cache_file.exists():
        print(f"Loading cached reference answers for article {article_id} ({len(questions)} questions)")
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        # Verify that cached questions match requested questions
        cached_question_ids = [item['question_id'] for item in cache_data['generations']]
        if cached_question_ids == question_ids:
            # Re-tokenize the cached text
            results = []
            for item in cache_data['generations']:
                token_ids = tokenizer.encode(item['generated_text'], add_special_tokens=False)
                results.append((item['question_id'], token_ids, item['generated_text']))
            return results
        else:
            print(f"Warning: Cached questions don't match requested questions, regenerating...")

    # Generate answers for all questions
    print(f"Generating reference answers for article {article_id} ({len(questions)} questions)")

    # Prepare all prompts
    question_ids = [q.get('question_unique_id', f'q_{i}') for i, q in enumerate(questions)]
    full_prompts = []

    for i, q in enumerate(questions):
        question_text = q['question']
        options = q.get('options', None)

        # Format question prompt
        question_formatted = format_question(tokenizer, question_text, options, model_name)

        # Create full prompt (article context + question)
        full_prompt = formatted_context + question_formatted
        full_prompts.append(full_prompt)

    # Generate all answers in batch using vLLM (preferred) or HuggingFace (fallback)
    if vllm_model is not None:
        from models.generate import generate_with_vllm_batch
        print(f"  Generating answers in batch using vLLM...")
        # Wake up vLLM model if it's sleeping
        # Clear CUDA cache before waking up vLLM to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vllm_model.wake_up()
        gen_params = get_generation_params(model)
        generated_texts = generate_with_vllm_batch(
            vllm_model=vllm_model,
            full_prompts=full_prompts,
            max_new_tokens=max_new_tokens,
            temperature=gen_params['temperature'],
            top_k=gen_params['top_k'],
            top_p=gen_params['top_p'],
        )
        vllm_model.sleep()
    else:
        from models.generate import generate_with_full_context_batch
        print(f"  Generating answers in batch using HuggingFace...")
        generated_texts = generate_with_full_context_batch(
            model=model,
            tokenizer=tokenizer,
            full_prompts=full_prompts,
            max_new_tokens=max_new_tokens,
        )

    # Process results
    results = []
    for question_id, generated_text in zip(question_ids, generated_texts):
        # Tokenize the generated text
        gen_token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
        results.append((question_id, gen_token_ids, generated_text))

    # Save to cache (only store text, not token IDs)
    cache_data = {
        'model_name': model_name,
        'article_id': article_id,
        'question_ids': question_ids,
        'question_ids_hash': question_ids_hash,
        'max_new_tokens': max_new_tokens,
        'generations': [
            {
                'question_id': qid,
                'generated_text': text
            }
            for qid, tids, text in results
        ]
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    total_tokens = sum(len(tids) for _, tids, _ in results)
    print(f"Cached reference answers for article {article_id} ({len(questions)} questions, {total_tokens} total tokens)")

    return results


def extract_query_vectors_from_generation(
    model,
    compacted_cache: Tuple,
    generated_token_ids: List[int],
    eval_queries_per_kv_head: int,
    device: str = "cuda",
    original_seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Extract query vectors from generated tokens via forward pass on compacted cache.

    This function performs a forward pass with the model conditioned on the compacted
    cache, using the provided generated token IDs. It extracts query vectors from
    the generation portion only, then subsamples to eval_queries_per_kv_head per KV head.

    Parameters
    ----------
    model : PreTrainedModel
        The model to use
    compacted_cache : tuple
        Compacted cache (C1, beta, C2) per layer. For sliding layers, C1/C2 contain
        keys/values and beta should be zeros.
    generated_token_ids : List[int]
        Token IDs of the generated text (from original cache generation)
    eval_queries_per_kv_head : int
        Maximum number of queries to extract per KV head (will subsample if more)
    device : str
        Device to use
    original_seq_len : int, optional
        Original sequence length before compaction (for correct RoPE positioning)

    Returns
    -------
    queries : torch.Tensor
        Query vectors of shape (num_layers, num_attention_heads, n_tokens, head_dim)
        where n_tokens <= eval_queries_per_kv_head * (num_attention_heads // num_key_value_heads)
    total_tokens_seen : int
        Total number of tokens seen during extraction
    """
    # Get model config
    num_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', model.config.hidden_size // num_attention_heads)

    # Calculate queries per query head (each KV head has multiple query heads)
    num_query_heads_per_kv_head = num_attention_heads // num_kv_heads
    max_queries_per_query_head = eval_queries_per_kv_head // num_query_heads_per_kv_head

    # Storage for all layer queries
    all_layer_queries = []

    # Convert token IDs to input tensor
    gen_token_ids_tensor = torch.tensor([generated_token_ids], dtype=torch.long, device=device)

    # Register hooks to capture query vectors during forward pass
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if len(args) > 0 else kwargs.get('hidden_states')
            if hidden_states is None:
                return

            position_embeddings = kwargs.get('position_embeddings')

            # Compute queries
            q = module.q_proj(hidden_states)
            batch_size, seq_len, _ = q.shape
            q = q.view(batch_size, seq_len, num_attention_heads, head_dim)
            from models.qwen3.modeling_qwen3 import Qwen3Attention
            from models.gemma3.modeling_gemma3 import Gemma3Attention
            if isinstance(module, (Qwen3Attention, Gemma3Attention)):
                q = module.q_norm(q)
            q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

            # Apply RoPE
            if position_embeddings is not None:
                from models.qwen3.modeling_qwen3 import rotate_half
                cos, sin = position_embeddings
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
                q = (q * cos) + (rotate_half(q) * sin)

            q = q.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)

            # Store queries: (seq_len, num_heads, head_dim)
            all_layer_queries.append(q[0].detach())

        return hook_fn

    try:
        # Register hooks on all layers
        for layer_idx in range(num_layers):
            target_layer = model.model.layers[layer_idx].self_attn
            handle = target_layer.register_forward_pre_hook(
                make_hook(layer_idx),
                with_kwargs=True
            )
            hooks.append(handle)

        # Prepare compacted cache for forward pass
        from models.cache import CompactedPrefixCache
        from models.generate import get_sliding_layer_info

        # Extract sliding layer info from model
        sliding_layer_indices, sliding_window = get_sliding_layer_info(model)

        moved_layers = []
        for (C1, beta, C2) in compacted_cache:
            moved_layers.append((
                C1.to(device=device, dtype=model.dtype),
                beta.to(device=device, dtype=model.dtype),
                C2.to(device=device, dtype=model.dtype),
            ))

        cache = CompactedPrefixCache(
            tuple(moved_layers),
            original_seq_len=original_seq_len,
            sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
            sliding_window=sliding_window,
        )

        # Forward pass with generated tokens
        with torch.no_grad():
            model(
                input_ids=gen_token_ids_tensor,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

    finally:
        # Remove hooks
        for handle in hooks:
            handle.remove()

    # Process the captured queries
    if all_layer_queries:
        # Stack to: (num_layers, seq_len, num_heads, head_dim)
        queries = torch.stack(all_layer_queries, dim=0)
        # Transpose to: (num_layers, num_heads, seq_len, head_dim)
        queries = queries.permute(0, 2, 1, 3)

        total_tokens_seen = queries.shape[2]

        # Subsample if we have more queries than max_queries_per_query_head
        if total_tokens_seen > max_queries_per_query_head:
            # Random subsample
            indices = torch.randperm(total_tokens_seen, device=device)[:max_queries_per_query_head]
            indices = indices.sort()[0]
            queries = queries[:, :, indices, :]

        return queries, total_tokens_seen
    else:
        # Return empty tensor
        return torch.zeros((num_layers, num_attention_heads, 0, head_dim), device=device), 0


def compute_perplexity_on_compacted_cache(
    model,
    tokenizer,
    compacted_cache: Tuple,
    generated_token_ids: List[int],
    question_prompt: str,
    device: str = "cuda",
    original_seq_len: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute perplexity of generated text conditioned on question prompt under the compacted cache.

    The perplexity is computed for the generated answer tokens, given the question prompt
    as conditioning context. This allows us to measure how well the compacted cache preserves
    the model's ability to generate the same answer given the same question.

    Perplexity is computed as exp(average_negative_log_likelihood), which is
    a quantity normalized by sequence length.

    Parameters
    ----------
    model : PreTrainedModel
        The model to use
    tokenizer : PreTrainedTokenizer
        The tokenizer to use for encoding the question prompt
    compacted_cache : tuple
        Compacted cache (C1, beta, C2) per layer. For sliding layers, C1/C2 contain
        keys/values and beta should be zeros.
    generated_token_ids : List[int]
        Token IDs of the generated answer text
    question_prompt : str
        The formatted question prompt (to be tokenized and prepended)
    device : str
        Device to use
    original_seq_len : int, optional
        Original sequence length before compaction (for correct RoPE positioning)

    Returns
    -------
    perplexity : float
        Perplexity of the generated text under the compacted cache, conditioned on the question
    log_perplexity : float
        Log perplexity (average negative log likelihood) of the generated text
    """
    # Tokenize the question prompt
    question_token_ids = tokenizer.encode(question_prompt, add_special_tokens=False)

    # Concatenate question tokens and generated answer tokens
    full_token_ids = question_token_ids + generated_token_ids
    full_token_ids_tensor = torch.tensor([full_token_ids], dtype=torch.long, device=device)

    # Prepare compacted cache for forward pass
    from models.cache import CompactedPrefixCache
    from models.generate import get_sliding_layer_info

    # Extract sliding layer info from model
    sliding_layer_indices, sliding_window = get_sliding_layer_info(model)

    moved_layers = []
    for (C1, beta, C2) in compacted_cache:
        moved_layers.append((
            C1.to(device=device, dtype=model.dtype),
            beta.to(device=device, dtype=model.dtype),
            C2.to(device=device, dtype=model.dtype),
        ))

    cache = CompactedPrefixCache(
        tuple(moved_layers),
        original_seq_len=original_seq_len,
        sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
        sliding_window=sliding_window,
    )

    # Forward pass with question + generated tokens
    with torch.no_grad():
        outputs = model(
            input_ids=full_token_ids_tensor,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

    # Get logits and compute negative log likelihood
    # logits: (batch_size, seq_len, vocab_size)
    logits = outputs.logits

    # We only want to compute perplexity on the generated answer tokens, not the question tokens
    # The question has len(question_token_ids) tokens
    # So we compute loss starting from position len(question_token_ids)
    num_question_tokens = len(question_token_ids)
    seq_len = full_token_ids_tensor.size(1)

    start_idx = max(num_question_tokens - 1, 0)

    # Shift logits and labels for next-token prediction
    # We want to predict answer token i+1 given question + answer tokens 0..i
    # logits[num_question_tokens-1] predicts answer token 0
    # logits[num_question_tokens] predicts answer token 1, etc.
    answer_logits = logits[:, start_idx:seq_len-1, :].contiguous()  # Predictions for answer tokens
    answer_labels = full_token_ids_tensor[:, start_idx+1:seq_len].contiguous()  # Answer tokens to predict

    if answer_labels.numel() == 0:
        # No tokens to evaluate -> return NaN or some sentinel
        return float("nan"), float("nan")

    # Compute cross-entropy loss (mean reduction over answer sequence)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(answer_logits.view(-1, answer_logits.size(-1)), answer_labels.view(-1))

    # Log perplexity = loss (average negative log likelihood)
    log_perplexity = loss.item()

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()

    return perplexity, log_perplexity


def compute_perplexity_on_ground_truth_with_text(
    model,
    tokenizer,
    context_text: str,
    question_text: str,
    ground_truth: str,
    device: str = "cuda",
) -> Tuple[float, float, int]:
    """
    Compute perplexity of ground truth text given context and question (no generation).

    This is used for datasets like LongSWE-bench where we want to measure how well
    the model can predict the ground truth patch given the code context and problem statement.

    The full input is: context + question, and we compute perplexity on the ground_truth tokens.

    Parameters
    ----------
    model : PreTrainedModel
        The model to use
    tokenizer : PreTrainedTokenizer
        The tokenizer
    context_text : str
        The formatted context (e.g., code repository content with chat template applied)
    question_text : str
        The formatted question prompt (e.g., problem statement with chat template applied)
    ground_truth : str
        The ground truth text to compute perplexity on (e.g., the patch)
    device : str
        Device to use

    Returns
    -------
    perplexity : float
        Perplexity of the ground truth under the model
    log_perplexity : float
        Log perplexity (average negative log likelihood)
    num_tokens : int
        Number of ground truth tokens evaluated
    """
    # Tokenize everything
    # Context already has chat template applied, question_text is formatted
    # Ground truth is the raw patch text
    context_token_ids = tokenizer.encode(context_text, add_special_tokens=False)
    question_token_ids = tokenizer.encode(question_text, add_special_tokens=False)
    ground_truth_token_ids = tokenizer.encode(ground_truth, add_special_tokens=False)

    if len(ground_truth_token_ids) == 0:
        return float("nan"), float("nan"), 0

    # Concatenate: context + question + ground_truth
    full_token_ids = context_token_ids + question_token_ids + ground_truth_token_ids
    full_token_ids_tensor = torch.tensor([full_token_ids], dtype=torch.long, device=device)

    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(
            input_ids=full_token_ids_tensor,
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits

    # We only want perplexity on ground_truth tokens
    # The prefix is context + question
    num_prefix_tokens = len(context_token_ids) + len(question_token_ids)
    seq_len = full_token_ids_tensor.size(1)

    # For next-token prediction:
    # logits[prefix_len - 1] predicts ground_truth[0]
    # logits[prefix_len] predicts ground_truth[1], etc.
    start_idx = max(num_prefix_tokens - 1, 0)

    # Shift logits and labels for next-token prediction
    answer_logits = logits[:, start_idx:seq_len-1, :].contiguous()
    answer_labels = full_token_ids_tensor[:, start_idx+1:seq_len].contiguous()

    if answer_labels.numel() == 0:
        return float("nan"), float("nan"), 0

    # Compute cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(answer_logits.view(-1, answer_logits.size(-1)), answer_labels.view(-1))

    log_perplexity = loss.item()
    perplexity = torch.exp(loss).item()
    num_tokens = answer_labels.numel()

    return perplexity, log_perplexity, num_tokens


def compute_perplexity_on_ground_truth_with_cache(
    model,
    tokenizer,
    compacted_cache: Tuple,
    question_text: str,
    ground_truth: str,
    device: str = "cuda",
    original_seq_len: Optional[int] = None,
) -> Tuple[float, float, int]:
    """
    Compute perplexity of ground truth text given a compacted cache and question.

    This is the compacted-cache version for evaluating how well cache compaction
    preserves the model's ability to predict the ground truth.

    Parameters
    ----------
    model : PreTrainedModel
        The model to use
    tokenizer : PreTrainedTokenizer
        The tokenizer
    compacted_cache : tuple
        Compacted cache (C1, beta, C2) per layer
    question_text : str
        The formatted question prompt
    ground_truth : str
        The ground truth text to compute perplexity on
    device : str
        Device to use
    original_seq_len : int, optional
        Original sequence length before compaction (for correct RoPE positioning)

    Returns
    -------
    perplexity : float
        Perplexity of the ground truth under the compacted cache
    log_perplexity : float
        Log perplexity (average negative log likelihood)
    num_tokens : int
        Number of ground truth tokens evaluated
    """
    # Tokenize question and ground truth
    question_token_ids = tokenizer.encode(question_text, add_special_tokens=False)
    ground_truth_token_ids = tokenizer.encode(ground_truth, add_special_tokens=False)

    if len(ground_truth_token_ids) == 0:
        return float("nan"), float("nan"), 0

    # Concatenate question + ground_truth
    full_token_ids = question_token_ids + ground_truth_token_ids
    full_token_ids_tensor = torch.tensor([full_token_ids], dtype=torch.long, device=device)

    # Prepare compacted cache for forward pass
    from models.cache import CompactedPrefixCache
    from models.generate import get_sliding_layer_info

    sliding_layer_indices, sliding_window = get_sliding_layer_info(model)

    moved_layers = []
    for (C1, beta, C2) in compacted_cache:
        moved_layers.append((
            C1.to(device=device, dtype=model.dtype),
            beta.to(device=device, dtype=model.dtype),
            C2.to(device=device, dtype=model.dtype),
        ))

    cache = CompactedPrefixCache(
        tuple(moved_layers),
        original_seq_len=original_seq_len,
        sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
        sliding_window=sliding_window,
    )

    # Forward pass with question + ground_truth tokens
    with torch.no_grad():
        outputs = model(
            input_ids=full_token_ids_tensor,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

    logits = outputs.logits

    # Compute perplexity only on ground_truth tokens
    num_question_tokens = len(question_token_ids)
    seq_len = full_token_ids_tensor.size(1)

    start_idx = max(num_question_tokens - 1, 0)

    # Shift logits and labels for next-token prediction
    answer_logits = logits[:, start_idx:seq_len-1, :].contiguous()
    answer_labels = full_token_ids_tensor[:, start_idx+1:seq_len].contiguous()

    if answer_labels.numel() == 0:
        return float("nan"), float("nan"), 0

    # Compute cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(answer_logits.view(-1, answer_logits.size(-1)), answer_labels.view(-1))

    log_perplexity = loss.item()
    perplexity = torch.exp(loss).item()
    num_tokens = answer_labels.numel()

    return perplexity, log_perplexity, num_tokens
