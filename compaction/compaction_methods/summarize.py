# compaction/compaction_methods/summarize.py
"""
Summarization-based compaction method.

This method uses vLLM to generate a summary of the context, replaces the article
portion with the summarized version, and extracts a new KV cache from the
summarized context using HuggingFace transformers.
"""
import torch
from typing import Tuple, Dict, Optional, Any

from .base import FullCacheCompactionAlgorithm
from ..query_generation import QueryConfig


class SummarizeCompaction(FullCacheCompactionAlgorithm):
    """
    Compaction via summarization.

    This method:
    1. Uses vLLM to generate a summary of the article portion of the context
    2. Replaces the article text with the summary
    3. Returns the summarized context text for generation with vLLM
    """

    def __init__(
        self,
        prompt: str = "Summarize the following text:\n\n{article_text}\n\nSummary:",
        config_name: Optional[str] = None,
        return_cache: bool = False
    ):
        """
        Initialize the summarization-based compaction method.

        Parameters
        ----------
        prompt : str, optional
            Prompt template for summarization. Must contain '{article_text}' placeholder
            that will be replaced with the actual article text.
            (default: "Summarize the following text:\n\n{article_text}\n\nSummary:")
        config_name : str, optional
            Name of the configuration (used for logging). If not provided, uses "summarize".
        return_cache : bool, optional
            If True, extract and return a KV cache from the summarized context.
            If False (default), return the summarized context as text.
            Set to True when using as inner method for ChunkedCompaction.
        """
        self.prompt = prompt
        self.config_name = config_name
        self.return_cache = return_cache

        # Validate that prompt contains the placeholder
        if '{article_text}' not in self.prompt:
            raise ValueError("prompt must contain '{article_text}' placeholder")

    def name(self) -> str:
        """Return the config name if provided, otherwise 'summarize'."""
        if self.config_name:
            return self.config_name
        return "summarize"

    def returns_cache(self) -> bool:
        """Return whether this method returns a cache or text."""
        return self.return_cache

    def requires_preextracted_cache(self) -> bool:
        """Summarize method doesn't need a pre-extracted cache."""
        return False

    def compact_kv_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        compute_stats: bool = False,
        vllm_model: Optional[Any] = None,
        verbose_logging: bool = False,
        sliding_layer_indices: Optional[set] = None,
        past_key_values_for_queries: Optional[Any] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact KV cache by summarizing the context.

        Parameters
        ----------
        past_key_values : tuple of tuples
            KV cache structure: ((keys_layer0, values_layer0), ...)
            keys/values shape: (batch_size, num_heads, seq_len, head_dim)
        target_size : int
            Target compacted sequence length for the full cache.
            If indices is provided, this is the total length after partial compaction.
            The max summary tokens is computed from this target size.
        indices : range, optional
            Indices of sequence positions to compact (the article portion).
            These correspond to token positions in formatted_context.
        query_config : QueryConfig
            Configuration for query generation (not used for summarization)
        model : Any
            HuggingFace model instance for KV cache extraction
        tokenizer : Any
            Tokenizer
        formatted_context : str
            Formatted context string containing the article
        compute_stats : bool
            If True, compute train stats (not implemented for summarization)
        vllm_model : optional
            Pre-initialized vLLM model for summary generation

        Returns
        -------
        compacted_cache : tuple of tuples
            ((C1_layer0, beta_layer0, C2_layer0), ...)
            where beta is set to zero (no compaction, just replacement)
        stats : dict
            Statistics about the summarization process
        """
        if vllm_model is None:
            raise ValueError(
                "vllm_model must be provided for summarization-based compaction. "
                "Please initialize vLLM and pass it to the compact_kv_cache method."
            )

        if indices is None:
            raise ValueError(
                "indices must be provided for summarization-based compaction. "
                "This should specify which portion of the context to summarize (typically the article)."
            )

        # Get sequence length from tokenizing the formatted context
        # (past_key_values may be None for text-based methods)
        tokenized_full = tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
        full_token_ids = tokenized_full.input_ids[0]  # (seq_len,)
        seq_len = len(full_token_ids)

        # Get model config for stats
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads

        print(f"\n{'='*60}")
        print("Summarization-based compaction")
        print(f"{'='*60}")

        # Step 1: Extract the article text from formatted_context
        # (full_token_ids was already computed above)
        # Extract article token indices
        article_token_ids = full_token_ids[indices.start:indices.stop]
        article_text = tokenizer.decode(article_token_ids, skip_special_tokens=True)

        print(f"Original article length: {len(article_token_ids)} tokens")
        print(f"Article text preview: {article_text[:200]}...")

        # Compute max summary tokens from target_size
        # target_size is the total desired length after compaction
        # We need to compute how many tokens the article should be compacted to
        num_to_compact = len(indices)
        num_to_keep = seq_len - num_to_compact

        # target_size = num_to_keep + max_summary_tokens
        # max_summary_tokens = target_size - num_to_keep
        max_summary_tokens = target_size - num_to_keep

        if max_summary_tokens <= 0:
            raise ValueError(
                f"target_size ({target_size}) must be greater than the number of "
                f"positions to keep ({num_to_keep}). Got max_summary_tokens = {max_summary_tokens}"
            )

        print(f"Target size: {target_size} tokens (keeping {num_to_keep} unchanged, summarizing to {max_summary_tokens})")

        # Step 2: Generate summary using vLLM
        print(f"\nGenerating summary with vLLM (max {max_summary_tokens} tokens)...")

        # Clear CUDA cache before waking up vLLM to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Print memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"GPU Memory before vLLM wake: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
            print(f"GPU Memory available: {total - allocated:.2f}GB")

        # Wake up vLLM
        vllm_model.wake_up()

        try:
            from vllm import SamplingParams

            # Create summarization prompt using the template
            summary_prompt_text = self.prompt.format(article_text=article_text)
            summary_messages = [{"role": "user", "content": summary_prompt_text}]
            summary_prompt = tokenizer.apply_chat_template(
                summary_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            # Strip <bos> for Gemma models (standalone prompt, not concatenated)
            if summary_prompt.startswith("<bos>"):
                summary_prompt = summary_prompt[len("<bos>"):]

            # Generate summary
            sampling_params = SamplingParams(
                temperature=0.7,
                top_k=20,
                top_p=0.8,
                max_tokens=max_summary_tokens,
            ) # Default Qwen3 sampling params

            summary_outputs = vllm_model.generate([summary_prompt], sampling_params)
            summary_text = summary_outputs[0].outputs[0].text

            print(f"Generated summary: {summary_text[:200]}...")

            # Tokenize summary to check length
            summary_tokens = tokenizer.encode(summary_text, add_special_tokens=False)
            print(f"Summary length: {len(summary_tokens)} tokens")

        finally:
            # Put vLLM back to sleep
            vllm_model.sleep()

        # Step 3: Reconstruct formatted_context with summary replacing the article
        # We need to replace the article portion while preserving the chat template structure

        # Decode tokens before and after the article
        tokens_before_article = full_token_ids[:indices.start]
        tokens_after_article = full_token_ids[indices.stop:]

        text_before = tokenizer.decode(tokens_before_article, skip_special_tokens=False)
        text_after = tokenizer.decode(tokens_after_article, skip_special_tokens=False)

        # Reconstruct with summary
        summarized_context = text_before + summary_text + text_after

        # Tokenize to get the new sequence length
        summarized_token_ids = tokenizer.encode(summarized_context, add_special_tokens=False)
        new_seq_len = len(summarized_token_ids)

        print(f"\nReconstructed context with summary")
        print(f"Original context length: {len(full_token_ids)} tokens")
        print(f"Summarized context length: {new_seq_len} tokens")
        print(f"Compaction ratio: {seq_len / new_seq_len:.2f}x")

        # Compile statistics
        stats = {
            'method': 'summarize',
            'original_seq_len': seq_len,
            'tensor_compacted_seq_len': new_seq_len,
            'effective_compacted_seq_len': new_seq_len,
            'compaction_ratio': seq_len / new_seq_len,
            'original_article_tokens': len(article_token_ids),
            'summary_tokens': len(summary_tokens),
            'article_compaction_ratio': len(article_token_ids) / len(summary_tokens),
            'effective_article_tokens': len(summary_tokens),
            'tensor_article_tokens': len(summary_tokens),  # Same as effective for summarization
            'summary_text': summary_text,  # The generated summary
            'per_layer_head_metrics': {},  # Required for test stats computation
            'train_stats_time': 0.0,  # No train stats for summarization
            'summarization_params': {
                'prompt': self.prompt,
                'max_summary_tokens': max_summary_tokens,
                'target_size': target_size,
            },
        }

        if indices is not None:
            stats['is_partial_compaction'] = True
            stats['compaction_indices'] = {
                'start': indices.start,
                'end': indices.stop,
                'num_positions': len(indices),
            }
            stats['sub_target_size'] = len(summary_tokens)

        print(f"\n{'='*60}")
        print(f"Summarization complete!")
        print(f"  Original: {seq_len} tokens")
        print(f"  Summarized: {new_seq_len} tokens")
        print(f"  Compaction: {stats['compaction_ratio']:.2f}x")
        print(f"  Article: {len(article_token_ids)} → {len(summary_tokens)} tokens ({stats['article_compaction_ratio']:.2f}x)")
        print(f"{'='*60}\n")

        if not self.return_cache:
            # Return the summarized context text (for vLLM-based generation)
            return summarized_context, stats

        # Extract KV cache from the summarized context using HuggingFace model
        print("Extracting KV cache from summarized context...")
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Tokenize and run forward pass
        summarized_input_ids = tokenizer(
            summarized_context, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model(summarized_input_ids, use_cache=True)

        summarized_past_key_values = outputs.past_key_values

        # Convert (K, V) format to (C1, beta, C2) format where beta=0
        # This represents a "passthrough" compaction where the cache is used as-is
        compacted_cache = []
        sliding_layer_indices = sliding_layer_indices or set()

        for layer_idx, (keys, values) in enumerate(summarized_past_key_values):
            if layer_idx in sliding_layer_indices:
                # For sliding window layers, create placeholder tensors
                # These will be replaced by the chunked compactor
                batch_size, num_heads_layer, _, head_dim = keys.shape
                placeholder_C1 = keys.new_zeros(batch_size, num_heads_layer, 0, head_dim)
                placeholder_beta = keys.new_zeros(batch_size, num_heads_layer, 0)
                placeholder_C2 = values.new_zeros(batch_size, num_heads_layer, 0, head_dim)
                compacted_cache.append((placeholder_C1, placeholder_beta, placeholder_C2))
            else:
                # beta=0 means softmax(0)=1, so the key is used with full weight
                beta = torch.zeros(
                    keys.shape[0], keys.shape[1], keys.shape[2],
                    device=device, dtype=dtype
                )
                compacted_cache.append((keys, beta, values))

        print(f"Extracted KV cache with {new_seq_len} tokens")

        return tuple(compacted_cache), stats
