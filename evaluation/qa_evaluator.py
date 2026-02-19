# evaluation/qa_evaluator.py
"""
QA Evaluation for KV Cache Compaction.

This module evaluates compaction methods by testing their impact on
question-answering performance.
"""
import torch
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from compaction.compaction_methods import FullCacheCompactionAlgorithm
from compaction.algorithms.base import evaluate_compaction
from models.generate import (
    generate_with_compacted_cache_batch,
    generate_with_vllm_batch,
    get_generation_params,
)
from compaction.query_generation import QueryConfig
from .utils import (
    load_model_and_tokenizer,
    initialize_vllm,
    extract_full_kv_cache,
    format_question,
    parse_model_choice,
    compute_cache_memory_stats,
    print_query_generation_stats,
    print_train_stats,
    print_test_stats,
    offload_model_to_cpu,
    reload_model_to_gpu,
    compute_article_indices,
)
from .datasets import load_dataset, is_perplexity_dataset


class QAEvaluator:
    """
    Evaluate KV cache compaction methods on QA tasks.

    This class handles:
    1. Loading QA dataset
    2. Generating with original vs compacted KV cache
    3. Evaluating QA accuracy
    4. Logging detailed statistics
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = None,
        dtype: Optional[torch.dtype] = None,
        max_model_len: Optional[int] = None,
    ):
        """
        Initialize the QA evaluator.

        Parameters
        ----------
        model_name : str
            HuggingFace model name
        device : str
            Device to use ('cpu' or 'cuda')
        dtype : torch.dtype, optional
            Data type for computations
        max_model_len : int, optional
            Maximum model context length for vLLM. If None, uses model default.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.model = None
        self.tokenizer = None
        self.vllm_model = None

    def _evaluate_questions_batched(
        self,
        questions: List[Dict],
        cache_for_generation: List,
        seq_len: int,
        max_new_tokens: int,
        batch_size: int,
        results_per_question: List[Dict],
    ):
        """
        Evaluate questions using batched generation with compacted cache.

        Parameters
        ----------
        questions : list of dict
            Questions to evaluate
        cache_for_generation : list
            Compacted cache for generation
        seq_len : int
            Original sequence length
        max_new_tokens : int
            Maximum tokens to generate
        batch_size : int
            Batch size for generation
        results_per_question : list
            List to append results to (modified in place)
        """
        num_questions = len(questions)

        # Process questions in batches
        for batch_start in range(0, num_questions, batch_size):
            batch_end = min(batch_start + batch_size, num_questions)
            batch_questions = questions[batch_start:batch_end]
            actual_batch_size = len(batch_questions)

            # Reset finished sequences tracking for this batch
            if hasattr(self, '_finished_sequences_set'):
                self._finished_sequences_set.clear()
            # Reset attention mask for this batch
            if hasattr(self, '_prefill_attention_mask'):
                self._prefill_attention_mask[0] = None

            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(num_questions + batch_size - 1)//batch_size} "
                  f"(questions {batch_start+1}-{batch_end})")

            # Format all questions in the batch
            question_texts = [q['question'] for q in batch_questions]
            options_list = [q.get('options', None) for q in batch_questions]
            gold_labels = [q.get('gold_label', None) for q in batch_questions]

            # Print questions in batch
            for i, (q_text, opts, g_label) in enumerate(zip(question_texts, options_list, gold_labels)):
                q_idx = batch_start + i
                print(f"\nQ{q_idx+1}: {q_text}")
                if opts:
                    for idx, opt in enumerate(opts):
                        print(f"  {chr(65+idx)}. {opt}")
                    print(f"  Gold answer: {chr(64+g_label)} ({g_label})")

            # Format prompts
            formatted_questions = [format_question(self.tokenizer, q_text, opts, self.model_name)
                                 for q_text, opts in zip(question_texts, options_list)]

            # Batch generate with compacted cache
            gen_start_time = time.time()
            device = next(self.model.parameters()).device
            compacted_cache_gpu = tuple([
                (c1.to(device), beta.to(device), c2.to(device))
                for c1, beta, c2 in cache_for_generation
            ])

            answers = generate_with_compacted_cache_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=formatted_questions,
                compacted_cache=compacted_cache_gpu,
                max_new_tokens=max_new_tokens,
                original_seq_len=seq_len,
            )
            del compacted_cache_gpu

            gen_time = time.time() - gen_start_time

            # First pass: count total tokens in batch to compute accurate time_per_token
            total_tokens_in_batch = 0
            batch_token_counts = []
            for answer in answers:
                gen_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
                num_gen_tokens = len(gen_tokens)
                batch_token_counts.append(num_gen_tokens)
                total_tokens_in_batch += num_gen_tokens

            # Compute time per token for the entire batch (since generation is parallel)
            batch_time_per_token = gen_time / total_tokens_in_batch if total_tokens_in_batch > 0 else 0.0

            # Process results for each question in batch
            for i, (answer, q, q_text, opts, g_label) in enumerate(zip(
                answers, batch_questions, question_texts, options_list, gold_labels
            )):
                q_idx = batch_start + i
                print(f"A{q_idx+1}: {answer[:100]}...")

                num_gen_tokens = batch_token_counts[i]

                # Parse the model's choice if options were provided
                model_choice = None
                is_correct = None
                if opts:
                    model_choice = parse_model_choice(answer, max_options=len(opts))
                    if model_choice is not None:
                        print(f"  Parsed choice: {chr(64+model_choice)} ({model_choice})")
                        if g_label is not None:
                            is_correct = (model_choice == g_label)
                            print(f"  Correct: {is_correct}")
                    else:
                        print(f"  Warning: Could not parse model choice from answer")

                # Store results
                question_result = {
                    'question_id': q.get('question_unique_id', f'q_{q_idx}'),
                    'question': q_text,
                    'options': opts,
                    'gold_label': g_label,
                    'model_answer_text': answer,
                    'model_choice': model_choice,
                    'is_correct': is_correct,
                    'generation_time': gen_time / actual_batch_size,  # Approximate per-question time
                    'num_generated_tokens': num_gen_tokens,
                    'time_per_token': batch_time_per_token  # Use batch-level time per token
                }
                results_per_question.append(question_result)

            print(f"  Batch generation: {gen_time:.2f}s, {total_tokens_in_batch} tokens, {batch_time_per_token:.3f}s/token")

    def evaluate_compaction_on_article(
        self,
        article_data: Dict,
        compaction_method: FullCacheCompactionAlgorithm,
        target_size: float,
        query_config: Optional[QueryConfig] = None,
        compute_stats: bool = False,
        verbose_logging: bool = False,
        compute_perplexity: bool = False,
        perplexity_only: bool = False,
        article_idx: int = 0,
        max_new_tokens: int = 2048,
        n_questions_per_article: Optional[int] = None,
        batch_size: Optional[int] = None,
        ignore_article_indices: bool = False,
        is_perplexity_eval: bool = False,
    ) -> Dict:
        """
        Evaluate a compaction method on a single article.

        Parameters
        ----------
        article_data : dict
            QA article data
        compaction_method : FullCacheCompactionAlgorithm
            The compaction method to evaluate
        target_size : float
            Target compacted sequence length. If between 0 and 1, treated as fraction of original size.
        query_config : QueryConfig, optional
            Configuration for query generation. If None, uses default (random queries only).
        compute_stats : bool
            Whether to compute detailed statistics (cosine similarity, etc.)
            Uses original cache generation to extract test queries.
        compute_perplexity : bool
            Whether to compute perplexity of the original generation under the compacted cache
        perplexity_only : bool
            Whether to only compute perplexity without generating new answers
        article_idx : int
            Article index for logging
        max_new_tokens : int
            Maximum number of tokens to generate per question
        n_questions_per_article : int, optional
            Number of questions to use per article. If specified, uses a random shuffled set of n questions
        batch_size : int, optional
            If specified, processes questions in batches of this size.
        is_perplexity_eval : bool
            If True, this is a perplexity-based dataset (like LongSWE-bench) where we compute
            perplexity on ground_truth text instead of generating answers.

        Returns
        -------
        results : dict
            Evaluation results including QA accuracy and optional stats
        """
        from .utils import format_context

        # Determine if this is a text-based method (returns context text instead of cache)
        use_text_based_generation = not compaction_method.returns_cache()

        print(f"\n{'='*60}")
        print(f"Evaluating on article: {article_data['title']}")
        print(f"Method: {compaction_method.name()}")
        print(f"{'='*60}")

        # Initialize timing variables
        query_generation_time = 0.0
        train_stats_time = 0.0
        test_stats_time = 0.0
        effective_memory_stats = None
        context_for_generation = None  # For text-based methods

        # Ensure model/tokenizer are loaded
        if self.model is None:
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.model_name,
                self.device,
                self.dtype,
                self.max_model_len,
            )

        # Text-based methods (original, summarize) don't need pre-extracted KV cache
        # They operate on context text and return context text
        if use_text_based_generation:
            # Format the context and estimate token counts
            formatted_context = format_context(self.tokenizer, article_data['article'], model_name=self.model_name)
            # Use add_special_tokens=False since formatted_context already has <bos> from chat template
            inputs = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
            seq_len = inputs['input_ids'].shape[1]
            print(f"Formatted context: {seq_len} tokens")

            # Compute article indices for text-based methods
            article_indices = compute_article_indices(
                self.tokenizer, formatted_context, article_data['article']
            )

            # Call compact_kv_cache to get the (possibly modified) context text
            # For 'original', this returns formatted_context unchanged
            # For 'summarize', this returns the summarized context
            start_time = time.time()
            context_for_generation, compaction_stats = compaction_method.compact_kv_cache(
                past_key_values=None,  # Text-based methods don't need pre-extracted cache
                target_size=seq_len,
                indices=article_indices,
                query_config=None,
                model=self.model,
                tokenizer=self.tokenizer,
                formatted_context=formatted_context,
                compute_stats=False,
                vllm_model=self.vllm_model,
            )
            compaction_time = time.time() - start_time
            extraction_time = 0.0

            # Get the actual context length after any transformation
            context_tokens = self.tokenizer.encode(context_for_generation, add_special_tokens=False)
            effective_seq_len = len(context_tokens)
            print(f"Context for generation: {effective_seq_len} tokens")

            # article_indices was computed above for text-based methods
            article_len = len(article_indices)
            non_article_tokens = seq_len - article_len
            past_key_values = None
            compacted_cache = None  # Will be built later if needed for perplexity/stats

        else:
            # Cache-based methods: extract KV cache and compact it
            # Check if method needs pre-extracted KV cache
            # Methods like ChunkedCompaction handle their own extraction to avoid OOM
            needs_preextracted_cache = compaction_method.requires_preextracted_cache()

            if needs_preextracted_cache:
                # Extract full KV cache from article (traditional path)
                print(f"Extracting KV cache from formatted article...")
                start_time = time.time()
                seq_len, past_key_values, article_indices, formatted_context, _ = extract_full_kv_cache(
                    self.model,
                    self.tokenizer,
                    article_data['article'],
                    self.device,
                    model_name=self.model_name,
                )
                extraction_time = time.time() - start_time

                print(f"Formatted context tokenized to {seq_len} tokens")

                if ignore_article_indices:
                    print(f"[ABLATION] Ignoring article boundaries - treating entire sequence as article")
                    print(f"Full sequence: {seq_len} tokens")
                else:
                    print(f"Article portion: tokens {article_indices.start}-{article_indices.stop} ({len(article_indices)} tokens)")
                    print(f"Tokens to keep unchanged: {seq_len - len(article_indices)} tokens")
                print(f"KV cache extraction took {extraction_time:.2f}s")
            else:
                # Method handles its own KV cache extraction (e.g., ChunkedCompaction)
                # Just format the context to get article token count
                print(f"Method handles its own KV cache extraction (memory-efficient path)")

                formatted_context = format_context(self.tokenizer, article_data['article'], model_name=self.model_name)

                # Obtain token counts without running full prefill
                # Use add_special_tokens=False since formatted_context already has <bos> from chat template
                inputs = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
                seq_len = inputs['input_ids'].shape[1]

                # Find article boundaries
                article_indices = compute_article_indices(
                    self.tokenizer, formatted_context, article_data['article']
                )
                past_key_values = None  # Will be handled by the compaction method
                extraction_time = 0.0

                print(f"Formatted context has {seq_len} tokens")
                print(f"Article portion: tokens {article_indices.start}-{article_indices.stop} ({len(article_indices)} tokens)")
                print(f"Skipped full prefill (method will extract caches as needed)")

            # Determine article length for cache-based methods
            if ignore_article_indices:
                article_len = seq_len
                non_article_tokens = 0
            else:
                article_len = len(article_indices)
                non_article_tokens = seq_len - article_len

            # For cache-based methods, effective_seq_len is set after compaction (to compacted length)
            # Initialize to seq_len here; will be updated after compaction
            effective_seq_len = seq_len

        target_size_param = target_size

        # Detect sliding window layers from model config (for models like Gemma3)
        # This is needed for proper cache handling in perplexity computation
        sliding_layer_indices = set()
        sliding_window = None
        if self.model is not None:
            config = getattr(self.model, 'config', None)
            if config is not None:
                layer_types = getattr(config, 'layer_types', None)
                sliding_window = getattr(config, 'sliding_window', None)
                if layer_types is not None:
                    for layer_idx, layer_type in enumerate(layer_types):
                        if layer_type == "sliding_attention":
                            sliding_layer_indices.add(layer_idx)
                if sliding_layer_indices:
                    print(f"Detected {len(sliding_layer_indices)} sliding window layers")

        # For cache-based methods, compute target size and run compaction
        # For text-based methods, compaction was already done above
        original_cache_size = None
        compacted_cache_size = None
        tensor_compacted_seq_len = None
        effective_compacted_seq_len = None

        if not use_text_based_generation:
            # Determine what to compact based on ignore_article_indices flag
            if ignore_article_indices:
                indices_for_compaction = None  # None means compact entire sequence
            else:
                indices_for_compaction = article_indices

            article_target_size = article_len

            # Compute actual target size (handle fractional values) with article-based semantics
            if article_len > 0:
                if 0 < target_size_param < 1:
                    requested_article_size = max(1, int(article_len * target_size_param))
                    source_desc = f"{target_size_param:.1%} of {article_len}"
                else:
                    requested_article_size = max(0, int(target_size_param))
                    source_desc = f"requested {target_size_param}"
                article_target_size = min(article_len, requested_article_size)
                if article_target_size != requested_article_size:
                    print(f"Target article size: requested {requested_article_size} tokens but article has {article_len}; using {article_target_size}")
                else:
                    print(f"Target article size: {source_desc} = {article_target_size} tokens")
            else:
                article_target_size = 0
                print("Article portion is empty; target article size set to 0")

            actual_target_size = article_target_size + non_article_tokens
            if ignore_article_indices:
                print(f"Total target size: {actual_target_size} tokens (entire sequence)")
            else:
                print(f"Total target size (article {article_target_size} + non-article {non_article_tokens}) = {actual_target_size} tokens")
            print(f"Compacting KV cache to size {actual_target_size}...")
            if ignore_article_indices:
                print(f"Compacting entire sequence")
            else:
                print(f"Only compacting article portion: tokens {article_indices.start}-{article_indices.stop}")

            start_time = time.time()

            # Use provided QueryConfig
            methods_str = ", ".join([f"{mc.method}({mc.fraction:.1%})" for mc in query_config.method_configs])
            print(f"Using QueryConfig: methods=[{methods_str}], "
                  f"max_query_vectors_per_kv_head={query_config.max_query_vectors_per_kv_head}")

            compacted_cache, compaction_stats = compaction_method.compact_kv_cache(
                past_key_values=past_key_values,
                target_size=actual_target_size,
                indices=indices_for_compaction,  # None if ignoring article boundaries, else article_indices
                query_config=query_config,
                model=self.model,
                tokenizer=self.tokenizer,
                formatted_context=formatted_context,
                compute_stats=compute_stats,
                verbose_logging=verbose_logging,
                vllm_model=self.vllm_model,  # Pass vLLM model
                sliding_layer_indices=sliding_layer_indices,
            )
            total_compaction_time = time.time() - start_time

            # Extract query generation time and compute compaction time excluding query generation
            if 'query_generation' in compaction_stats:
                qstats = compaction_stats['query_generation']
                if 'query_generation_time' in qstats:
                    query_generation_time = qstats['query_generation_time']

            # Extract train stats computation time if available
            if 'train_stats_time' in compaction_stats:
                train_stats_time = compaction_stats['train_stats_time']

            # Compaction time excludes query generation and train stats computation
            compaction_time = total_compaction_time - query_generation_time - train_stats_time

            # For models with sliding window layers, inject the original KV cache into the compacted_cache
            # The compacted_cache has placeholder (size-0) tensors for sliding layers that need to be replaced
            if sliding_layer_indices and sliding_window is not None and past_key_values is not None:
                # Non-chunked path: inject sliding layer data from original cache
                compacted_cache_list = list(compacted_cache)
                for layer_idx in sliding_layer_indices:
                    keys = past_key_values[layer_idx][0].clone()
                    values = past_key_values[layer_idx][1].clone()
                    beta = torch.zeros(keys.shape[0], keys.shape[1], keys.shape[2],
                                     device=keys.device, dtype=keys.dtype)
                    compacted_cache_list[layer_idx] = (keys, beta, values)
                compacted_cache = tuple(compacted_cache_list)
                print(f"Injected sliding layer data from original cache ({len(sliding_layer_indices)} layers)")
            # For chunked compaction, sliding layer data is already in the compacted_cache

            # Compute memory stats now while we still have past_key_values
            # If past_key_values is None (chunked compaction path), estimate from compaction_stats
            if past_key_values is not None:
                memory_stats = compute_cache_memory_stats(past_key_values, compacted_cache)
                original_cache_size = memory_stats['original_cache_size']
                compacted_cache_size = memory_stats['compacted_cache_size']
            else:
                # Estimate original cache size from article length and compacted cache shape
                # For chunked compaction, we don't have the original cache
                num_layers = len(compacted_cache)
                num_heads = compacted_cache[0][0].shape[1]
                head_dim = compacted_cache[0][0].shape[3]
                # Original would have had seq_len tokens for global layers,
                # but sliding layers only store min(seq_len, sliding_window - 1) tokens
                num_global_layers = num_layers - len(sliding_layer_indices)
                num_sliding_layers = len(sliding_layer_indices)
                global_layer_size = 2 * num_heads * seq_len * head_dim
                if sliding_window is not None:
                    sliding_layer_size = 2 * num_heads * min(seq_len, sliding_window - 1) * head_dim
                else:
                    sliding_layer_size = global_layer_size
                original_cache_size = (num_global_layers * global_layer_size +
                                       num_sliding_layers * sliding_layer_size)
                # compacted_cache now includes sliding layer data in C1/C2
                compacted_cache_size = sum(
                    layer[0].numel() + layer[1].numel() + layer[2].numel()
                    for layer in compacted_cache
                )

            reported_tensor_len = compaction_stats.get('tensor_compacted_seq_len')
            # Compute average tensor length across all global (non-sliding) layers
            # For nonuniform caches, different layers can have different sequence lengths
            if reported_tensor_len is None:
                total_tensor_len = 0
                num_global_layers = 0
                for i in range(len(compacted_cache)):
                    if i not in sliding_layer_indices:
                        total_tensor_len += compacted_cache[i][0].shape[2]
                        num_global_layers += 1
                tensor_compacted_seq_len = total_tensor_len / num_global_layers if num_global_layers > 0 else actual_target_size
                compaction_stats['tensor_compacted_seq_len'] = tensor_compacted_seq_len
            else:
                tensor_compacted_seq_len = reported_tensor_len

            effective_len = compaction_stats.get('effective_compacted_seq_len')
            if effective_len is None:
                effective_len = tensor_compacted_seq_len
                compaction_stats['effective_compacted_seq_len'] = effective_len
            effective_compacted_seq_len = effective_len
            if effective_compacted_seq_len > 0:
                compaction_stats['compaction_ratio'] = seq_len / effective_compacted_seq_len

            effective_ratio = effective_compacted_seq_len / tensor_compacted_seq_len if tensor_compacted_seq_len > 0 else 1.0
            effective_ratio = min(1.0, effective_ratio) if effective_ratio > 0 else 0.0
            effective_compacted_cache_size = compacted_cache_size * effective_ratio
            effective_memory_reduction_pct = 100 * (1 - effective_compacted_cache_size / original_cache_size)
            effective_memory_stats = {
                'effective_compacted_cache_size': effective_compacted_cache_size,
                'effective_memory_reduction_pct': effective_memory_reduction_pct,
            }

            timing_parts = [f"query gen: {query_generation_time:.2f}s"]
            if train_stats_time > 0:
                timing_parts.append(f"train stats: {train_stats_time:.2f}s")
            timing_parts.append(f"compaction: {compaction_time:.2f}s")
            print(f"Total time: {total_compaction_time:.2f}s ({', '.join(timing_parts)})")

            # Update effective_seq_len to the compacted length for cache-based methods
            effective_seq_len = effective_compacted_seq_len
        else:
            # Text-based methods: compaction already done, set variables for consistency
            tensor_compacted_seq_len = effective_seq_len
            effective_compacted_seq_len = effective_seq_len

        # Print query generation stats
        print_query_generation_stats(compaction_stats)

        # Print aggregated train stats if available
        print_train_stats(compaction_stats)

        # Evaluate on questions (will also extract test queries if compute_stats=True)
        # We need to do this before computing detailed stats to extract test queries
        all_questions = article_data['questions']

        # Shuffle and select n_questions_per_article if specified
        if n_questions_per_article is not None:
            import random
            rng = random.Random(67)
            questions = all_questions.copy()
            rng.shuffle(questions)
            questions = questions[:n_questions_per_article]
            print(f"Article has {len(all_questions)} questions (using random {len(questions)} questions)")
        else:
            questions = all_questions
            print(f"Article has {len(questions)} questions")

        # For text-based methods, use context_for_generation; otherwise use formatted_context
        context_for_ref_answers = context_for_generation if use_text_based_generation else formatted_context

        # Generate or load reference answers if needed for stats or perplexity
        # Skip for perplexity-based datasets (like LongSWE-bench) - no generation needed
        reference_answers = None
        if (compute_stats or compute_perplexity) and not is_perplexity_eval:
            from .utils import get_or_generate_reference_answers, check_reference_answers_cached

            # Check if answers are already cached before offloading model
            is_cached = check_reference_answers_cached(
                article_id=article_data['article_id'],
                questions=questions,
                model_name=self.model_name,
            )

            # Only offload HuggingFace model if we need to generate (not cached)
            needs_offload = not is_cached and self.vllm_model is not None and self.model is not None
            if needs_offload:
                print("Offloading HuggingFace model to CPU for vLLM generation...")
                offload_model_to_cpu(self.model)

            reference_answers = get_or_generate_reference_answers(
                article_id=article_data['article_id'],  # globally unique article id
                model=self.model,
                tokenizer=self.tokenizer,
                formatted_context=context_for_ref_answers,
                questions=questions,
                model_name=self.model_name,
                max_new_tokens=max_new_tokens,
                device=self.device,
                vllm_model=self.vllm_model,
            )

            # Reload HuggingFace model back to GPU after vLLM generation
            if needs_offload:
                print("Reloading HuggingFace model to GPU...")
                reload_model_to_gpu(self.model, self.device)

            total_tokens = sum(len(tids) for _, tids, _ in reference_answers)
            print(f"Reference answers: {len(reference_answers)} questions, {total_tokens} total tokens")

        # For text-based methods, we may need to extract KV cache later for perplexity/stats
        # This is done lazily after answer generation
        context_past_key_values = None

        # Free the KV cache after creating sliding layer data (if needed)
        if past_key_values is not None and not compute_stats:
            del past_key_values

        # QA evaluation - generate answers for all questions
        results_per_question = []
        num_test_questions = len(questions)

        # Initialize perplexity tracking variables
        perplexities_per_question = []
        avg_perplexity = None
        avg_log_perplexity = None

        # Handle perplexity-based datasets (like LongSWE-bench)
        # These have ground_truth instead of options/gold_label
        if is_perplexity_eval:
            print(f"\n{'='*60}")
            print(f"Computing perplexity on ground_truth (perplexity-based dataset)...")
            print(f"{'='*60}")

            from .utils import (
                compute_perplexity_on_ground_truth_with_text,
                compute_perplexity_on_ground_truth_with_cache,
            )

            for i, q in enumerate(questions):
                question_text = q['question']
                ground_truth = q.get('ground_truth', '')

                if not ground_truth:
                    print(f"  Warning: No ground_truth for question {i+1}, skipping")
                    continue

                # Format the question (for perplexity eval, no options)
                question_formatted = format_question(self.tokenizer, question_text, options=None, model_name=self.model_name)

                print(f"  Question {i+1}/{len(questions)}: {q.get('question_unique_id', f'q_{i}')}")
                print(f"    Ground truth length: {len(ground_truth)} chars")

                if use_text_based_generation:
                    # Text-based methods (original, summarize, etc.): use full context text
                    perplexity, log_perplexity, num_tokens = compute_perplexity_on_ground_truth_with_text(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        context_text=context_for_generation,
                        question_text=question_formatted,
                        ground_truth=ground_truth,
                        device=self.device,
                    )
                else:
                    # Cache-based methods: use compacted cache
                    perplexity, log_perplexity, num_tokens = compute_perplexity_on_ground_truth_with_cache(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        compacted_cache=compacted_cache,
                        question_text=question_formatted,
                        ground_truth=ground_truth,
                        device=self.device,
                        original_seq_len=seq_len,
                    )

                perplexities_per_question.append({
                    'question_id': q.get('question_unique_id', f'q_{i}'),
                    'perplexity': perplexity,
                    'log_perplexity': log_perplexity,
                    'num_tokens': num_tokens,
                })
                print(f"    Perplexity: {perplexity:.2f}, Log perplexity: {log_perplexity:.4f} ({num_tokens} tokens)")

            # Compute average perplexity and log perplexity
            valid_perplexities = [p for p in perplexities_per_question if not (p['perplexity'] != p['perplexity'])]  # filter NaN
            if valid_perplexities:
                avg_perplexity = sum(p['perplexity'] for p in valid_perplexities) / len(valid_perplexities)
                avg_log_perplexity = sum(p['log_perplexity'] for p in valid_perplexities) / len(valid_perplexities)
                print(f"\n  Average perplexity: {avg_perplexity:.2f}")
                print(f"  Average log perplexity: {avg_log_perplexity:.4f}")
            else:
                avg_perplexity = float('nan')
                avg_log_perplexity = float('nan')

        elif not perplexity_only:
            if use_text_based_generation:
                # Text-based methods: use vLLM for generation
                print(f"\n{'='*60}")
                print(f"Generating answers using vLLM with {compaction_method.name()} context...")
                print(f"{'='*60}")

                if self.vllm_model is None:
                    raise ValueError(
                        f"vLLM model required for text-based method '{compaction_method.name()}'. "
                        "Please initialize with use_vllm=True."
                    )

                # Offload HuggingFace model to CPU before using vLLM
                if self.model is not None:
                    offload_model_to_cpu(self.model)
                    torch.cuda.empty_cache()

                # Wake up vLLM
                self.vllm_model.wake_up()

                try:
                    # Use batch_size=1 for sequential generation
                    effective_batch_size = batch_size if batch_size is not None and batch_size > 0 else 1

                    for batch_start in range(0, len(questions), effective_batch_size):
                        batch_end = min(batch_start + effective_batch_size, len(questions))
                        batch_questions = questions[batch_start:batch_end]

                        # Format prompts
                        prompts = []
                        for q in batch_questions:
                            question_text = q['question']
                            options = q.get('options', None)
                            question_formatted = format_question(self.tokenizer, question_text, options, self.model_name)
                            full_prompt = context_for_generation + question_formatted
                            prompts.append(full_prompt)

                        # Generate batch
                        gen_start = time.time()
                        gen_params = get_generation_params(self.model)
                        answers = generate_with_vllm_batch(
                            vllm_model=self.vllm_model,
                            full_prompts=prompts,
                            max_new_tokens=max_new_tokens,
                            temperature=gen_params['temperature'],
                            top_k=gen_params['top_k'],
                            top_p=gen_params['top_p'],
                        )
                        gen_time = time.time() - gen_start

                        # Process results
                        for i, (q, answer) in enumerate(zip(batch_questions, answers)):
                            correct_answer = q.get('gold_label', None)
                            model_choice = parse_model_choice(answer)
                            is_correct = model_choice == correct_answer if model_choice and correct_answer else False

                            gen_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
                            num_gen_tokens = len(gen_tokens)

                            print(f"Q{batch_start + i + 1}: {q['question'][:50]}...")
                            print(f"  A: {answer[:100]}... | Choice: {model_choice} | Correct: {correct_answer} | {'✓' if is_correct else '✗'}")

                            per_question_time = gen_time / len(batch_questions)
                            time_per_token = per_question_time / num_gen_tokens if num_gen_tokens > 0 else 0.0
                            results_per_question.append({
                                'question_id': q.get('question_unique_id', f'q_{batch_start + i}'),
                                'question': q['question'],
                                'options': q.get('options', []),
                                'gold_label': correct_answer,
                                'model_answer_text': answer,
                                'model_choice': model_choice,
                                'is_correct': is_correct,
                                'generation_time': per_question_time,
                                'num_generated_tokens': num_gen_tokens,
                                'time_per_token': time_per_token,
                            })
                finally:
                    # Put vLLM back to sleep
                    self.vllm_model.sleep()

                    # Reload HuggingFace model back to GPU
                    if self.model is not None:
                        reload_model_to_gpu(self.model, self.device)
            else:
                # Cache-based methods: use compacted cache for generation
                print(f"\n{'='*60}")
                print(f"Generating answers using COMPACTED cache...")

                if batch_size == 0:
                    batch_size = None
                if batch_size is not None:
                    print(f"Using batched generation with batch_size={batch_size}")
                print(f"{'='*60}")

                # Convert cache format for generation
                # Keep master copy on CPU to save GPU memory; will transfer per-batch
                cache_for_generation = []
                for (C1, beta, C2) in compacted_cache:
                    cache_for_generation.append((C1.cpu(), beta.cpu(), C2.cpu()))

                effective_batch_size = batch_size if batch_size is not None and batch_size > 0 else 1

                self._evaluate_questions_batched(
                    questions=questions,
                    cache_for_generation=cache_for_generation,
                    seq_len=seq_len,
                    max_new_tokens=max_new_tokens,
                    batch_size=effective_batch_size,
                    results_per_question=results_per_question,
                )
        else:
            print(f"\n{'='*60}")
            print(f"Skipping answer generation (perplexity-only mode)...")
            print(f"{'='*60}")

        # For text-based methods, extract KV cache from context_for_generation if needed for perplexity/stats
        # Also extract original cache from formatted_context if compute_stats is enabled
        original_cache_for_text_based = None
        if use_text_based_generation and (compute_stats or compute_perplexity):
            # For compute_stats, we need the original cache to compare against the current cache
            if compute_stats:
                print(f"\nExtracting ORIGINAL KV cache for stats comparison...")
                # Use add_special_tokens=False since formatted_context already has <bos> from chat template
                original_inputs = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False).to(self.device)
                with torch.no_grad():
                    original_outputs = self.model(
                        **original_inputs,
                        use_cache=True,
                        return_dict=True,
                    )
                original_cache_for_text_based = original_outputs.past_key_values
                print(f"Extracted original cache: {len(original_cache_for_text_based)} layers, {original_cache_for_text_based[0][0].shape[2]} tokens")

            print(f"\nExtracting KV cache from current context for perplexity/stats computation...")

            # Run forward pass to get KV cache from current context
            inputs = self.tokenizer(context_for_generation, return_tensors="pt", add_special_tokens=False).to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    use_cache=True,
                    return_dict=True,
                )
            context_past_key_values = outputs.past_key_values

            # Convert KV cache to compacted format (C1, beta=0, C2) for all layers
            # For sliding layers, this stores the original KV data; the generation
            # functions will extract sliding layer info from the model config
            compacted_layers = []
            num_layers = len(context_past_key_values)
            for layer_idx in range(num_layers):
                keys = context_past_key_values[layer_idx][0]
                values = context_past_key_values[layer_idx][1]
                beta = torch.zeros(keys.shape[:-1], device=keys.device, dtype=keys.dtype)
                compacted_layers.append((keys, beta, values))
            compacted_cache = tuple(compacted_layers)

            print(f"Built compacted cache: {num_layers} layers, {effective_seq_len} tokens")

        # Compute detailed stats using queries from reference answers if requested
        if compute_stats:
            if reference_answers is not None and compacted_cache is not None:
                print(f"\n{'='*60}")
                print(f"Computing detailed stats using queries from reference answers...")
                print(f"{'='*60}")

                # Concatenate all generated token IDs from all questions
                all_gen_token_ids = []
                for _, token_ids, _ in reference_answers:
                    all_gen_token_ids.extend(token_ids)

                # Extract query vectors from original generation via forward pass on compacted cache
                # Use effective_seq_len for text-based methods, seq_len for cache-based
                original_seq_len_for_stats = effective_seq_len if use_text_based_generation else seq_len

                from .utils import extract_query_vectors_from_generation
                stats_start_time = time.time()
                test_queries_concat, total_tokens_seen = extract_query_vectors_from_generation(
                    model=self.model,
                    compacted_cache=compacted_cache,
                    generated_token_ids=all_gen_token_ids,
                    eval_queries_per_kv_head=query_config.eval_queries_per_kv_head,
                    device=self.device,
                    original_seq_len=original_seq_len_for_stats,
                )

                if test_queries_concat.shape[2] > 0:
                    print(f"Extracted {test_queries_concat.shape[2]} query tokens per head (sampled from {total_tokens_seen} seen)")

                    # Compute detailed stats (requires original cache)
                    # For text-based methods, use the original (un-summarized) cache we extracted earlier
                    # For cache-based methods, use past_key_values
                    if use_text_based_generation:
                        original_cache_for_stats = original_cache_for_text_based
                    else:
                        original_cache_for_stats = past_key_values

                    if original_cache_for_stats is not None:
                        detailed_stats = self._compute_detailed_stats(
                            original_cache_for_stats,
                            compacted_cache,
                            test_queries_concat,
                            query_config.eval_queries_per_kv_head,
                            sliding_layer_indices=sliding_layer_indices,
                        )
                        test_stats_time = time.time() - stats_start_time
                        print(f"Computing detailed stats took {test_stats_time:.2f}s")

                        # Merge test stats into per_layer_head_metrics if it exists
                        per_layer_head_test_stats = detailed_stats['per_layer_head_test_stats']
                        if 'per_layer_head_metrics' in compaction_stats:
                            for head_key, test_metrics in per_layer_head_test_stats.items():
                                if head_key in compaction_stats['per_layer_head_metrics']:
                                    # Add test_stats to the existing head entry
                                    compaction_stats['per_layer_head_metrics'][head_key]['test_stats'] = test_metrics
                                else:
                                    # If for some reason the head doesn't exist, create it
                                    compaction_stats['per_layer_head_metrics'][head_key] = {'test_stats': test_metrics}

                        # Add all-head test stats to compaction_stats
                        from .utils import compute_all_head_stats
                        compaction_stats['all_head_test_stats'] = compute_all_head_stats(
                            detailed_stats['per_layer_head_test_stats'],
                            query_config.eval_queries_per_kv_head
                        )

                        # Print all-head test stats
                        print_test_stats(compaction_stats)
                        if 'per_layer_head_metrics' in compaction_stats:
                            print(f"Test stats merged into per_layer_head_metrics successfully")
                    elif '_original_chunk_caches' in compaction_stats and compaction_stats['_original_chunk_caches'] is not None:
                        # Chunked compaction path: compute test stats per-chunk and aggregate
                        print(f"Computing test stats for chunked compaction...")
                        detailed_stats = self._compute_chunked_test_stats(
                            compaction_stats['_original_chunk_caches'],
                            compaction_stats['_compacted_chunk_caches'],
                            test_queries_concat,
                            query_config.eval_queries_per_kv_head,
                            compaction_stats.get('chunk_stats', [])
                        )
                        test_stats_time = time.time() - stats_start_time
                        print(f"Computing chunked test stats took {test_stats_time:.2f}s")

                        # Add all-head test stats
                        from .utils import compute_all_head_stats
                        compaction_stats['all_head_test_stats'] = compute_all_head_stats(
                            detailed_stats['per_layer_head_test_stats'],
                            query_config.eval_queries_per_kv_head
                        )

                        # Print all-head test stats
                        print_test_stats(compaction_stats)

                        # Clean up the temporary caches from stats (don't serialize tensors to JSON)
                        del compaction_stats['_original_chunk_caches']
                        del compaction_stats['_compacted_chunk_caches']
                    else:
                        # No original cache available
                        print(f"Skipping detailed stats (no original cache available for comparison)")
                        test_stats_time = 0.0
                else:
                    print(f"Warning: No test queries extracted, skipping detailed stats")
                    test_stats_time = 0.0
            else:
                print(f"Warning: No reference answers available, skipping detailed stats")
                test_stats_time = 0.0
        else:
            test_stats_time = 0.0

        # Compute perplexity if requested (for QA datasets, not perplexity-based datasets)
        # Skip if is_perplexity_eval since perplexity was already computed above
        if compute_perplexity and not is_perplexity_eval:
            if reference_answers is not None and compacted_cache is not None:
                print(f"\n{'='*60}")
                print(f"Computing perplexity of reference answers under compacted cache...")
                print(f"{'='*60}")

                from .utils import compute_perplexity_on_compacted_cache

                # Compute perplexity for each question's generation
                for i, (question_id, gen_token_ids, gen_text) in enumerate(reference_answers):
                    q = questions[i]
                    question_text = q['question']
                    options = q.get('options', None)

                    # Format the question
                    question_formatted = format_question(self.tokenizer, question_text, options, self.model_name)

                    print(f"  Computing perplexity for question {i+1}/{len(reference_answers)}: {question_id}")

                    # Compute perplexity conditioned on article + question
                    # Use effective_seq_len for text-based methods, seq_len for cache-based
                    original_seq_len_for_ppl = effective_seq_len if use_text_based_generation else seq_len
                    perplexity, log_perplexity = compute_perplexity_on_compacted_cache(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        compacted_cache=compacted_cache,
                        generated_token_ids=gen_token_ids,
                        question_prompt=question_formatted,
                        device=self.device,
                        original_seq_len=original_seq_len_for_ppl,
                    )

                    perplexities_per_question.append({
                        'question_id': question_id,
                        'perplexity': perplexity,
                        'log_perplexity': log_perplexity,
                        'num_tokens': len(gen_token_ids)
                    })
                    print(f"    Perplexity: {perplexity:.2f}, Log perplexity: {log_perplexity:.4f} ({len(gen_token_ids)} tokens)")

                # Compute average perplexity and average log perplexity
                avg_perplexity = sum(p['perplexity'] for p in perplexities_per_question) / len(perplexities_per_question)
                avg_log_perplexity = sum(p['log_perplexity'] for p in perplexities_per_question) / len(perplexities_per_question)
                print(f"\n  Average perplexity: {avg_perplexity:.2f}")
                print(f"  Average log perplexity: {avg_log_perplexity:.4f}")
            else:
                print(f"Warning: No reference answers available, skipping perplexity computation")

        # Compute accuracy metrics
        if is_perplexity_eval:
            # Perplexity-based dataset: no accuracy metrics, just perplexity
            qa_results = {
                'num_questions': len(questions),
                'results_per_question': [],
                'note': 'Perplexity-based evaluation - computed perplexity on ground_truth',
                'is_perplexity_eval': True,
            }
        elif not perplexity_only:
            total_questions = len(results_per_question)
            correct_answers = sum(1 for r in results_per_question if r.get('is_correct', False))
            parseable_answers = sum(1 for r in results_per_question if r.get('model_choice') is not None)

            # Compute generation timing metrics
            total_gen_time = sum(r.get('generation_time', 0.0) for r in results_per_question)
            total_gen_tokens = sum(r.get('num_generated_tokens', 0) for r in results_per_question)
            avg_gen_time_per_question = total_gen_time / total_questions if total_questions > 0 else 0.0
            avg_time_per_token = total_gen_time / total_gen_tokens if total_gen_tokens > 0 else 0.0

            qa_results = {
                'num_questions': len(questions),
                'results_per_question': results_per_question,
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'parseable_answers': parseable_answers,
                'accuracy': correct_answers / total_questions if total_questions > 0 else 0.0,
                'parse_rate': parseable_answers / total_questions if total_questions > 0 else 0.0,
                'total_generation_time': total_gen_time,
                'total_generated_tokens': total_gen_tokens,
                'avg_generation_time_per_question': avg_gen_time_per_question,
                'avg_time_per_token': avg_time_per_token,
            }

            if use_text_based_generation:
                qa_results['note'] = f'Generated answers with {compaction_method.name()} context for {num_test_questions} questions'
            else:
                qa_results['note'] = f'Generated answers with compacted cache for {num_test_questions} questions'

            print(f"\n{'='*60}")
            print(f"QA Results Summary:")
            print(f"  Total questions: {total_questions}")
            print(f"  Parseable answers: {parseable_answers} ({qa_results['parse_rate']:.1%})")
            print(f"  Correct answers: {correct_answers} ({qa_results['accuracy']:.1%})")
            print(f"  Total generation time: {total_gen_time:.2f}s")
            print(f"  Total generated tokens: {total_gen_tokens}")
            print(f"  Avg generation time/question: {avg_gen_time_per_question:.2f}s")
            if avg_time_per_token > 0:
                print(f"  Avg time per token: {avg_time_per_token:.3f}s/token ({1.0/avg_time_per_token:.1f} tokens/s)")
            print(f"{'='*60}")
        else:
            # perplexity_only mode: no QA evaluation performed
            qa_results = {
                'num_questions': len(questions),
                'results_per_question': [],
                'note': 'Perplexity-only mode - no answers generated',
            }

        # Get article-specific compaction stats
        # tensor_article_tokens = actual tensor size of article portion (includes padding)
        tensor_article_tokens = compaction_stats.get('tensor_article_tokens')
        if isinstance(tensor_article_tokens, list):
            tensor_article_tokens = (
                sum(float(x) for x in tensor_article_tokens) / len(tensor_article_tokens)
                if tensor_article_tokens else 0.0
            )
        if tensor_article_tokens is None:
            tensor_article_tokens = tensor_compacted_seq_len - non_article_tokens
        tensor_article_tokens = float(max(0.0, tensor_article_tokens))

        # effective_article_tokens = mean actual tokens per head (excludes -inf padding)
        effective_article_tokens = compaction_stats.get('effective_article_tokens')
        if isinstance(effective_article_tokens, list):
            effective_article_tokens = (
                sum(float(x) for x in effective_article_tokens) / len(effective_article_tokens)
                if effective_article_tokens else 0.0
            )
        if effective_article_tokens is None:
            effective_article_tokens = tensor_article_tokens
        effective_article_tokens = float(max(0.0, effective_article_tokens))

        safe_effective_article = max(effective_article_tokens, 1e-8)
        safe_tensor_article = max(tensor_article_tokens, 1e-8)

        article_compaction_ratio = (
            len(article_indices) / safe_effective_article if len(article_indices) > 0 else 0.0
        )
        article_tensor_compaction_ratio = (
            len(article_indices) / safe_tensor_article if len(article_indices) > 0 else 0.0
        )

        # Show article compaction statistics (similar to example.py)
        print(f"\nArticle compaction statistics:")
        print(f"  Original article tokens:  {len(article_indices)}")
        print(f"  Effective compacted article tokens: {effective_article_tokens:.2f}")
        if effective_article_tokens != tensor_article_tokens:
            print(f"  Tensor compacted article tokens:   {tensor_article_tokens:.0f}")
        print(f"  Article compaction ratio: {article_compaction_ratio:.2f}x")

        # Show memory savings
        if original_cache_size is not None:
            memory_reduction = 100 * (1 - compacted_cache_size/original_cache_size)

            print(f"\nMemory statistics:")
            print(f"  Original cache:  {original_cache_size:,} elements ({seq_len} tokens)")
            print(f"  Compacted cache (tensor): {compacted_cache_size:,} elements ({tensor_compacted_seq_len} tokens)")
            print(f"  Memory reduction: {memory_reduction:.1f}%")
            if effective_memory_stats is not None:
                print(f"  Effective tokens: {effective_compacted_seq_len:.2f}")
                print(f"  Effective memory reduction: {effective_memory_stats['effective_memory_reduction_pct']:.1f}%")
            article_ratio_display = (
                article_len / safe_effective_article if article_len > 0 else float('nan')
            )
            print(f"  Article compaction ratio: {article_ratio_display:.2f}x")

        # Compile results
        results = {
            'article_id': article_data['article_id'],
            'article_title': article_data['title'],
            'article_idx': article_idx,
            'method': compaction_method.name(),
            'target_size_param': target_size,
            # Article compaction stats
            'original_article_tokens': article_len,
            'effective_article_tokens': effective_article_tokens,
            'tensor_article_tokens': tensor_article_tokens,
            'non_article_tokens': non_article_tokens,
            'article_compaction_ratio': article_compaction_ratio,
            'article_tensor_compaction_ratio': article_tensor_compaction_ratio,
            # Timing
            'extraction_time': extraction_time,
            'compaction_time': compaction_time,
            'query_generation_time': query_generation_time,
            'train_stats_time': train_stats_time,
            'test_stats_time': test_stats_time,
            'compaction_stats': compaction_stats,
            'qa_results': qa_results,
        }

        # Add perplexity if computed
        if avg_perplexity is not None:
            results['avg_perplexity'] = avg_perplexity
            results['avg_log_perplexity'] = avg_log_perplexity
            results['perplexities_per_question'] = perplexities_per_question

        # Add memory statistics to results (for aggregation)
        if original_cache_size is not None:
            memory_reduction = 100 * (1 - compacted_cache_size/original_cache_size)
            results['original_cache_size'] = original_cache_size
            results['compacted_cache_size'] = compacted_cache_size
            results['memory_reduction_pct'] = memory_reduction
            if effective_memory_stats is not None:
                results['effective_memory_stats'] = effective_memory_stats

        return results

    def _compute_detailed_stats(
        self,
        original_cache: Tuple,
        compacted_cache: Tuple,
        test_queries: torch.Tensor,
        eval_queries_per_kv_head: int,
        sliding_layer_indices: Optional[set] = None,
    ) -> Dict:
        """
        Compute detailed statistics comparing original and compacted cache.

        This computes per-layer-head metrics like cosine similarity,
        output MSE, etc., similar to what run_experiments does.

        Parameters
        ----------
        original_cache : tuple
            Original KV cache
        compacted_cache : tuple
            Compacted cache with (C1, beta, C2) per layer
        test_queries : torch.Tensor
            Test query vectors of shape (num_layers, num_heads, n_tokens, head_dim)
        eval_queries_per_kv_head : int
            Maximum number of queries to use per KV head for evaluation
        sliding_layer_indices : set, optional
            Set of layer indices that use sliding window attention (to skip)

        Returns
        -------
        stats : dict
            Detailed statistics per layer and head, with 'per_layer_head_test_stats'
            containing test metrics. Aggregation is done separately via compute_all_head_stats().
        """
        num_layers = len(original_cache)
        batch_size, num_kv_heads, seq_len, head_dim = original_cache[0][0].shape
        sliding_layer_indices = sliding_layer_indices or set()

        all_stats = {
            'per_layer_head_test_stats': {},
        }

        # Get number of attention heads from test_queries shape
        num_attention_heads = test_queries.shape[1]
        num_query_heads_per_kv_head = num_attention_heads // num_kv_heads

        for layer_idx in range(num_layers):
            # Skip sliding window layers (they have placeholder caches with zero length)
            if layer_idx in sliding_layer_indices:
                continue
            orig_keys = original_cache[layer_idx][0][0]  # (num_kv_heads, seq_len, head_dim)
            orig_values = original_cache[layer_idx][1][0]  # (num_kv_heads, seq_len, head_dim)

            comp_keys = compacted_cache[layer_idx][0][0]  # (num_kv_heads, t, head_dim)
            comp_beta = compacted_cache[layer_idx][1][0]  # (num_kv_heads, t)
            comp_values = compacted_cache[layer_idx][2][0]  # (num_kv_heads, t, head_dim)

            for kv_head_idx in range(num_kv_heads):
                # Extract for this KV head
                K = orig_keys[kv_head_idx]  # (seq_len, head_dim)
                V = orig_values[kv_head_idx]  # (seq_len, head_dim)
                C1 = comp_keys[kv_head_idx]  # (t, head_dim)
                beta = comp_beta[kv_head_idx]  # (t,)
                C2 = comp_values[kv_head_idx]  # (t, head_dim)

                # Get test queries for all query heads associated with this KV head
                # In GQA, multiple query heads share one KV head
                query_head_start = kv_head_idx * num_query_heads_per_kv_head
                query_head_end = query_head_start + num_query_heads_per_kv_head

                # Collect queries from all query heads for this KV head
                all_queries_for_kv_head = []
                for q_head_idx in range(query_head_start, query_head_end):
                    queries_q_head = test_queries[layer_idx, q_head_idx, :, :]  # (n_tokens, head_dim)
                    all_queries_for_kv_head.append(queries_q_head)

                # Concatenate all queries from query heads that use this KV head
                if all_queries_for_kv_head:
                    queries_kv_head = torch.cat(all_queries_for_kv_head, dim=0)  # (n_tokens * num_query_heads_per_kv_head, head_dim)
                else:
                    queries_kv_head = torch.empty((0, test_queries.shape[-1]), device=test_queries.device)

                # Subsample to eval_queries_per_kv_head queries per KV head
                n_queries = queries_kv_head.shape[0]
                if n_queries > eval_queries_per_kv_head:
                    subsample_indices = torch.randperm(n_queries, device=queries_kv_head.device)[:eval_queries_per_kv_head]
                    queries_subsample = queries_kv_head[subsample_indices]
                else:
                    queries_subsample = queries_kv_head

                # Evaluate using the existing function
                metrics = evaluate_compaction(K, V, C1, beta, C2, queries_subsample)

                key = f'L{layer_idx}H{kv_head_idx}'
                all_stats['per_layer_head_test_stats'][key] = {k: float(v) for k, v in metrics.items()}

        return all_stats

    def _compute_chunked_test_stats(
        self,
        original_chunk_caches: List[Tuple],
        compacted_chunk_caches: List[Tuple],
        test_queries: torch.Tensor,
        eval_queries_per_kv_head: int,
        chunk_stats: List[Dict]
    ) -> Dict:
        """
        Compute test stats for chunked compaction by concatenating chunk caches.

        For chunked compaction, we have per-chunk original and compacted caches.
        We concatenate them to form the full original and compacted article caches,
        then evaluate as normal.

        Parameters
        ----------
        original_chunk_caches : list of tuples
            Original (K, V) caches for the article portion of each chunk
        compacted_chunk_caches : list of tuples
            Compacted (C1, beta, C2) caches for the article portion of each chunk
        test_queries : torch.Tensor
            Test query vectors of shape (num_layers, num_heads, n_tokens, head_dim)
        eval_queries_per_kv_head : int
            Maximum queries per KV head
        chunk_stats : list of dict
            Stats for each chunk (used for debugging/logging)

        Returns
        -------
        stats : dict
            Same format as _compute_detailed_stats
        """
        if not original_chunk_caches or not compacted_chunk_caches:
            return {'per_layer_head_test_stats': {}}

        num_layers = len(original_chunk_caches[0])

        # Concatenate all chunk caches along the sequence dimension
        # Original format: ((K, V), ...) per layer, where K/V are (batch, heads, seq, dim)
        # Compacted format: ((C1, beta, C2), ...) per layer

        combined_original = []
        combined_compacted = []

        for layer_idx in range(num_layers):
            # Concatenate original caches for this layer
            orig_keys = [chunk[layer_idx][0] for chunk in original_chunk_caches]
            orig_vals = [chunk[layer_idx][1] for chunk in original_chunk_caches]
            K_combined = torch.cat(orig_keys, dim=2)  # concat along seq dim
            V_combined = torch.cat(orig_vals, dim=2)
            combined_original.append((K_combined, V_combined))

            # Concatenate compacted caches for this layer
            comp_keys = [chunk[layer_idx][0] for chunk in compacted_chunk_caches]
            comp_betas = [chunk[layer_idx][1] for chunk in compacted_chunk_caches]
            comp_vals = [chunk[layer_idx][2] for chunk in compacted_chunk_caches]
            C1_combined = torch.cat(comp_keys, dim=2)  # concat along seq dim
            beta_combined = torch.cat(comp_betas, dim=2)
            C2_combined = torch.cat(comp_vals, dim=2)
            combined_compacted.append((C1_combined, beta_combined, C2_combined))

        combined_original = tuple(combined_original)
        combined_compacted = tuple(combined_compacted)

        print(f"Concatenated {len(original_chunk_caches)} chunks: "
              f"original seq_len={combined_original[0][0].shape[2]}, "
              f"compacted seq_len={combined_compacted[0][0].shape[2]}")

        # Now use the standard detailed stats computation
        return self._compute_detailed_stats(
            combined_original,
            combined_compacted,
            test_queries,
            eval_queries_per_kv_head
        )

    def _compute_overall_stats(self, all_results: List[Dict]) -> Dict:
        """
        Compute overall aggregate statistics across all articles and methods.

        Parameters
        ----------
        all_results : list of dict
            Results from all evaluations

        Returns
        -------
        stats : dict
            Overall statistics grouped by method
        """
        from collections import defaultdict

        # Group results by method
        results_by_method = defaultdict(list)
        for result in all_results:
            method = result['method']
            results_by_method[method].append(result)

        overall_stats = {}

        for method, method_results in results_by_method.items():
            # Aggregate QA metrics across all articles
            total_questions = sum(r['qa_results'].get('total_questions', 0) for r in method_results)
            total_correct = sum(r['qa_results'].get('correct_answers', 0) for r in method_results)
            total_parseable = sum(r['qa_results'].get('parseable_answers', 0) for r in method_results)

            # Aggregate timing metrics
            total_extraction_time = sum(r['extraction_time'] for r in method_results)
            total_compaction_time = sum(r['compaction_time'] for r in method_results)
            total_query_generation_time = sum(r.get('query_generation_time', 0.0) for r in method_results)
            total_train_stats_time = sum(r.get('train_stats_time', 0.0) for r in method_results)
            total_test_stats_time = sum(r.get('test_stats_time', 0.0) for r in method_results)

            # Aggregate generation timing metrics
            total_generation_time = 0.0
            total_generated_tokens = 0
            for r in method_results:
                for q_result in r['qa_results']['results_per_question']:
                    if 'generation_time' in q_result:
                        total_generation_time += q_result['generation_time']
                    if 'num_generated_tokens' in q_result:
                        total_generated_tokens += q_result['num_generated_tokens']

            avg_time_per_token = total_generation_time / total_generated_tokens if total_generated_tokens > 0 else 0.0
            avg_tokens_per_second = total_generated_tokens / total_generation_time if total_generation_time > 0 else 0.0

            # Aggregate article compaction metrics
            article_compaction_stats = None
            if method != 'original':
                # Collect article compaction statistics
                article_original_tokens = []
                article_effective_tokens = []
                article_tensor_tokens = []
                article_compaction_ratios = []
                article_tensor_ratios = []
                non_article_tokens_list = []

                for r in method_results:
                    orig_tok = r.get('original_article_tokens')
                    if orig_tok is not None:
                        article_original_tokens.append(orig_tok)
                    eff_tok = r.get('effective_article_tokens')
                    if eff_tok is not None:
                        article_effective_tokens.append(eff_tok)
                    tensor_tok = r.get('tensor_article_tokens')
                    if tensor_tok is not None:
                        article_tensor_tokens.append(tensor_tok)
                    if 'article_compaction_ratio' in r:
                        article_compaction_ratios.append(r['article_compaction_ratio'])
                    if 'article_tensor_compaction_ratio' in r:
                        article_tensor_ratios.append(r['article_tensor_compaction_ratio'])
                    non_art = r.get('non_article_tokens')
                    if non_art is not None:
                        non_article_tokens_list.append(non_art)

                if article_original_tokens:
                    article_compaction_stats = {
                        'avg_original_article_tokens': sum(article_original_tokens) / len(article_original_tokens),
                        'avg_effective_article_tokens': sum(article_effective_tokens) / len(article_effective_tokens) if article_effective_tokens else 0.0,
                        'avg_article_compaction_ratio': sum(article_compaction_ratios) / len(article_compaction_ratios) if article_compaction_ratios else 0.0,
                    }
                    if article_tensor_tokens:
                        article_compaction_stats['avg_tensor_article_tokens'] = sum(article_tensor_tokens) / len(article_tensor_tokens)
                    if article_tensor_ratios:
                        article_compaction_stats['avg_tensor_article_compaction_ratio'] = sum(article_tensor_ratios) / len(article_tensor_ratios)
                    if non_article_tokens_list:
                        article_compaction_stats['avg_non_article_tokens'] = sum(non_article_tokens_list) / len(non_article_tokens_list)

            # Get target_size_param for overall stats
            avg_target_size_param = sum(r.get('target_size_param', 0.0) for r in method_results) / len(method_results)

            # Aggregate memory statistics
            memory_stats = None
            results_with_memory = [r for r in method_results if 'original_cache_size' in r]
            if results_with_memory:
                avg_original_cache_size = sum(r['original_cache_size'] for r in results_with_memory) / len(results_with_memory)
                avg_compacted_cache_size = sum(r['compacted_cache_size'] for r in results_with_memory) / len(results_with_memory)
                avg_memory_reduction_pct = sum(r['memory_reduction_pct'] for r in results_with_memory) / len(results_with_memory)
                memory_stats = {
                    'avg_original_cache_size': avg_original_cache_size,
                    'avg_compacted_cache_size': avg_compacted_cache_size,
                    'avg_memory_reduction_pct': avg_memory_reduction_pct,
                }
                results_with_effective = [r for r in results_with_memory if 'effective_memory_stats' in r]
                if results_with_effective:
                    avg_effective_size = sum(r['effective_memory_stats']['effective_compacted_cache_size'] for r in results_with_effective) / len(results_with_effective)
                    avg_effective_reduction = sum(r['effective_memory_stats']['effective_memory_reduction_pct'] for r in results_with_effective) / len(results_with_effective)
                    memory_stats['avg_effective_compacted_cache_size'] = avg_effective_size
                    memory_stats['avg_effective_memory_reduction_pct'] = avg_effective_reduction

            # Aggregate detailed train and test stats if available
            overall_all_head_train_stats = None
            overall_all_head_test_stats = None

            # Check for train stats in compaction_stats
            results_with_train_stats = [r for r in method_results
                                       if 'compaction_stats' in r and 'all_head_train_stats' in r['compaction_stats']]
            if results_with_train_stats:
                # Collect all train stats across all articles and compute means
                # We need to aggregate all the metrics from each article's all_head_train_stats
                aggregated_metrics = {}

                # Get all metric keys from first article
                first_stats = results_with_train_stats[0]['compaction_stats']['all_head_train_stats']
                metric_keys = [k for k in first_stats.keys() if k != 'eval_queries_per_kv_head']

                # For each metric, collect values across articles and compute mean
                for metric_key in metric_keys:
                    values = []
                    for r in results_with_train_stats:
                        stats = r['compaction_stats']['all_head_train_stats']
                        if metric_key in stats:
                            values.append(stats[metric_key])

                    if values:
                        aggregated_metrics[metric_key] = sum(values) / len(values)

                # Add metadata
                aggregated_metrics['num_articles_with_stats'] = len(results_with_train_stats)
                # Use the eval_queries_per_kv_head from the first article (should be same for all)
                aggregated_metrics['eval_queries_per_kv_head'] = first_stats.get('eval_queries_per_kv_head', None)

                overall_all_head_train_stats = aggregated_metrics

            # Check for test stats in compaction_stats
            results_with_test_stats = [r for r in method_results
                                      if 'compaction_stats' in r and 'all_head_test_stats' in r['compaction_stats']]
            if results_with_test_stats:
                # Collect all test stats across all articles and compute means
                aggregated_metrics = {}

                # Get all metric keys from first article
                first_stats = results_with_test_stats[0]['compaction_stats']['all_head_test_stats']
                metric_keys = [k for k in first_stats.keys() if k != 'eval_queries_per_kv_head']

                # For each metric, collect values across articles and compute mean
                for metric_key in metric_keys:
                    values = []
                    for r in results_with_test_stats:
                        stats = r['compaction_stats']['all_head_test_stats']
                        if metric_key in stats:
                            values.append(stats[metric_key])

                    if values:
                        aggregated_metrics[metric_key] = sum(values) / len(values)

                # Add metadata
                aggregated_metrics['num_articles_with_stats'] = len(results_with_test_stats)
                # Use the eval_queries_per_kv_head from the first article (should be same for all)
                aggregated_metrics['eval_queries_per_kv_head'] = first_stats.get('eval_queries_per_kv_head', None)

                overall_all_head_test_stats = aggregated_metrics

            # Aggregate perplexity metrics if available
            # Check for both old key (avg_perplexity_of_reference_answers) and new key (avg_perplexity)
            results_with_perplexity = [
                r for r in method_results
                if 'avg_perplexity' in r or 'avg_perplexity_of_reference_answers' in r
            ]
            avg_perplexity_across_articles = None
            avg_log_perplexity_across_articles = None
            if results_with_perplexity:
                perplexities = []
                log_perplexities = []
                for r in results_with_perplexity:
                    # Use new key if available, otherwise fall back to old key
                    if 'avg_perplexity' in r:
                        perplexities.append(r['avg_perplexity'])
                        log_perplexities.append(r['avg_log_perplexity'])
                    else:
                        perplexities.append(r['avg_perplexity_of_reference_answers'])
                        log_perplexities.append(r['avg_log_perplexity_of_reference_answers'])
                avg_perplexity_across_articles = sum(perplexities) / len(perplexities)
                avg_log_perplexity_across_articles = sum(log_perplexities) / len(log_perplexities)

            overall_stats[method] = {
                'num_articles': len(method_results),
                'total_questions': total_questions,
                'total_correct': total_correct,
                'total_parseable': total_parseable,
                'overall_accuracy': total_correct / total_questions if total_questions > 0 else 0.0,
                'overall_parse_rate': total_parseable / total_questions if total_questions > 0 else 0.0,
                'avg_target_size_param': avg_target_size_param,
                'total_extraction_time': total_extraction_time,
                'total_compaction_time': total_compaction_time,
                'total_query_generation_time': total_query_generation_time,
                'total_generation_time': total_generation_time,
                'total_train_stats_time': total_train_stats_time,
                'total_test_stats_time': total_test_stats_time,
                'avg_extraction_time_per_article': total_extraction_time / len(method_results),
                'avg_compaction_time_per_article': total_compaction_time / len(method_results),
                'avg_query_generation_time_per_article': total_query_generation_time / len(method_results),
                'avg_train_stats_time_per_article': total_train_stats_time / len(method_results),
                'avg_test_stats_time_per_article': total_test_stats_time / len(method_results),
                'avg_generation_time_per_question': total_generation_time / total_questions if total_questions > 0 else 0.0,
                'avg_time_per_token': avg_time_per_token,
                'avg_tokens_per_second': avg_tokens_per_second,
                'total_generated_tokens': total_generated_tokens,
            }

            # Add memory stats if available
            if memory_stats:
                overall_stats[method]['memory_stats'] = memory_stats

            # Add article compaction stats if available
            if article_compaction_stats:
                overall_stats[method]['article_compaction_stats'] = article_compaction_stats

            # Add perplexity if available
            if avg_perplexity_across_articles is not None:
                overall_stats[method]['overall_avg_perplexity'] = avg_perplexity_across_articles
                overall_stats[method]['overall_avg_log_perplexity'] = avg_log_perplexity_across_articles
                overall_stats[method]['num_articles_with_perplexity'] = len(results_with_perplexity)

            # Add train and test stats if available
            if overall_all_head_train_stats:
                overall_stats[method]['overall_all_head_train_stats'] = overall_all_head_train_stats
            if overall_all_head_test_stats:
                overall_stats[method]['overall_all_head_test_stats'] = overall_all_head_test_stats

            # Per-article breakdown
            overall_stats[method]['per_article'] = []
            for r in method_results:
                overall_stats[method]['per_article'].append({
                    'article_idx': r['article_idx'],
                    'article_title': r['article_title'],
                    'num_questions': r['qa_results'].get('total_questions', r['qa_results']['num_questions']),
                    'correct': r['qa_results'].get('correct_answers', 0),
                    'accuracy': r['qa_results'].get('accuracy', 0.0),
                    'article_compaction_ratio': r.get('article_compaction_ratio', 0.0),
                })

        return overall_stats

    def _log_overall_results(self, overall_stats: Dict, methods: List[str]):
        """
        Log overall results summary to console.

        Parameters
        ----------
        overall_stats : dict
            Overall statistics from _compute_overall_stats
        methods : list of str
            List of method names (may differ from actual method names in results)
        """
        print(f"\n{'='*80}")
        print(f"OVERALL RESULTS SUMMARY")
        print(f"{'='*80}")

        for method in overall_stats.keys():
            stats = overall_stats[method]
            print(f"\nMethod: {method.upper()}")
            print(f"{'-'*80}")
            print(f"  Articles evaluated: {stats['num_articles']}")
            print(f"  Total questions: {stats['total_questions']}")
            print(f"  Overall accuracy: {stats['overall_accuracy']:.2%} ({stats['total_correct']}/{stats['total_questions']})")
            print(f"  Overall parse rate: {stats['overall_parse_rate']:.2%} ({stats['total_parseable']}/{stats['total_questions']})")

            # Perplexity if available
            if 'overall_avg_perplexity' in stats:
                print(f"\n  Perplexity:")
                print(f"    Overall avg perplexity: {stats['overall_avg_perplexity']:.2f}")
                print(f"    Overall avg log perplexity: {stats['overall_avg_log_perplexity']:.4f}")
                print(f"    Articles with perplexity: {stats['num_articles_with_perplexity']}")

            # Article compaction statistics
            if 'article_compaction_stats' in stats:
                astats = stats['article_compaction_stats']
                print(f"\n  Article compaction statistics:")
                print(f"    Avg original article tokens:   {astats['avg_original_article_tokens']:.0f}")
                print(f"    Avg effective article tokens:  {astats['avg_effective_article_tokens']:.0f}")
                if 'avg_tensor_article_tokens' in astats and astats['avg_tensor_article_tokens'] != astats['avg_effective_article_tokens']:
                    print(f"    Avg tensor article tokens:     {astats['avg_tensor_article_tokens']:.0f}")
                print(f"    Avg article compaction ratio:  {astats['avg_article_compaction_ratio']:.2f}x")
                if 'avg_tensor_article_compaction_ratio' in astats:
                    print(f"    Avg tensor compaction ratio:   {astats['avg_tensor_article_compaction_ratio']:.2f}x")
                if 'avg_non_article_tokens' in astats:
                    print(f"    Avg non-article tokens:        {astats['avg_non_article_tokens']:.0f}")

            # Memory statistics
            if 'memory_stats' in stats:
                mstats = stats['memory_stats']
                print(f"\n  Memory Statistics:")
                print(f"    Original cache:  {mstats['avg_original_cache_size']:,.0f} elements")
                print(f"    Compacted cache: {mstats['avg_compacted_cache_size']:,.0f} elements")
                print(f"    Target size param:  {stats['avg_target_size_param']:.3f}")
                print(f"    Memory reduction: {mstats['avg_memory_reduction_pct']:.1f}%")
                if 'avg_effective_compacted_cache_size' in mstats:
                    print(f"    Effective compacted cache: {mstats['avg_effective_compacted_cache_size']:,.0f} elements")
                if 'avg_effective_memory_reduction_pct' in mstats:
                    print(f"    Effective memory reduction: {mstats['avg_effective_memory_reduction_pct']:.1f}%")
            print(f"\n  Timing:")
            print(f"    Total extraction time: {stats['total_extraction_time']:.2f}s")
            print(f"    Total query generation time: {stats['total_query_generation_time']:.2f}s")
            print(f"    Total compaction time: {stats['total_compaction_time']:.2f}s")
            print(f"    Total train stats time: {stats['total_train_stats_time']:.2f}s")
            print(f"    Total test stats time: {stats['total_test_stats_time']:.2f}s")
            print(f"    Avg extraction time/article: {stats['avg_extraction_time_per_article']:.2f}s")
            print(f"    Avg query generation time/article: {stats['avg_query_generation_time_per_article']:.2f}s")
            print(f"    Avg compaction time/article: {stats['avg_compaction_time_per_article']:.2f}s")
            print(f"    Avg train stats time/article: {stats['avg_train_stats_time_per_article']:.2f}s")
            print(f"    Avg test stats time/article: {stats['avg_test_stats_time_per_article']:.2f}s")
            print(f"\n  Generation:")
            print(f"    Total generation time: {stats['total_generation_time']:.2f}s")
            print(f"    Total generated tokens: {stats['total_generated_tokens']}")
            print(f"    Avg generation time/question: {stats['avg_generation_time_per_question']:.2f}s")
            if stats['avg_time_per_token'] > 0:
                print(f"    Avg time per token: {stats['avg_time_per_token']:.3f}s/token")
                print(f"    Avg tokens per second: {stats['avg_tokens_per_second']:.1f} tokens/s")
            else:
                print(f"    Avg time per token: N/A")
                print(f"    Avg tokens per second: N/A")

            # Train stats if available
            if 'overall_all_head_train_stats' in stats:
                tstats = stats['overall_all_head_train_stats']
                print(f"\n  Overall All-Head Train Stats (averaged over {tstats['num_articles_with_stats']} articles, "
                      f"eval queries per KV head: {tstats.get('eval_queries_per_kv_head', 'N/A')}):")
                print(f"    Mean of mean output MSE: {tstats.get('mean_mean_output_mse', 0):.6e}")
                print(f"    Mean of mean cosine sim: {tstats.get('mean_mean_output_cosine_sim', 0):.6f}")
                print(f"    Max of max output MSE: {tstats.get('max_max_output_mse', 0):.6e}")
                print(f"    Mean of max output MSE: {tstats.get('mean_max_output_mse', 0):.6e}")
                print(f"    RMS of RMS output MSE: {tstats.get('rms_rms_output_mse', 0):.6e}")
                print(f"    Mean of RMS output MSE: {tstats.get('mean_rms_output_mse', 0):.6e}")

            # Test stats if available
            if 'overall_all_head_test_stats' in stats:
                tstats = stats['overall_all_head_test_stats']
                print(f"\n  Overall All-Head Test Stats (averaged over {tstats['num_articles_with_stats']} articles, "
                      f"eval queries per KV head: {tstats.get('eval_queries_per_kv_head', 'N/A')}):")
                print(f"    Mean of mean output MSE: {tstats.get('mean_mean_output_mse', 0):.6e}")
                print(f"    Mean of mean cosine sim: {tstats.get('mean_mean_output_cosine_sim', 0):.6f}")
                print(f"    Max of max output MSE: {tstats.get('max_max_output_mse', 0):.6e}")
                print(f"    Mean of max output MSE: {tstats.get('mean_max_output_mse', 0):.6e}")
                print(f"    RMS of RMS output MSE: {tstats.get('rms_rms_output_mse', 0):.6e}")
                print(f"    Mean of RMS output MSE: {tstats.get('mean_rms_output_mse', 0):.6e}")


            # Per-article breakdown
            print(f"\n  Per-article breakdown:")
            print(f"    {'Idx':<6} {'Title':<40} {'Qs':<4} {'Correct':<8} {'Accuracy':<10} {'Ratio':<10}")
            print(f"    {'-'*81}")
            for article_stats in stats['per_article']:
                title_short = article_stats['article_title'][:37] + '...' if len(article_stats['article_title']) > 40 else article_stats['article_title']
                print(f"    {article_stats['article_idx']:<6} {title_short:<40} {article_stats['num_questions']:<4} "
                      f"{article_stats['correct']:<8} {article_stats['accuracy']:<10.2%} "
                      f"{article_stats['article_compaction_ratio']:<10.2f}")

        print(f"\n{'='*80}\n")

    def run_evaluation(
        self,
        dataset_name: str,
        compaction_methods: List[str],
        target_size: float,
        query_config: QueryConfig,
        n_articles: int = 1,
        start_article: int = 0,
        n_questions_per_article: Optional[int] = None,
        max_new_tokens: int = 2048,
        batch_size: Optional[int] = None,
        compute_stats: bool = False,
        verbose_logging: bool = False,
        compute_perplexity: bool = False,
        perplexity_only: bool = False,
        method_kwargs: Optional[Dict[str, Dict]] = None,
        log_dir: str = 'logs/qa_evaluation',
        experiment_name: Optional[str] = None,
        algorithm_config_file: Optional[str] = None,
        query_config_file: Optional[str] = None,
        ignore_article_indices: bool = False,
    ) -> Dict:
        """
        Run evaluation across multiple articles and methods.

        Parameters
        ----------
        dataset_name : str
            Name of dataset to evaluate ('quality', 'longhealth', etc.)
        compaction_methods : list of str
            List of compaction method names to evaluate
        target_size : float
            Target compacted sequence length. If between 0 and 1, treated as fraction of original size.
        query_config : QueryConfig
            Configuration for query generation
        n_articles : int
            Number of articles to evaluate. Use -1 for all articles.
        start_article : int
            Starting article index. Evaluates articles from start_article to start_article + n_articles.
        n_questions_per_article : int, optional
            Number of questions to use per article. If specified, uses a random shuffled set of n questions.
        max_new_tokens : int
            Maximum number of tokens to generate per question
        batch_size : int, optional
            If specified, processes questions in batches of this size for faster generation.
            Query extraction (for compute_stats) is supported in both batched and sequential modes.
        compute_stats : bool
            Whether to compute detailed statistics (uses original generation to extract query vectors)
        compute_perplexity : bool
            Whether to compute perplexity of original generation under compacted cache
        perplexity_only : bool
            Whether to only compute perplexity without generating new answers
        method_kwargs : dict, optional
            Keyword arguments for each method (keyed by method name)
        log_dir : str
            Directory to save logs
        experiment_name : str, optional
            Name for this experiment
        algorithm_config_file : str, optional
            Path to the algorithm config file used for hyperparameters
        query_config_file : str, optional
            Path to the query config file used for query generation
        ignore_article_indices : bool
            If True, ignore article boundaries and compact the entire sequence.
            target_size then refers to the whole sequence rather than just the article portion.

        Returns
        -------
        results : dict
            Full evaluation results
        """
        from compaction.compaction_methods import get_compaction_method

        # Load dataset
        dataset = load_dataset(dataset_name)

        # Determine which articles to use
        if n_articles == -1:
            # Use all articles from start_article to end
            article_indices = list(range(start_article, len(dataset)))
        else:
            # Use n_articles starting from start_article
            end_article = min(start_article + n_articles, len(dataset))
            article_indices = list(range(start_article, end_article))

        print(f"\n{'='*60}")
        print(f"QA EVALUATION")
        print(f"{'='*60}")
        print(f"Methods: {', '.join(compaction_methods)}")
        print(f"Articles: {len(article_indices)}")
        print(f"Target size: {target_size}")
        print(f"Ignore article indices: {ignore_article_indices}")
        print(f"Compute stats: {compute_stats}")
        print(f"Device: {self.device}")
        methods_str = ", ".join([f"{mc.method}({mc.fraction:.1%})" for mc in query_config.method_configs])
        print(f"Query config: methods=[{methods_str}], "
              f"max_query_vectors_per_kv_head={query_config.max_query_vectors_per_kv_head}")
        print(f"{'='*60}\n")

        # Initialize vLLM once if needed (will be reused across all articles)
        # vLLM is needed for:
        # 1. Self-study query generation
        # 2. Summarization compaction methods
        # 3. Original method generation (vLLM handles long contexts more efficiently)
        # 4. Reference answer generation for compute_stats/compute_perplexity
        need_vllm = False

        # Check if self_study query generation needs vLLM
        self_study_config = query_config.get_method_config('self_study')
        if self_study_config is not None:
            need_vllm = True

        # Check if compute_stats or compute_perplexity needs vLLM for reference answer generation
        if compute_stats or compute_perplexity:
            need_vllm = True

        # Check if any compaction method uses summarization or original
        for method_name in compaction_methods:
            if method_name == 'original':
                # Use vLLM for original method to handle long contexts efficiently
                need_vllm = True
                break
            kwargs = (method_kwargs or {}).get(method_name, {})
            base_algorithm = kwargs.get('algorithm', method_name)
            if base_algorithm == 'summarize':
                need_vllm = True
                break

        if need_vllm and self.vllm_model is None:
            self.vllm_model = initialize_vllm(self.model_name, max_model_len=self.max_model_len)

        # Detect if this is a perplexity-based dataset
        is_perplexity_eval = is_perplexity_dataset(dataset_name)
        if is_perplexity_eval:
            print(f"Dataset '{dataset_name}' uses perplexity-based evaluation (ground_truth instead of options)")

        all_results = []

        for method_name in compaction_methods:
            # Get method-specific kwargs
            kwargs = (method_kwargs or {}).get(method_name, {})
            compaction_method = get_compaction_method(method_name, kwargs)

            for article_idx in article_indices:
                article_data = dataset[article_idx]

                result = self.evaluate_compaction_on_article(
                    article_data=article_data,
                    compaction_method=compaction_method,
                    target_size=target_size,
                    query_config=query_config,
                    compute_stats=compute_stats,
                    verbose_logging=verbose_logging,
                    compute_perplexity=compute_perplexity,
                    perplexity_only=perplexity_only,
                    article_idx=article_idx,
                    max_new_tokens=max_new_tokens,
                    n_questions_per_article=n_questions_per_article,
                    batch_size=batch_size,
                    ignore_article_indices=ignore_article_indices,
                    is_perplexity_eval=is_perplexity_eval,
                )

                all_results.append(result)

        # Save results
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            base_filename = f"{experiment_name}_{timestamp}"
        else:
            base_filename = f"qa_evaluation_{timestamp}"

        # Avoid overwriting existing files by appending a suffix
        filepath = log_path / f"{base_filename}.json"
        suffix = 1
        while filepath.exists():
            filepath = log_path / f"{base_filename}_{suffix}.json"
            suffix += 1

        # Filter hyperparameters to only include methods we actually ran
        filtered_hyperparameters = {}
        if method_kwargs:
            for method_name in compaction_methods:
                if method_name in method_kwargs:
                    filtered_hyperparameters[method_name] = method_kwargs[method_name]

        # Serialize query_config, removing non-JSON-serializable fields
        query_config_dict = None
        if query_config:
            query_config_dict = asdict(query_config)
            # Remove extraction_fn from ConversationSpecs (callable, not JSON-serializable)
            for mc in query_config_dict['method_configs']:
                if mc['method'] == 'self_study' and 'config' in mc and 'conversation_specs' in mc['config']:
                    for spec in mc['config']['conversation_specs']:
                        spec.pop('extraction_fn', None)

        output = {
            'timestamp': timestamp,
            'experiment_name': experiment_name,
            'algorithm_config_file': algorithm_config_file,
            'query_config_file': query_config_file,
            'hyperparameters': filtered_hyperparameters,
            'config': {
                'model_name': self.model_name,
                'dataset_name': dataset_name,
                'methods': compaction_methods,
                'target_size': target_size,
                'ignore_article_indices': ignore_article_indices,
                'n_articles': len(article_indices),
                'start_article': start_article,
                'article_indices': article_indices,
                'n_questions_per_article': n_questions_per_article,
                'query_config': query_config_dict,
                'compute_stats': compute_stats,
                'compute_perplexity': compute_perplexity,
            },
            'results': all_results
        }

        # Compute overall aggregate statistics
        overall_stats = self._compute_overall_stats(all_results)
        output['overall_stats'] = overall_stats

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        # Log overall results
        self._log_overall_results(overall_stats, compaction_methods)

        print(f"\n{'='*60}")
        print(f"Evaluation complete! Results saved to:")
        print(f"{filepath}")
        if algorithm_config_file:
            print(f"Algorithm config file: {algorithm_config_file}")
        if query_config_file:
            print(f"Query config file: {query_config_file}")
        print(f"{'='*60}\n")

        return output
