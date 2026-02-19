# compaction/query_generation/self_study.py
# inspired by Cartridges: https://github.com/HazyResearch/cartridges/blob/05f629f519662122cf465c849e421334cc6a4d97/cartridges/synthesizers/self_study.py#L37
"""Self-study query generation for KV cache compaction."""

import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer

from evaluation.utils import detect_user_tags
from .config import SelfStudyConfig
from models.cache import clone_dynamic_cache, clone_compacted_prefix_cache


def _get_sampling_params(enable_thinking: bool, max_tokens: int):
    """
    Get sampling parameters based on thinking mode. 
    Using default Qwen settings for now; can probably get better diversity with higher temperature, especially for instance A.

    Parameters
    ----------
    enable_thinking : bool
        Whether thinking mode is enabled
    max_tokens : int
        Maximum tokens to generate

    Returns
    -------
    SamplingParams
        vLLM sampling parameters

    Note
    ----
    Sampling params are automatically set based on enable_thinking:
    - With thinking: temperature=0.6, top_p=0.95, top_k=20
    - Without thinking: temperature=0.7, top_p=0.8, top_k=20
    """
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("vLLM is required")

    if enable_thinking:
        return SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=max_tokens,
        )
    else:
        return SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=max_tokens,
        )


class SelfStudyQueryGenerator:
    """
    Generate queries for KV cache compaction using self-study.

    This class implements the self-study approach where:
    1. Model A generates a question/prompt given the context
    2. Model B generates an answer given the context + question
    3. Query vectors are extracted from both the question and answer

    Additionally supports prefill mode where the article is used directly
    instead of generating Model B's response.
    """

    def __init__(
        self,
        model,  # The model instance (e.g., Qwen3ForCausalLM)
        tokenizer: AutoTokenizer,
        config: SelfStudyConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
        vllm_model: Optional[Any] = None,
    ):
        """
        Initialize the self-study query generator.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            The language model to use for generation
        tokenizer : AutoTokenizer
            Tokenizer for the model
        config : SelfStudyConfig
            Configuration for self-study query generation
        device : str, optional
            Device to use. If None, uses model's device.
        dtype : torch.dtype, optional
            Data type for queries. If None, uses model's dtype.
        verbose : bool, optional
            Enable debug logging (default: False)
        vllm_model : optional
            Pre-initialized vLLM model. Required for self-study query generation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.verbose = verbose
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        # Use provided vLLM model (required for generation, not for query extraction)
        # Allow vllm_model=None when config is None (extraction-only) or when all
        # conversation specs use prefill_with_article (no vLLM generation needed)
        if config is not None and vllm_model is None:
            needs_vllm = any(
                not spec.is_prefill() or not spec.is_direct()
                for spec in config.conversation_specs
            )
            assert not needs_vllm, (
                "vllm_model must be provided when conversation specs require generation. "
                "Initialize vLLM once and pass it to all SelfStudyQueryGenerator instances. "
                "If all specs use prefill_with_article=True, vLLM is not needed."
            )
        self.vllm_model = vllm_model
        if self.verbose and vllm_model is not None:
            print("Using provided vLLM model instance")

        # Detect if model doesn't use </think> tags (Llama, Gemma, Qwen3-4B-Instruct, etc.)
        model_name = self.model.config._name_or_path.lower()
        self._no_think_tags = (
            "llama" in model_name or
            "gemma" in model_name or
            "qwen3-4b-instruct" in model_name
        )

    def _extract_article_from_formatted_context(self, formatted_context: str) -> str:
        """
        Extract the article content from formatted context.

        Supports both Qwen (<|im_start|>...<|im_end|>) and
        Llama (<|start_header_id|>...<|eot_id|>) chat templates.

        This method extracts just the {article} portion.

        Parameters
        ----------
        formatted_context : str
            The formatted context string with chat template applied

        Returns
        -------
        str
            The extracted article text
        """
        user_start_tag, user_end_tag = detect_user_tags(formatted_context)

        user_start_pos = formatted_context.find(user_start_tag)
        if user_start_pos == -1:
            raise ValueError(f"Could not find '{user_start_tag}' tag in formatted context")

        # The article starts after the user tag and a newline
        article_text_start = formatted_context.find('\n', user_start_pos + len(user_start_tag))
        if article_text_start == -1:
            raise ValueError("Could not find newline after user start tag")
        article_text_start += 1  # Skip the newline itself

        # Find the end tag after the article (the user section is the second occurrence)
        article_text_end = formatted_context.find(user_end_tag, article_text_start)
        if article_text_end == -1:
            raise ValueError(f"Could not find '{user_end_tag}' tag after article content")

        # Extract the article text
        article = formatted_context[article_text_start:article_text_end]
        return article

    def generate_queries(
        self,
        n_queries_per_attention_head: int,
        formatted_context: str,
        past_key_values: Optional[Tuple] = None,
        return_sequences: bool = False,
        indices: Optional[range] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[list]]:
        """
        Generate self-study queries using vLLM for generation and HuggingFace for query extraction.

        This is a two-phase approach:
        1. Use vLLM to batch-generate all questions and answers (fast)
        2. Use HuggingFace with hooks to extract query vectors in prefill mode (parallel)

        Parameters
        ----------
        n_queries_per_attention_head : int
            Number of queries to generate per attention head
        formatted_context : str
            The formatted context string
        past_key_values : tuple, optional
            Pre-computed KV cache for the context (used for query extraction)
        return_sequences : bool, optional
            If True, return the full text sequences for later re-extraction (default: False)

        Returns
        -------
        queries : torch.Tensor
            Shape: (num_layers, num_heads, n_tokens_total, head_dim)
        stats : dict
            Query generation statistics
        sequences : list or None
            If return_sequences=True, list of dicts with:
                - 'full_text': formatted_context + answer_prompt + answer
                - 'starter': conversation starter
                - 'answer': model B's answer
                - 'enable_thinking_b': whether thinking was enabled for model B
                - 'n_context_tokens': number of tokens in formatted_context
            Otherwise None
        """
        stats = {
            'n_conversations': 0,
            'n_self_study_tokens_extracted': 0,
            'conversation_starters_used': [],
        }
        n_queries = n_queries_per_attention_head
        # Get conversation specs from config
        specs = self.config.conversation_specs

        # Wake up vLLM before use (only needed for non-prefill specs)
        if self.vllm_model is not None:
            if self.verbose:
                print("Waking up vLLM for generation...")

            # Clear CUDA cache before waking up vLLM to maximize available memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Print memory stats
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                if self.verbose:
                    print(f"GPU Memory before vLLM wake: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
                    print(f"GPU Memory available: {total - allocated:.2f}GB")

            self.vllm_model.wake_up()

        try:
            # Get head_dim from model config
            head_dim = getattr(self.model.config, 'head_dim',
                               self.model.config.hidden_size // self.model.config.num_attention_heads)

            if self.verbose:
                print(f"Generating conversations from {len(specs)} ConversationSpecs with vLLM...")

            # Phase 1: Batch-generate questions with vLLM (Model A) for seed-based specs
            # Collect specs that need Model A generation
            specs_needing_a = [(i, spec) for i, spec in enumerate(specs) if not spec.is_direct()]

            # Track which spec index maps to which conversation starters
            spec_to_starters = {}  # {spec_index: [conversation_starter1, conversation_starter2, ...]}

            if specs_needing_a:
                # Group specs by (enable_thinking_a, max_tokens_a) to batch efficiently
                # param_groups: {(enable_thinking, max_tokens): [(spec_idx, spec), ...]}
                param_groups = {}
                for spec_idx, spec in specs_needing_a:
                    if spec.enable_thinking_a is None or spec.max_tokens_a is None:
                        raise ValueError(
                            f"ConversationSpec at index {spec_idx} must explicitly set "
                            f"enable_thinking_a and max_tokens_a"
                        )
                    enable_thinking = spec.enable_thinking_a
                    max_tokens = spec.max_tokens_a
                    group_key = (enable_thinking, max_tokens)
                    if group_key not in param_groups:
                        param_groups[group_key] = []
                    param_groups[group_key].append((spec_idx, spec))

                # Process each parameter group separately (to batch prompts with same settings)
                for (enable_thinking, max_tokens), group_specs in param_groups.items():
                    question_prompts = []
                    question_spec_indices = []

                    for spec_idx, spec in group_specs:
                        question_messages = [{"role": "user", "content": spec.seed_prompt}]
                        question_prompt = self.tokenizer.apply_chat_template(
                            question_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking
                        )
                        # Strip <bos> for Gemma models (it's already in formatted_context)
                        if question_prompt.startswith("<bos>"):
                            question_prompt = question_prompt[len("<bos>"):]
                        full_prompt_a = formatted_context + question_prompt
                        question_prompts.append(full_prompt_a)
                        question_spec_indices.append(spec_idx)

                    # Batch generate all questions in this group
                    sampling_params_a = _get_sampling_params(enable_thinking, max_tokens)
                    question_outputs = self.vllm_model.generate(question_prompts, sampling_params_a)

                    # Process outputs and apply extraction if needed
                    # Track specs that need retry (extraction failed)
                    specs_to_retry = []  # [(spec_idx, spec, attempt_count), ...]

                    for spec_idx, output in zip(question_spec_indices, question_outputs):
                        generated_text = output.outputs[0].text
                        spec = specs[spec_idx]

                        if spec.uses_extraction():
                            # Apply extraction function to get multiple starters
                            starters = spec.extraction_fn(generated_text)
                            if not starters:
                                # Extraction failed - queue for retry
                                specs_to_retry.append((spec_idx, spec, 1))
                                extraction_fn_name = getattr(spec.extraction_fn, '__name__', str(spec.extraction_fn))
                                print(f"[SelfStudy] Warning: Extraction failed for spec {spec_idx} "
                                      f"(extraction_fn={extraction_fn_name}), attempt 1/3. "
                                      f"Generated text preview: {generated_text[:200]}...")
                        else:
                            # Use entire output as single starter
                            starters = [generated_text]

                        spec_to_starters[spec_idx] = starters

                    # Retry failed extractions up to 3 times
                    while specs_to_retry:
                        retry_prompts = []
                        retry_info = []  # [(spec_idx, spec, attempt_count), ...]

                        for spec_idx, spec, attempt in specs_to_retry:
                            if attempt >= 3:
                                # Max retries reached, log and skip
                                extraction_fn_name = getattr(spec.extraction_fn, '__name__', str(spec.extraction_fn))
                                print(f"[SelfStudy] Error: Extraction failed for spec {spec_idx} "
                                      f"(extraction_fn={extraction_fn_name}) after 3 attempts. Skipping.")
                                continue

                            # Build prompt for retry
                            question_messages = [{"role": "user", "content": spec.seed_prompt}]
                            question_prompt = self.tokenizer.apply_chat_template(
                                question_messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=enable_thinking
                            )
                            # Strip <bos> for Gemma models (it's already in formatted_context)
                            if question_prompt.startswith("<bos>"):
                                question_prompt = question_prompt[len("<bos>"):]
                            full_prompt_a = formatted_context + question_prompt
                            retry_prompts.append(full_prompt_a)
                            retry_info.append((spec_idx, spec, attempt))

                        if not retry_prompts:
                            break

                        # Batch generate retries
                        retry_outputs = self.vllm_model.generate(retry_prompts, sampling_params_a)

                        # Clear retry list and process results
                        specs_to_retry = []
                        for (spec_idx, spec, attempt), output in zip(retry_info, retry_outputs):
                            generated_text = output.outputs[0].text
                            starters = spec.extraction_fn(generated_text)

                            if starters:
                                # Success!
                                spec_to_starters[spec_idx] = starters
                                if self.verbose:
                                    print(f"[SelfStudy] Retry succeeded for spec {spec_idx} on attempt {attempt + 1}")
                            else:
                                # Still failed, queue for another retry
                                specs_to_retry.append((spec_idx, spec, attempt + 1))
                                extraction_fn_name = getattr(spec.extraction_fn, '__name__', str(spec.extraction_fn))
                                print(f"[SelfStudy] Warning: Extraction failed for spec {spec_idx} "
                                      f"(extraction_fn={extraction_fn_name}), attempt {attempt + 1}/3. "
                                      f"Generated text preview: {generated_text[:200]}...")

                if self.verbose:
                    total_starters = sum(len(starters) for starters in spec_to_starters.values())
                    print(f"Generated {total_starters} conversation starters from {len(specs_needing_a)} seed prompts with vLLM")

            # Add direct conversation starters
            for i, spec in enumerate(specs):
                if spec.is_direct():
                    spec_to_starters[i] = [spec.conversation_starter]

            # Phase 2: Batch-generate answers with vLLM (Model B)
            # Extract article for prefill specs
            article = None
            has_prefill_specs = any(spec.is_prefill() for spec in specs)
            if has_prefill_specs:
                # If indices is provided, extract the text at those token positions
                # Otherwise, fall back to detecting article boundaries from formatted_context
                if indices is not None:
                    # Decode the tokens at the specified indices
                    context_tokens = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False).input_ids[0]
                    article_tokens = context_tokens[indices.start:indices.stop]
                    article = self.tokenizer.decode(article_tokens, skip_special_tokens=False)
                else:
                    article = self._extract_article_from_formatted_context(formatted_context)
                if self.verbose:
                    print(f"Extracted article for prefill ({len(article)} chars)")

            # Group conversation starters by their Model B parameters (enable_thinking_b, max_tokens_b)
            # First, flatten all conversation starters and track their spec
            all_starters = []
            starter_to_spec_idx = []  # Track which spec each starter came from
            for spec_idx in sorted(spec_to_starters.keys()):
                for starter in spec_to_starters[spec_idx]:
                    all_starters.append(starter)
                    starter_to_spec_idx.append(spec_idx)

            # Group starters by Model B parameters
            # Separate prefill specs from generation specs
            prefill_starters = []  # [(starter, spec_idx), ...]
            generation_starters = []  # [(starter, spec_idx), ...]

            for starter, spec_idx in zip(all_starters, starter_to_spec_idx):
                spec = specs[spec_idx]
                if spec.is_prefill():
                    prefill_starters.append((starter, spec_idx))
                else:
                    generation_starters.append((starter, spec_idx))

            # We need to maintain order, so we'll store results in a dict
            starter_to_answer = {}  # {(starter, spec_idx): answer}

            # Handle prefill starters (bypass vLLM generation)
            if prefill_starters:
                for starter, spec_idx in prefill_starters:
                    starter_to_answer[(starter, spec_idx)] = article
                if self.verbose:
                    print(f"Prefilled {len(prefill_starters)} answers with article content (bypassing vLLM)")

            # Handle generation starters (use vLLM)
            if generation_starters:
                # param_groups_b: {(enable_thinking_b, max_tokens_b): [(starter, spec_idx), ...]}
                param_groups_b = {}
                for starter, spec_idx in generation_starters:
                    spec = specs[spec_idx]
                    # Use defaults if not set
                    enable_thinking_b = spec.enable_thinking_b
                    max_tokens_b = spec.max_tokens_b
                    group_key = (enable_thinking_b, max_tokens_b)
                    if group_key not in param_groups_b:
                        param_groups_b[group_key] = []
                    param_groups_b[group_key].append((starter, spec_idx))

                # Generate answers for each parameter group
                for (enable_thinking_b, max_tokens_b), group_starters in param_groups_b.items():
                    answer_prompts = []
                    for starter, _ in group_starters:
                        answer_messages = [{"role": "user", "content": starter}]
                        answer_prompt = self.tokenizer.apply_chat_template(
                            answer_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking_b
                        )
                        # Strip <bos> for Gemma models (it's already in formatted_context)
                        if answer_prompt.startswith("<bos>"):
                            answer_prompt = answer_prompt[len("<bos>"):]
                        full_prompt_b = formatted_context + answer_prompt
                        answer_prompts.append(full_prompt_b)

                    # Batch generate all answers in this group
                    sampling_params_b = _get_sampling_params(enable_thinking_b, max_tokens_b)
                    answer_outputs = self.vllm_model.generate(answer_prompts, sampling_params_b)

                    # Store answers
                    for (starter, spec_idx), output in zip(group_starters, answer_outputs):
                        starter_to_answer[(starter, spec_idx)] = output.outputs[0].text

                if self.verbose:
                    print(f"Generated {len(generation_starters)} answers with vLLM across {len(param_groups_b)} parameter groups")

            # Phase 3: Extract query vectors from answers using HuggingFace in prefill mode
            all_query_vectors = []
            sequences = [] if return_sequences else None

            # Pre-compute context tokens once (needed for all conversations)
            # Use add_special_tokens=False since formatted_context already has <bos> from chat template
            context_inputs = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False).to(self.device)
            n_context_tokens = context_inputs.input_ids.shape[1]

            total_conversations = len(all_starters)
            for idx, (starter, spec_idx) in enumerate(zip(all_starters, starter_to_spec_idx), 1):
                spec = specs[spec_idx]
                answer = starter_to_answer[(starter, spec_idx)]

                # Get the enable_thinking_b for this spec
                # For prefill specs, enable_thinking_b can be None
                if spec.is_prefill():
                    enable_thinking_b = False  # Prefill doesn't use thinking
                else:
                    # Use default if not set (enable_thinking_b=False)
                    enable_thinking_b = spec.enable_thinking_b

                if self.verbose:
                    print(f"  [Conv {idx}/{total_conversations}] Starter: {starter}")
                    print(f"  [Conv {idx}/{total_conversations}] Answer: {answer[:100]}...")

                # Check if thinking is enabled for Model B and </think> is missing
                # Skip this check for models that don't use </think> tags (Llama, Gemma, Qwen3-4B-Instruct, etc.)
                if enable_thinking_b and "</think>" not in answer and not self._no_think_tags:
                    if self.verbose:
                        print(f"  [Conv {idx}/{total_conversations}] Skipping: Model B thinking response missing </think> tag")
                    continue

                if self.verbose:
                    print(f"  [Conv {idx}/{total_conversations}] Extracting queries from answer...")

                # Tokenize the answer prompt (conversation starter)
                answer_messages = [{"role": "user", "content": starter}]
                answer_prompt = self.tokenizer.apply_chat_template(
                    answer_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking_b
                )
                # Strip <bos> for Gemma models (it's already in formatted_context)
                if answer_prompt.startswith("<bos>"):
                    answer_prompt = answer_prompt[len("<bos>"):]

                # Tokenize context + answer prompt + generated answer
                # This ensures queries attend over the same KV cache we plan to compact
                full_answer_text = formatted_context + answer_prompt + answer
                answer_inputs = self.tokenizer(full_answer_text, return_tensors="pt", add_special_tokens=False).to(self.device)

                # When past_key_values already encode the context, avoid re-processing the
                # context prefix so cache positions stay aligned with actual inference.
                extraction_input_ids = answer_inputs.input_ids
                extraction_start_token_idx = n_context_tokens
                if past_key_values is not None:
                    if extraction_input_ids.shape[1] < n_context_tokens:
                        raise ValueError(
                            "Context tokens longer than provided input_ids; cannot align with past_key_values."
                        )
                    extraction_input_ids = extraction_input_ids[:, n_context_tokens:]
                    extraction_start_token_idx = 0

                # Extract query vectors using prefill (single forward pass)
                # We want: all queries from answer_prompt + answer (excluding context)
                query_vectors = self._extract_query_vectors_from_prefill(
                    input_ids=extraction_input_ids,
                    past_key_values=past_key_values,
                    head_dim=head_dim,
                    start_token_idx=extraction_start_token_idx,  # Exclude context tokens
                )

                if query_vectors is not None:
                    n_tokens = query_vectors.shape[2]
                    if self.verbose:
                        print(f"  [Conv {idx}/{total_conversations}] Extracted {n_tokens} tokens")
                    # Accumulate on CPU to avoid GPU OOM, will transfer subsampled result back
                    all_query_vectors.append(query_vectors.cpu())

                    # Store sequence info if requested
                    if return_sequences:
                        # Store the actual token IDs for exact re-extraction
                        # Extract the suffix: tokens from n_context_tokens onwards (excluding context)
                        suffix_token_ids = answer_inputs.input_ids[:, n_context_tokens:]  # (1, suffix_len)

                        sequences.append({
                            'full_text': full_answer_text,
                            'starter': starter,
                            'answer': answer,
                            'answer_prompt': answer_prompt,
                            'enable_thinking_b': enable_thinking_b,
                            'n_context_tokens': n_context_tokens,
                            'suffix_token_ids': suffix_token_ids,  # Store token IDs for re-extraction
                        })
                else:
                    if self.verbose:
                        print(f"  [Conv {idx}/{total_conversations}] Failed to extract query vectors")

                # Track conversation starter in stats
                stats['conversation_starters_used'].append(starter)
                stats['n_conversations'] += 1

            # Check if we have any query vectors
            if not all_query_vectors:
                raise RuntimeError(
                    "Self-study query generation failed: no query vectors extracted. "
                    "This may indicate OOM during extraction. "
                    "Try reducing batch size, max_tokens_a/max_tokens_b, or using fewer conversation specs."
                )

            # Concatenate and subsample queries (same as transformers path)
            concatenated_queries = torch.cat(all_query_vectors, dim=2)
            n_extracted = concatenated_queries.shape[2]

            # Subsample on CPU, then transfer only what we need to GPU
            if n_extracted > n_queries:
                indices = torch.randperm(n_extracted)[:n_queries]
                indices = indices.sort()[0]
                final_queries = concatenated_queries[:, :, indices, :].to(self.device)
                stats['n_self_study_tokens_extracted'] = n_extracted
                stats['n_self_study_tokens_subsampled'] = n_queries
                # Only save subsample_indices when return_sequences=True (for on-policy methods)
                if return_sequences:
                    stats["subsample_indices"] = indices.tolist()
                if self.verbose:
                    print(f"  Subsampled {n_extracted} → {n_queries} tokens per attention head")
            else:
                final_queries = concatenated_queries.to(self.device)
                stats['n_self_study_tokens_extracted'] = n_extracted
                if self.verbose:
                    print(f"  Using all {n_extracted} extracted tokens per attention head (requested {n_queries})")

            return final_queries, stats, sequences
        finally:
            # Put vLLM back to sleep after use to free GPU memory
            if self.vllm_model is not None:
                if self.verbose:
                    print("Putting vLLM to sleep...")
                self.vllm_model.sleep()

    def _extract_query_vectors_from_prefill(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple],
        head_dim: int,
        start_token_idx: Optional[int] = None,
        max_layer: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Extract query vectors from pre-generated tokens using a single prefill pass.

        This is much faster than token-by-token extraction during generation,
        as it processes all tokens in a single forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs for the entire sequence (context + answer prompt + generated answer)
            Shape: (batch_size, seq_len)
        past_key_values : tuple, optional
            Pre-computed KV cache for the context. Can be either:
            - Standard KV cache: ((keys, values), ...) where keys/values are (batch, heads, seq, dim)
            - CompactedPrefixCache: for on-policy query extraction (auto-detected)
        head_dim : int
            Dimension of each attention head
        start_token_idx : int, optional
            If provided, only extract query vectors from this token index onwards.
            This is useful when input_ids contains context + answer, but we only want
            queries from answer (and last context token).
        max_layer : int, optional
            If provided, only extract queries up to and including this layer (0-indexed).
            This allows early stopping for efficiency when we don't need all layers.

        Returns
        -------
        query_vectors : torch.Tensor or None
            Query vectors of shape (num_layers, num_attention_heads, n_tokens, head_dim)
            where n_tokens is determined by start_token_idx (if provided) or full sequence length
        """
        try:
            num_layers = self.model.config.num_hidden_layers
            num_attention_heads = self.model.config.num_attention_heads

            # Determine how many layers we actually need
            layers_to_process = num_layers if max_layer is None else (max_layer + 1)

            # Storage for queries from all layers (dict keyed by layer_idx)
            # Each layer maps to a list of query tensors (one per chunk when using chunked_prefill)
            all_layer_queries = {layer_idx: [] for layer_idx in range(layers_to_process)}

            # Hook function to capture queries at each layer for ALL tokens
            def make_hook_all_tokens(layer_idx):
                def hook_fn(module, args, kwargs):
                    # Extract hidden states
                    hidden_states = None
                    if len(args) > 0:
                        hidden_states = args[0]
                    elif 'hidden_states' in kwargs:
                        hidden_states = kwargs['hidden_states']

                    if hidden_states is not None:
                        # Get position_embeddings from kwargs (passed by the model)
                        position_embeddings = kwargs.get('position_embeddings')

                        # Apply query projection: Q = hidden_states @ W_q
                        q = module.q_proj(hidden_states)  # (batch, seq_len, num_heads * head_dim)

                        batch_size, seq_len, _ = q.shape
                        # Reshape to (batch, seq_len, num_attention_heads, head_dim)
                        q = q.view(batch_size, seq_len, num_attention_heads, head_dim)

                        # Apply q_norm
                        from models.qwen3.modeling_qwen3 import Qwen3Attention
                        from models.gemma3.modeling_gemma3 import Gemma3Attention
                        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
                            q = module.q_norm(q)

                        # Transpose to (batch, num_heads, seq_len, head_dim)
                        q = q.transpose(1, 2)

                        # Apply RoPE to queries if we have position embeddings
                        if position_embeddings is not None:
                            from models.qwen3.modeling_qwen3 import rotate_half
                            cos, sin = position_embeddings
                            cos = cos.unsqueeze(1)  # Add head dimension
                            sin = sin.unsqueeze(1)
                            q = (q * cos) + (rotate_half(q) * sin)

                        # Transpose back to (batch, seq_len, num_heads, head_dim)
                        q = q.transpose(1, 2)

                        # Store queries from ALL tokens in the sequence
                        # q: (batch, seq_len, num_attention_heads, head_dim)
                        # Append to the list for this layer (supports multiple chunks)
                        all_layer_queries[layer_idx].append(q[0])  # (seq_len, num_attention_heads, head_dim)

                return hook_fn

            # Register hooks only on the layers we need
            hooks = []
            try:
                for layer_idx in range(layers_to_process):
                    target_layer = self.model.model.layers[layer_idx].self_attn
                    handle = target_layer.register_forward_pre_hook(
                        make_hook_all_tokens(layer_idx),
                        with_kwargs=True
                    )
                    hooks.append(handle)

                # Forward pass for ALL tokens (or up to max_layer)
                # Use chunked_prefill for memory-efficient processing
                with torch.no_grad():
                    # Make a copy of past_key_values to avoid mutation (forward pass mutates it)
                    past_kv_copy = None
                    if past_key_values is not None:
                        from models.cache import CompactedPrefixCache
                        from transformers.cache_utils import DynamicCache

                        if isinstance(past_key_values, CompactedPrefixCache):
                            past_kv_copy = clone_compacted_prefix_cache(past_key_values)
                        elif isinstance(past_key_values, DynamicCache):
                            past_kv_copy = clone_dynamic_cache(past_key_values)
                        else:
                            # For unknown cache types, raise an error rather than risking mutation
                            raise TypeError(
                                f"Unsupported cache type: {type(past_key_values)}. "
                                f"Expected CompactedPrefixCache or DynamicCache."
                            )

                    # Use chunked_prefill for memory-efficient processing
                    # This handles both CompactedPrefixCache and DynamicCache
                    from models.generate import chunked_prefill
                    kv_seq_len = past_kv_copy.get_seq_length()
                    if kv_seq_len > 20000:
                        chunk_size = 256
                    elif kv_seq_len > 10000:
                        chunk_size = 1024
                    else:
                        chunk_size = 4096
                    chunked_prefill(
                        self.model,
                        input_ids=input_ids,
                        past_key_values=past_kv_copy,
                        max_layer=max_layer,
                        chunk_size=chunk_size,
                    )

                # Process the captured queries
                # all_layer_queries: dict mapping layer_idx -> list of (seq_len, num_attention_heads, head_dim)
                # Each layer may have multiple tensors if chunked_prefill processed in multiple chunks
                if all_layer_queries and any(len(v) > 0 for v in all_layer_queries.values()):
                    # Concatenate queries per layer (along seq_len dimension), then stack layers
                    layer_queries = []
                    for layer_idx in range(layers_to_process):
                        layer_query_list = all_layer_queries[layer_idx]
                        if len(layer_query_list) == 1:
                            # Single chunk - no concatenation needed
                            layer_queries.append(layer_query_list[0])
                        else:
                            # Multiple chunks - concatenate along seq_len (dim=0)
                            layer_queries.append(torch.cat(layer_query_list, dim=0))

                    # Stack to: (num_layers, seq_len, num_attention_heads, head_dim)
                    queries = torch.stack(layer_queries, dim=0)
                    # Transpose to: (num_layers, num_attention_heads, seq_len, head_dim)
                    queries = queries.permute(0, 2, 1, 3)

                    # If start_token_idx is specified, only return queries from that index onwards
                    if start_token_idx is not None:
                        queries = queries[:, :, start_token_idx:, :]

                    return queries
                else:
                    return None

            finally:
                # Always remove hooks, even if an error occurred
                for handle in hooks:
                    handle.remove()

        except Exception as e:
            print(f"Warning: Failed to extract query vectors in prefill mode: {e}")
            import traceback
            traceback.print_exc()
            return None
