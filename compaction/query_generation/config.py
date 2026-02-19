# compaction/query_generation/config.py
"""Configuration for query generation strategies."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable


@dataclass
class ConversationSpec:
    """
    Specification for one or more conversations in self-study.

    A ConversationSpec can represent:
    1. A seed prompt that goes through Model A to generate a conversation starter
    2. A direct conversation starter that bypasses Model A
    3. A seed prompt with extraction logic to create multiple conversation starters
    4. A prefill mode that bypasses Model B and prefills with the article content

    Exactly one of (seed_prompt, conversation_starter) must be provided.

    Parameters
    ----------
    seed_prompt : str, optional
        Prompt to give Model A to generate a conversation starter.
        If provided, Model A will generate text, which may be used directly
        or processed by extraction_fn.
    conversation_starter : str, optional
        Direct conversation starter for Model B, bypassing Model A entirely.
    extraction_fn : Callable[[str], List[str]], optional
        Function to extract multiple conversation starters from Model A's output.
        If None and seed_prompt is provided, Model A's entire output is used as
        a single conversation starter.
        Example: lambda text: text.split("\n\n") to split on double newlines.
    prefill_with_article : bool, optional
        If True, bypass Model B generation and prefill with the article content
        extracted from formatted_context. This saves time by avoiding generation
        and directly using the article as the "answer".
        (default: False)
    enable_thinking_a : bool, optional
        Enable thinking mode for Model A when generating from this seed prompt.
        Only applies when seed_prompt is provided. Must be explicitly set for
        specs with seed_prompt.
        (default: None)
    max_tokens_a : int, optional
        Max tokens for Model A when generating from this seed prompt.
        Only applies when seed_prompt is provided. Must be explicitly set for
        specs with seed_prompt.
        (default: None)
    enable_thinking_b : bool, optional
        Enable thinking mode for Model B when answering this conversation.
        Ignored if prefill_with_article is True.
        (default: False)
    max_tokens_b : int, optional
        Max tokens for Model B when answering this conversation.
        Ignored if prefill_with_article is True.
        (default: 4096)
    """
    seed_prompt: Optional[str] = None
    conversation_starter: Optional[str] = None
    extraction_fn: Optional[Callable[[str], List[str]]] = None
    prefill_with_article: bool = False
    enable_thinking_a: Optional[bool] = None
    max_tokens_a: Optional[int] = None
    enable_thinking_b: Optional[bool] = False
    max_tokens_b: Optional[int] = 4096

    def __post_init__(self):
        """Validate that exactly one of seed_prompt or conversation_starter is provided."""
        # Check mutually exclusive modes
        modes_set = sum([
            self.seed_prompt is not None,
            self.conversation_starter is not None,
        ])

        if modes_set == 0:
            raise ValueError(
                "ConversationSpec: exactly one of 'seed_prompt' or 'conversation_starter' must be provided"
            )
        if modes_set > 1:
            raise ValueError(
                "ConversationSpec: only one of 'seed_prompt' or 'conversation_starter' can be provided"
            )

        # Original validations for conversation_starter
        if self.conversation_starter is not None and self.extraction_fn is not None:
            raise ValueError(
                "ConversationSpec: extraction_fn cannot be used with direct conversation_starter"
            )
        if self.conversation_starter is not None and self.enable_thinking_a is not None:
            raise ValueError(
                "ConversationSpec: enable_thinking_a cannot be used with direct conversation_starter"
            )
        if self.conversation_starter is not None and self.max_tokens_a is not None:
            raise ValueError(
                "ConversationSpec: max_tokens_a cannot be used with direct conversation_starter"
            )

    def is_direct(self) -> bool:
        """Return True if this is a direct conversation starter (no Model A)."""
        return self.conversation_starter is not None

    def uses_extraction(self) -> bool:
        """Return True if this spec uses extraction to create multiple starters."""
        return self.extraction_fn is not None

    def is_prefill(self) -> bool:
        """Return True if this spec uses prefill mode (bypasses Model B generation)."""
        return self.prefill_with_article


@dataclass
class SelfStudyConfig:
    """
    Configuration for self-study query generation.

    Parameters
    ----------
    conversation_specs : List[ConversationSpec]
        List of conversation specifications. Each spec defines either a seed prompt
        for Model A or a direct conversation starter for Model B.

    Note
    ----
    Temperature, top_k, and top_p are automatically set based on enable_thinking:
    - With thinking: temperature=0.6, top_p=0.95, top_k=20
    - Without thinking: temperature=0.7, top_p=0.8, top_k=20

    vLLM is always used for generation, with HuggingFace for query extraction.
    """
    conversation_specs: List[ConversationSpec] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        if not self.conversation_specs:
            raise ValueError("conversation_specs cannot be empty")


@dataclass
class RandomVectorConfig:
    """
    Configuration for random vector query generation.

    Parameters
    ----------
    scale_by_qnorm : bool
        If True, scale random vectors by q_norm weights for the corresponding
        layer (default: True)
    """
    scale_by_qnorm: bool = True


@dataclass
class CacheKeysConfig:
    """
    Configuration for cache keys query generation.

    This method uses the key vectors from the KV cache itself as query vectors.
    Optionally applies q_norm scaling to match the scale of actual queries.

    Parameters
    ----------
    scale_by_qnorm : bool
        If True, scale random vectors by q_norm weights for the corresponding
        layer (default: True)
    """
    scale_by_qnorm: bool = True


@dataclass
class ContextPrefillConfig:
    """
    Configuration for context prefill query generation.

    This method extracts queries directly from the article portion of the
    formatted_context by running a single prefill pass. This is the simplest
    query generation method - the article "studies itself".

    Parameters
    ----------
    # No configuration parameters needed for basic context prefill
    """
    pass


@dataclass
class QueryMethodConfig:
    """
    Configuration for a single query generation method.

    Parameters
    ----------
    method : str
        Method name ('self_study', 'random_vectors', etc.)
    fraction : float
        Fraction of queries to generate using this method (0.0 to 1.0)
    config : Any
        Method-specific configuration (SelfStudyConfig, RandomVectorConfig, etc.)
    """
    method: str
    fraction: float
    config: Any

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(
                f"fraction must be between 0.0 and 1.0, got {self.fraction}"
            )


@dataclass
class QueryConfig:
    """
    Configuration for query generation strategies.

    This config allows mixing multiple query generation methods
    (e.g., self-study, random vectors) for KV cache compaction.

    Parameters
    ----------
    method_configs : List[QueryMethodConfig]
        List of method configurations with their respective fractions.
        Fractions should sum to 1.0 (or close to it).
        Example:
            [
                QueryMethodConfig('self_study', 0.7, SelfStudyConfig(...)),
                QueryMethodConfig('random_vectors', 0.3, RandomVectorConfig(...))
            ]
    max_query_vectors_per_kv_head : int
        Maximum number of query vectors per KV head
    eval_queries_per_kv_head : int
        Maximum number of queries to use per KV head for evaluation (train/test stats).
        This is used to subsample queries during compaction stats computation. (default: 1000)
    verbose : bool
        Enable debug logging for query generation (default: False)
    """
    method_configs: List[QueryMethodConfig] = field(default_factory=list)
    max_query_vectors_per_kv_head: int = 10000
    eval_queries_per_kv_head: int = 1000
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.max_query_vectors_per_kv_head < 1:
            raise ValueError(
                f"max_query_vectors_per_kv_head must be at least 1, "
                f"got {self.max_query_vectors_per_kv_head}"
            )

        if not self.method_configs:
            raise ValueError("method_configs cannot be empty")

        # Check that fractions sum to approximately 1.0
        total_fraction = sum(mc.fraction for mc in self.method_configs)
        if not (0.99 <= total_fraction <= 1.01):
            raise ValueError(
                f"method_configs fractions should sum to 1.0, got {total_fraction}"
            )

        # Check for duplicate methods
        methods = [mc.method for mc in self.method_configs]
        if len(methods) != len(set(methods)):
            raise ValueError(f"Duplicate methods found in method_configs: {methods}")

    def get_method_config(self, method: str) -> Optional[QueryMethodConfig]:
        """Get configuration for a specific method."""
        for mc in self.method_configs:
            if mc.method == method:
                return mc
        return None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QueryConfig':
        """
        Create QueryConfig from dictionary (e.g., from asdict()).

        Parameters
        ----------
        config_dict : dict
            Dictionary representation of QueryConfig

        Returns
        -------
        QueryConfig
        """
        # Reconstruct method_configs
        method_configs = []
        for mc_dict in config_dict['method_configs']:
            method = mc_dict['method']
            fraction = mc_dict['fraction']

            # Reconstruct method-specific config
            if method == 'self_study':
                # Reconstruct SelfStudyConfig with ConversationSpecs
                config_data = mc_dict['config'].copy()

                # Reconstruct ConversationSpec objects from dicts
                if 'conversation_specs' in config_data and config_data['conversation_specs']:
                    conversation_specs = []
                    for spec_dict in config_data['conversation_specs']:
                        # Handle extraction_fn which can't be serialized
                        spec_dict_copy = spec_dict.copy()

                        # extraction_fn is not serializable, so we skip it during deserialization
                        # The user should reconstruct the ConversationSpec from the registry if needed
                        if 'extraction_fn' in spec_dict_copy:
                            del spec_dict_copy['extraction_fn']

                        conversation_specs.append(ConversationSpec(**spec_dict_copy))
                    config_data['conversation_specs'] = conversation_specs

                method_config = SelfStudyConfig(**config_data)
            elif method == 'random_vectors':
                method_config = RandomVectorConfig(**mc_dict['config'])
            elif method == 'cache_keys':
                method_config = CacheKeysConfig(**mc_dict['config'])
            elif method == 'context_prefill':
                method_config = ContextPrefillConfig(**mc_dict['config'])
            else:
                # Unknown method, keep as dict
                method_config = mc_dict['config']

            method_configs.append(QueryMethodConfig(
                method=method,
                fraction=fraction,
                config=method_config
            ))

        return cls(
            method_configs=method_configs,
            max_query_vectors_per_kv_head=config_dict['max_query_vectors_per_kv_head'],
            eval_queries_per_kv_head=config_dict['eval_queries_per_kv_head'],
            verbose=config_dict.get('verbose', False)
        )
