# compaction/query_generation/conversation_specs.py
"""Registry of predefined conversation specs for self-study query generation."""

from typing import Dict, List
from .config import ConversationSpec


def extract_after_thinking(text: str) -> List[str]:
    """
    Extract text after thinking tags.

    Useful when Model A has thinking enabled and you only want to use
    the non-thinking portion as the conversation starter.

    For models without thinking support (e.g., Gemma, Llama, Qwen3-4B),
    returns the full text when no </think> tag is found.

    Parameters
    ----------
    text : str
        Generated text potentially containing thinking tags

    Returns
    -------
    List[str]
        List with a single extracted text (after thinking), or full text if no </think> tag found
    """
    # Look for </think> tag (HTML-style closing tag)
    if "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) > 1:
            extracted = parts[1].strip()
            if extracted:  # Only return if there's content after </think>
                return [extracted]
        # Had </think> but nothing after it - return empty
        return []
    # No thinking tags found - model doesn't support thinking, use full text
    stripped = text.strip()
    return [stripped] if stripped else []


def split_on_double_newline(text: str) -> List[str]:
    """
    Split text on double newlines to extract multiple items.

    Useful for extracting multiple MCQs, questions, etc. that are
    separated by blank lines.

    Parameters
    ----------
    text : str
        Generated text with items separated by double newlines

    Returns
    -------
    List[str]
        List of extracted items
    """
    items = text.split("\n\n")
    return [item.strip() for item in items if item.strip()]


def extract_after_thinking_then_split(text: str) -> List[str]:
    """
    Extract text after thinking tags, then split on separators. 
    Annoying parsing to support different models behaviors. TODO: switch to json outputs

    Useful when Model A has thinking enabled and generates multiple items.
    First removes thinking tags (if present), then splits on double newlines (or other indicators).

    Parameters
    ----------
    text : str
        Generated text potentially containing thinking tags and multiple items

    Returns
    -------
    List[str]
        List of extracted items, or empty list if more than 5 items
    """
    import re

    # Try to extract after </think>, but if not found, use full text
    if "</think>" in text:
        parts = text.split("</think>", 1)
        content = parts[1].strip() if len(parts) > 1 else text
    else:
        content = text.strip()

    if not content:
        return []

    # Try splitting on dash separator lines first (handles ---+ separators)
    items = re.split(r'\n+\s*-{3,}\s*\n+', content)

    # If no dash separators found, try splitting on numbered items (1. 2. 3. etc.)
    # This keeps each numbered question with its options together
    if len(items) == 1:
        # Split on pattern: newline(s) followed by a number and period/parenthesis at start of line
        # Use lookahead to keep the number with the following content
        numbered_items = re.split(r'\n+(?=\d+[\.\)]\s)', content)
        if len(numbered_items) > 1:
            items = numbered_items

    # If still only 1 item, fall back to double newline
    if len(items) == 1:
        items = content.split("\n\n")

    items = [item.strip() for item in items if item.strip()]

    # Check if we have question/options pairs that got split apart
    # (e.g., 6 items where items 1,3,5 are options starting with "A)")
    # If so, merge each options block back with the preceding question
    if len(items) > 5:
        # Check if every other item (starting from index 1) starts with A) or A.
        options_pattern = re.compile(r'^A[\)\.]')
        might_be_pairs = all(
            options_pattern.match(items[i])
            for i in range(1, len(items), 2)
        )
        if might_be_pairs and len(items) % 2 == 0:
            # Merge pairs: item[0]+item[1], item[2]+item[3], etc.
            merged = []
            for i in range(0, len(items), 2):
                merged.append(items[i] + "\n\n" + items[i + 1])
            items = merged

    # Discard if more than 5 items
    if len(items) > 5:
        return []
    return items


# Each spec is identified by a unique key that can be used in configs
CONVERSATION_SPEC_REGISTRY: Dict[str, ConversationSpec] = {
    "question": ConversationSpec(
        seed_prompt=(
            "Generate a question for another LLM that will test its knowledge of the "
            "information in the context. In your question be sure to include details "
            "that make it clear what you are asking about. Output only a single question. "
            "Do not say the correct answer or give any other explanation. "
        ),
        extraction_fn=extract_after_thinking,
        enable_thinking_a=True,
        max_tokens_a=8192,
        enable_thinking_b=True,
        max_tokens_b=4096,
    ),
    "3_question": ConversationSpec(
        seed_prompt=(
            "Write 3 questions that test understanding of different parts of the context. "
            "Answer with just the 3 questions and options (do not say the correct answer), each one separated with 2 newlines."
        ),
        extraction_fn=extract_after_thinking_then_split,
        enable_thinking_a=True,
        max_tokens_a=8192,
        enable_thinking_b=True,
        max_tokens_b=4096,
    ),
    "repeat": ConversationSpec(
        conversation_starter="Repeat the previous context verbatim.",
        prefill_with_article=True,
    ),
    "summarize": ConversationSpec(
        conversation_starter="Summarize the main points of the context.",
        enable_thinking_b=False,
        max_tokens_b=2048,
    ),
    "summarize_backward": ConversationSpec(
        conversation_starter="In reverse chronological order, summarize the main points of the context.",
        enable_thinking_b=False,
        max_tokens_b=2048,
    ),
    "structure_json": ConversationSpec(
        conversation_starter="Structure the information in JSON form and include all important details like dates, times, names, and numerical values.",
        enable_thinking_b=False,
        max_tokens_b=2048,
    ),
    "structure_yaml": ConversationSpec(
        conversation_starter="Structure the information in YAML form and include all important details like dates, times, names, and numerical values.",
        enable_thinking_b=False,
        max_tokens_b=2048,
    ),
    "aggregate": ConversationSpec(
        conversation_starter="Aggregate all the key facts mentioned in the context.",
        enable_thinking_b=False,
        max_tokens_b=2048,
    ),
    "rephrase": ConversationSpec(
        conversation_starter="Rephrase the entire context from start to finish using different wording while preserving the original meaning.",
        enable_thinking_b=False,
        max_tokens_b=8192,
    ),
}


def get_spec(spec_key: str) -> ConversationSpec:
    """
    Get a conversation spec from the registry by key.

    Parameters
    ----------
    spec_key : str
        Key identifying the spec in the registry

    Returns
    -------
    ConversationSpec
        The conversation spec

    Raises
    ------
    KeyError
        If spec_key is not in the registry
    """
    if spec_key not in CONVERSATION_SPEC_REGISTRY:
        raise KeyError(
            f"Unknown conversation spec: {spec_key}. "
            f"Available specs: {list(CONVERSATION_SPEC_REGISTRY.keys())}"
        )
    return CONVERSATION_SPEC_REGISTRY[spec_key]


def get_specs(spec_keys: List[str]) -> List[ConversationSpec]:
    """
    Get multiple conversation specs from the registry.

    Parameters
    ----------
    spec_keys : list of str
        Keys identifying the specs in the registry

    Returns
    -------
    list of ConversationSpec
        The conversation specs
    """
    return [get_spec(key) for key in spec_keys]


def repeat_specs(spec_config: List[tuple[str, int]]) -> List[ConversationSpec]:
    """
    Create a list of ConversationSpecs by repeating registry specs N times.

    This is a helper function for configs to easily specify which specs they want
    and how many times to repeat each.

    Parameters
    ----------
    spec_config : list of (str, int) tuples
        List of (spec_key, count) tuples specifying which specs to use and how many times.
        Example: [("question", 2), ("difficult_mcq", 1), ("repeat", 1)]
        This would create a list with 2 question entries, 1 difficult_mcq entry, and 1 repeat entry.

    Returns
    -------
    list of ConversationSpec
        List of conversation specs with repetitions.

    Examples
    --------
    >>> repeat_specs([("question", 2), ("repeat", 1)])
    [ConversationSpec(seed_prompt="Generate a question..."),
     ConversationSpec(seed_prompt="Generate a question..."),
     ConversationSpec(conversation_starter="Repeat the previous context verbatim.")]
    """
    result = []
    for spec_key, count in spec_config:
        spec = get_spec(spec_key)
        result.extend([spec] * count)
    return result
