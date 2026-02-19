# compaction/chunking/strategies.py
"""
Chunking strategies for splitting articles into processable chunks.

Each strategy implements a different way to split articles:
- FixedSizeChunking: Split into fixed N-token pieces
- LongHealthChunking: Split on <text_0> tags, grouping note chains together
- LongHealthFineChunking: Split on each <text_X> tag (medical notes)
- LQAChunking: Split on [start of {filename}] markers (code files)
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Chunk:
    """Represents a chunk of text from an article."""
    text: str
    start_token_idx: int  # Position in original article (approximate)
    end_token_idx: int    # Position in original article (approximate)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, article_text: str, tokenizer) -> List[Chunk]:
        """
        Split article into chunks.

        Parameters
        ----------
        article_text : str
            The article text to chunk
        tokenizer : PreTrainedTokenizer
            Tokenizer for computing token positions

        Returns
        -------
        chunks : List[Chunk]
            List of chunks with text and metadata
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this chunking strategy."""
        pass


class FixedSizeChunking(ChunkingStrategy):
    """Split article into fixed-size token chunks."""

    def __init__(self, chunk_size: int = 4096):
        """
        Initialize fixed-size chunking.

        Parameters
        ----------
        chunk_size : int
            Number of tokens per chunk (default: 4096)
        """
        self.chunk_size = chunk_size

    @property
    def name(self) -> str:
        return f"fixed_{self.chunk_size}"

    def chunk(self, article_text: str, tokenizer) -> List[Chunk]:
        """Split article into fixed-size token chunks."""
        # Tokenize full article
        tokens = tokenizer.encode(article_text, add_special_tokens=False)

        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(Chunk(
                text=chunk_text,
                start_token_idx=i,
                end_token_idx=min(i + self.chunk_size, len(tokens)),
                metadata={'chunk_idx': len(chunks)}
            ))

        return chunks


class LongHealthChunking(ChunkingStrategy):
    """Split article on <text_0> tags, grouping all notes until the next <text_0>."""

    @property
    def name(self) -> str:
        return "longhealth"

    def chunk(self, article_text: str, tokenizer) -> List[Chunk]:
        """
        Split article on <text_0> tags, grouping consecutive notes.

        Each chunk contains all notes from <text_0> up to </text_n>
        (right before the next <text_0>). This groups related medical
        note chains together.
        """
        # Find all <text_0> positions to use as split points
        split_pattern = r'(?=<text_0>)'
        parts = re.split(split_pattern, article_text)

        chunks = []
        cumulative_len = 0

        for part in parts:
            part_stripped = part.strip()
            if not part_stripped:
                cumulative_len += len(part)
                continue

            # Only create a chunk if this part starts with <text_0>
            if part_stripped.startswith('<text_0>'):
                # Compute approximate token positions
                chunk_tokens = tokenizer.encode(part_stripped, add_special_tokens=False)
                prefix_tokens = tokenizer.encode(article_text[:cumulative_len], add_special_tokens=False)

                # Find the highest note number in this chunk
                note_ids = re.findall(r'<text_(\d+)>', part_stripped)
                max_note = max(int(n) for n in note_ids) if note_ids else 0

                chunks.append(Chunk(
                    text=part_stripped,
                    start_token_idx=len(prefix_tokens),
                    end_token_idx=len(prefix_tokens) + len(chunk_tokens),
                    metadata={
                        'note_range': f'text_0-text_{max_note}',
                        'num_notes': len(note_ids)
                    }
                ))

            cumulative_len += len(part)

        # If no chunks found (no <text_0> tags), return the whole article as one chunk
        if not chunks:
            tokens = tokenizer.encode(article_text, add_special_tokens=False)
            chunks.append(Chunk(
                text=article_text,
                start_token_idx=0,
                end_token_idx=len(tokens),
                metadata={'note_range': None, 'num_notes': 0}
            ))

        return chunks


class LongHealthFineChunking(ChunkingStrategy):
    """Split article on each <text_X> tag (LongHealth medical notes)."""

    @property
    def name(self) -> str:
        return "longhealth_fine"

    def chunk(self, article_text: str, tokenizer) -> List[Chunk]:
        """
        Split article on <text_X>...</text_X> blocks.

        Each medical note in LongHealth is wrapped in tags like:
        <text_0>...</text_0>, <text_1>...</text_1>, etc.
        """
        # Match <text_X>...</text_X> blocks including the tags
        pattern = r'(<text_\d+>.*?</text_\d+>)'
        parts = re.split(pattern, article_text, flags=re.DOTALL)

        chunks = []
        cumulative_len = 0

        for part in parts:
            part_stripped = part.strip()
            if not part_stripped:
                cumulative_len += len(part)
                continue

            # Check if this is a text block
            note_match = re.match(r'<(text_\d+)>', part_stripped)
            if note_match:
                note_id = note_match.group(1)

                # Compute approximate token positions
                chunk_tokens = tokenizer.encode(part_stripped, add_special_tokens=False)
                prefix_tokens = tokenizer.encode(article_text[:cumulative_len], add_special_tokens=False)

                chunks.append(Chunk(
                    text=part_stripped,
                    start_token_idx=len(prefix_tokens),
                    end_token_idx=len(prefix_tokens) + len(chunk_tokens),
                    metadata={'note_id': note_id}
                ))

            cumulative_len += len(part)

        # If no chunks found (no <text_X> tags), return the whole article as one chunk
        if not chunks:
            tokens = tokenizer.encode(article_text, add_special_tokens=False)
            chunks.append(Chunk(
                text=article_text,
                start_token_idx=0,
                end_token_idx=len(tokens),
                metadata={'note_id': None}
            ))

        return chunks


class LQAChunking(ChunkingStrategy):
    """Split article on [start of {filename}] markers (LongCodeQA code files)."""

    def __init__(self, max_chunk_size: int = 10000):
        """
        Initialize LQA chunking.

        Parameters
        ----------
        max_chunk_size : int
            Maximum tokens per chunk. Files larger than this will be
            split into multiple chunks (default: 10000)
        """
        self.max_chunk_size = max_chunk_size

    @property
    def name(self) -> str:
        return "lqa"

    def _split_large_chunk(self, text: str, filename: str, start_token_idx: int,
                           tokenizer) -> List[Chunk]:
        """Split a chunk that exceeds max_chunk_size into smaller pieces."""
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_chunk_size:
            return [Chunk(
                text=text,
                start_token_idx=start_token_idx,
                end_token_idx=start_token_idx + len(tokens),
                metadata={'filename': filename}
            )]

        # Split into multiple chunks
        chunks = []
        for i in range(0, len(tokens), self.max_chunk_size):
            chunk_tokens = tokens[i:i + self.max_chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            part_num = i // self.max_chunk_size + 1
            total_parts = (len(tokens) + self.max_chunk_size - 1) // self.max_chunk_size

            chunks.append(Chunk(
                text=chunk_text,
                start_token_idx=start_token_idx + i,
                end_token_idx=start_token_idx + i + len(chunk_tokens),
                metadata={
                    'filename': filename,
                    'part': part_num,
                    'total_parts': total_parts
                }
            ))

        return chunks

    def chunk(self, article_text: str, tokenizer) -> List[Chunk]:
        """
        Split article on [start of {filename}] markers.

        LongCodeQA articles contain code files marked with:
        [start of path/to/file.py]
        ... code content ...
        [start of another/file.py]
        ... more code ...
        """
        # Match [start of {anything}] markers
        pattern = r'(\[start of [^\]]+\])'
        parts = re.split(pattern, article_text)

        chunks = []
        current_chunk_text = ""
        current_filename = None
        cumulative_char_pos = 0

        for part in parts:
            if part.startswith('[start of '):
                # Save previous chunk if exists
                if current_chunk_text.strip():
                    # Compute token positions
                    prefix_tokens = tokenizer.encode(
                        article_text[:cumulative_char_pos - len(current_chunk_text)],
                        add_special_tokens=False
                    )

                    # Split into smaller chunks if needed
                    chunks.extend(self._split_large_chunk(
                        current_chunk_text,
                        current_filename,
                        len(prefix_tokens),
                        tokenizer
                    ))

                # Extract filename from [start of {filename}]
                filename_match = re.search(r'\[start of (.+)\]', part)
                current_filename = filename_match.group(1) if filename_match else None
                current_chunk_text = part
            else:
                current_chunk_text += part

            cumulative_char_pos += len(part)

        # Don't forget the last chunk
        if current_chunk_text.strip():
            prefix_tokens = tokenizer.encode(
                article_text[:cumulative_char_pos - len(current_chunk_text)],
                add_special_tokens=False
            )

            # Split into smaller chunks if needed
            chunks.extend(self._split_large_chunk(
                current_chunk_text,
                current_filename,
                len(prefix_tokens),
                tokenizer
            ))

        # If no chunks found, return the whole article as one chunk (possibly split)
        if not chunks:
            chunks.extend(self._split_large_chunk(
                article_text,
                None,
                0,
                tokenizer
            ))

        return chunks


def get_chunking_strategy(name: str, **kwargs) -> Optional[ChunkingStrategy]:
    """
    Factory function for chunking strategies.

    Parameters
    ----------
    name : str
        Name of the chunking strategy:
        - 'fixed': Fixed-size chunking (requires chunk_size kwarg)
        - 'longhealth': LongHealth medical note chunking
        - 'lqa': LongCodeQA code file chunking
        - 'none' or None: No chunking (returns None)
    **kwargs : dict
        Additional arguments for the strategy (e.g., chunk_size for fixed)

    Returns
    -------
    strategy : ChunkingStrategy or None
        The chunking strategy instance, or None if name is 'none'/None
    """
    if name is None or name.lower() == 'none':
        return None

    name_lower = name.lower()

    if name_lower == 'fixed':
        chunk_size = kwargs.get('chunk_size', 4096)
        return FixedSizeChunking(chunk_size=chunk_size)
    elif name_lower == 'longhealth':
        return LongHealthChunking()
    elif name_lower == 'longhealth_fine':
        return LongHealthFineChunking()
    elif name_lower == 'lqa':
        max_chunk_size = kwargs.get('max_chunk_size', 10000)
        return LQAChunking(max_chunk_size=max_chunk_size)
    else:
        raise ValueError(
            f"Unknown chunking strategy: {name}. "
            f"Supported strategies: 'fixed', 'longhealth', 'longhealth_fine', 'lqa', 'none'"
        )
