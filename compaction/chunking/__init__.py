# compaction/chunking/__init__.py
"""
Chunking module for splitting articles into processable chunks.

This module provides strategies for splitting long articles into smaller chunks
that can be independently processed by the compaction pipeline.
"""

from .strategies import (
    Chunk,
    ChunkingStrategy,
    FixedSizeChunking,
    LongHealthChunking,
    LQAChunking,
    get_chunking_strategy,
)

__all__ = [
    'Chunk',
    'ChunkingStrategy',
    'FixedSizeChunking',
    'LongHealthChunking',
    'LQAChunking',
    'get_chunking_strategy',
]
