"""Tests for the chunker -- the component whose bug produced 49 docs -> 49 chunks."""
from __future__ import annotations

import pytest

from pipeline.etl import chunk_text


def test_empty_and_whitespace_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t ") == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("CPI inflation was 5.2 percent", chunk_size=180)
    assert chunks == ["CPI inflation was 5.2 percent"]


def test_long_text_is_split_into_many_chunks():
    """The core regression: a long document must not collapse to one chunk."""
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=180, overlap=40)
    assert len(chunks) > 5
    assert all(len(c.split()) <= 180 for c in chunks)


def test_chunks_overlap_by_the_requested_number_of_words():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    first_tail = chunks[0].split()[-20:]
    second_head = chunks[1].split()[:20]
    assert first_tail == second_head


def test_every_word_is_preserved_across_chunks():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(text.split())


def test_overlap_at_or_above_chunk_size_is_clamped_and_terminates():
    """Guard against a non-advancing window (an infinite loop in the old code)."""
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=50, overlap=50)
    assert len(chunks) > 1
    chunks = chunk_text(text, chunk_size=50, overlap=999)
    assert len(chunks) > 1


def test_non_positive_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=0)


def test_short_trailing_fragment_is_merged_into_previous_chunk():
    text = " ".join(f"w{i}" for i in range(105))
    chunks = chunk_text(text, chunk_size=100, overlap=0, min_words=20)
    assert len(chunks) == 1
    assert len(chunks[0].split()) == 105


def test_short_trailing_fragment_kept_when_min_words_is_zero():
    text = " ".join(f"w{i}" for i in range(105))
    chunks = chunk_text(text, chunk_size=100, overlap=0, min_words=0)
    assert len(chunks) == 2
