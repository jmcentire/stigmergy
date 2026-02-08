"""Tests for advanced data structures: bloom filter, LSH, trie, Aho-Corasick, SimHash, ring buffer."""

from __future__ import annotations

import numpy as np
import pytest

from stigmergy.structures.bloom import CountingBloomFilter
from stigmergy.structures.lsh import LSHIndex
from stigmergy.structures.ring_buffer import RingBuffer
from stigmergy.structures.simhash import SimHash, SimHashIndex
from stigmergy.structures.trie import AhoCorasick, Trie


# --- Counting Bloom Filter ---

class TestCountingBloomFilter:
    def test_membership(self):
        bf = CountingBloomFilter(capacity=1000)
        bf.add("hello")
        bf.add("world")
        assert "hello" in bf
        assert "world" in bf
        assert "missing" not in bf

    def test_no_false_negatives(self):
        """Items that were added must always be found."""
        bf = CountingBloomFilter(capacity=10000)
        words = [f"word_{i}" for i in range(500)]
        bf.add_many(words)
        for w in words:
            assert w in bf

    def test_removal(self):
        bf = CountingBloomFilter(capacity=1000)
        bf.add("removeme")
        assert "removeme" in bf
        bf.remove("removeme")
        assert "removeme" not in bf

    def test_overlap_identical_filters(self):
        a = CountingBloomFilter(capacity=1000, fp_rate=0.01)
        b = CountingBloomFilter(capacity=1000, fp_rate=0.01)
        words = ["alpha", "beta", "gamma"]
        a.add_many(words)
        b.add_many(words)
        overlap = a.estimate_jaccard(b)
        assert overlap > 0.8  # should be close to 1.0

    def test_overlap_disjoint_filters(self):
        a = CountingBloomFilter(capacity=1000, fp_rate=0.01)
        b = CountingBloomFilter(capacity=1000, fp_rate=0.01)
        a.add_many(["alpha", "beta", "gamma"])
        b.add_many(["delta", "epsilon", "zeta"])
        overlap = a.estimate_jaccard(b)
        assert overlap < 0.3  # should be close to 0.0

    def test_overlap_partial(self):
        a = CountingBloomFilter(capacity=1000, fp_rate=0.01)
        b = CountingBloomFilter(capacity=1000, fp_rate=0.01)
        a.add_many(["shared", "only_a"])
        b.add_many(["shared", "only_b"])
        overlap = a.estimate_jaccard(b)
        assert 0.2 < overlap < 0.8

    def test_count_tracking(self):
        bf = CountingBloomFilter(capacity=100)
        assert bf.count == 0
        bf.add("a")
        bf.add("b")
        assert bf.count == 2

    def test_clear(self):
        bf = CountingBloomFilter(capacity=100)
        bf.add_many(["a", "b", "c"])
        bf.clear()
        assert bf.count == 0
        assert "a" not in bf


# --- LSH Index ---

class TestLSHIndex:
    def test_similar_vectors_found(self):
        lsh = LSHIndex(dimensions=64, num_tables=20, num_hyperplanes=6, seed=42)
        rng = np.random.RandomState(1)
        base = rng.randn(64)
        base = base / np.linalg.norm(base)
        # Add base vector and a similar one (small perturbation)
        similar = base + rng.randn(64) * 0.1
        similar = similar / np.linalg.norm(similar)
        lsh.add("base", base.tolist())
        lsh.add("similar", similar.tolist())
        lsh.add("random", rng.randn(64).tolist())

        results = lsh.query(base.tolist(), top_k=2)
        ids = [r[0] for r in results]
        assert "base" in ids
        # Similar should be ranked high
        if len(results) > 1:
            assert results[0][1] >= results[1][1]  # sorted by similarity

    def test_remove(self):
        lsh = LSHIndex(dimensions=32, seed=42)
        lsh.add("x", [1.0] * 32)
        assert "x" in lsh
        lsh.remove("x")
        assert "x" not in lsh

    def test_update(self):
        lsh = LSHIndex(dimensions=32, seed=42)
        lsh.add("x", [1.0] * 32)
        lsh.update("x", [0.0] * 32)
        assert len(lsh) == 1


# --- Trie ---

class TestTrie:
    def test_insert_and_search(self):
        t = Trie()
        t.insert("#engineering", "ctx_1")
        t.insert("#engineering-platform", "ctx_2")
        t.insert("#support", "ctx_3")

        assert t.search("#engineering") == ["ctx_1"]
        assert t.search("#support") == ["ctx_3"]
        assert t.search("#missing") is None

    def test_prefix_search(self):
        t = Trie()
        t.insert("#engineering", "ctx_1")
        t.insert("#engineering-platform", "ctx_2")
        t.insert("#engineering-data", "ctx_3")
        t.insert("#support", "ctx_4")

        results = t.starts_with("#engineering")
        keys = {k for k, _ in results}
        assert "#engineering" in keys
        assert "#engineering-platform" in keys
        assert "#engineering-data" in keys
        assert "#support" not in keys

    def test_longest_prefix(self):
        t = Trie()
        t.insert("#eng", "short")
        t.insert("#engineering", "long")

        result = t.longest_prefix("#engineering-platform")
        assert result is not None
        assert result[0] == "#engineering"

    def test_contains(self):
        t = Trie()
        t.insert("key")
        assert "key" in t
        assert "missing" not in t

    def test_len(self):
        t = Trie()
        assert len(t) == 0
        t.insert("a")
        t.insert("b")
        assert len(t) == 2


# --- Aho-Corasick ---

class TestAhoCorasick:
    def test_single_pattern(self):
        ac = AhoCorasick()
        ac.add_pattern("error", "bug")
        ac.build()
        results = ac.search("there was an error in the system")
        assert len(results) == 1
        assert results[0][1] == "error"
        assert results[0][2] == "bug"

    def test_multiple_patterns(self):
        ac = AhoCorasick()
        ac.add_pattern("booking", "domain")
        ac.add_pattern("sync", "domain")
        ac.add_pattern("error", "severity")
        ac.build()
        results = ac.search("booking sync error on guesty webhook")
        matched = {r[1] for r in results}
        assert "booking" in matched
        assert "sync" in matched
        assert "error" in matched

    def test_no_match(self):
        ac = AhoCorasick()
        ac.add_pattern("xyz123", "test")
        ac.build()
        assert not ac.has_any_match("nothing relevant here")

    def test_has_any_match_fast(self):
        ac = AhoCorasick()
        ac.add_pattern("secret", "cred")
        ac.build()
        assert ac.has_any_match("this contains a secret value")
        assert not ac.has_any_match("clean content here")

    def test_case_insensitive(self):
        ac = AhoCorasick()
        ac.add_pattern("Error", "bug")
        ac.build()
        assert ac.has_any_match("THERE WAS AN ERROR")

    def test_overlapping_patterns(self):
        ac = AhoCorasick()
        ac.add_pattern("he", "short")
        ac.add_pattern("her", "medium")
        ac.add_pattern("hers", "long")
        ac.build()
        results = ac.search("hers")
        assert len(results) >= 2  # should find overlapping matches

    def test_matched_categories(self):
        ac = AhoCorasick()
        ac.add_pattern("password", "cred")
        ac.add_pattern("token", "cred")
        ac.add_pattern("salary", "finance")
        ac.build()
        cats = ac.matched_categories("reset your password and token")
        assert "cred" in cats
        assert "finance" not in cats


# --- SimHash ---

class TestSimHash:
    def test_identical_fingerprints(self):
        terms = ["booking", "sync", "error", "guesty"]
        fp1 = SimHash.fingerprint(terms)
        fp2 = SimHash.fingerprint(terms)
        assert fp1 == fp2

    def test_similar_content_small_distance(self):
        terms_a = ["booking", "sync", "error", "guesty", "webhook"]
        terms_b = ["booking", "sync", "error", "guesty", "callback"]
        fp_a = SimHash.fingerprint(terms_a)
        fp_b = SimHash.fingerprint(terms_b)
        dist = SimHash.hamming_distance(fp_a, fp_b)
        assert dist < 20  # similar content, small distance

    def test_different_content_large_distance(self):
        terms_a = ["booking", "sync", "error"]
        terms_b = ["revenue", "forecast", "quarterly"]
        fp_a = SimHash.fingerprint(terms_a)
        fp_b = SimHash.fingerprint(terms_b)
        dist = SimHash.hamming_distance(fp_a, fp_b)
        assert dist > 10  # different content, larger distance

    def test_similarity_range(self):
        fp_a = SimHash.fingerprint(["hello", "world"])
        fp_b = SimHash.fingerprint(["hello", "world"])
        assert SimHash.similarity(fp_a, fp_b) == 1.0

    def test_empty_terms(self):
        fp = SimHash.fingerprint([])
        assert fp == 0


class TestSimHashIndex:
    def test_find_near_duplicate(self):
        idx = SimHashIndex(num_blocks=16, distance_threshold=10)
        terms_a = ["booking", "sync", "error", "guesty", "webhook"]
        terms_b = ["booking", "sync", "error", "guesty", "callback"]
        fp_a = SimHash.fingerprint(terms_a)
        fp_b = SimHash.fingerprint(terms_b)
        idx.add("a", fp_a)
        assert idx.has_near_duplicate(fp_b)

    def test_no_false_duplicate(self):
        idx = SimHashIndex(distance_threshold=3)
        terms_a = ["booking", "sync"]
        terms_b = ["revenue", "forecast", "quarterly", "analysis"]
        fp_a = SimHash.fingerprint(terms_a)
        fp_b = SimHash.fingerprint(terms_b)
        idx.add("a", fp_a)
        assert not idx.has_near_duplicate(fp_b)

    def test_remove(self):
        idx = SimHashIndex(distance_threshold=5)
        fp = SimHash.fingerprint(["test"])
        idx.add("x", fp)
        idx.remove("x")
        assert not idx.has_near_duplicate(fp)


# --- Ring Buffer ---

class TestRingBuffer:
    def test_append_and_read(self):
        rb = RingBuffer(3)
        rb.append("a")
        rb.append("b")
        rb.append("c")
        assert rb[0] == "a"
        assert rb[1] == "b"
        assert rb[2] == "c"
        assert rb[-1] == "c"

    def test_overflow_evicts_oldest(self):
        rb = RingBuffer(3)
        rb.append("a")
        rb.append("b")
        rb.append("c")
        evicted = rb.append("d")
        assert evicted == "a"
        assert rb[0] == "b"
        assert rb[-1] == "d"
        assert len(rb) == 3

    def test_last(self):
        rb = RingBuffer(5)
        for i in range(5):
            rb.append(i)
        assert rb.last(3) == [4, 3, 2]

    def test_iteration(self):
        rb = RingBuffer(3)
        rb.append(1)
        rb.append(2)
        rb.append(3)
        assert list(rb) == [1, 2, 3]

    def test_iteration_after_overflow(self):
        rb = RingBuffer(3)
        for i in range(5):
            rb.append(i)
        assert list(rb) == [2, 3, 4]

    def test_full(self):
        rb = RingBuffer(2)
        assert not rb.full
        rb.append("a")
        rb.append("b")
        assert rb.full

    def test_clear(self):
        rb = RingBuffer(3)
        rb.append("a")
        rb.clear()
        assert len(rb) == 0

    def test_index_out_of_range(self):
        rb = RingBuffer(3)
        rb.append("a")
        with pytest.raises(IndexError):
            _ = rb[1]
