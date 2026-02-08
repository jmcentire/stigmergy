"""Tests for AliasStore — JSON persistence for learned aliases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stigmergy.identity.store import AliasStore


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "learned_aliases.json"


@pytest.fixture
def store(store_path):
    return AliasStore(path=store_path)


class TestAliasStore:
    def test_empty_store(self, store):
        assert store.group_count == 0
        assert store.total_aliases == 0
        assert store.dirty is False

    def test_add_alias(self, store):
        store.add("alice", "alice-gh")
        assert store.group_count == 1
        assert store.total_aliases == 1
        assert store.dirty is True

    def test_add_multiple_aliases(self, store):
        store.add("alice", "alice-gh")
        store.add("alice", "alice.smith")
        store.add("bob", "bobjones")
        assert store.group_count == 2
        assert store.total_aliases == 3

    def test_add_duplicate_alias(self, store):
        store.add("alice", "alice-gh")
        store.add("alice", "alice-gh")  # duplicate
        assert store.total_aliases == 1

    def test_add_normalizes(self, store):
        store.add("alice", "  Alice-GH  ")
        aliases = store.all_aliases()
        assert "alice-gh" in aliases["alice"]

    def test_add_empty_alias_ignored(self, store):
        store.add("alice", "")
        store.add("alice", "   ")
        assert store.group_count == 0

    def test_save_and_load_roundtrip(self, store, store_path):
        store.add("alice", "alice-gh")
        store.add("alice", "alice.smith")
        store.add("bob", "bobjones")
        store.save()

        assert store_path.exists()

        # Load in new store instance
        store2 = AliasStore(path=store_path)
        assert store2.group_count == 2
        assert store2.total_aliases == 3
        aliases = store2.all_aliases()
        assert "alice-gh" in aliases["alice"]
        assert "bobjones" in aliases["bob"]

    def test_save_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "aliases.json"
        store = AliasStore(path=deep_path)
        store.add("alice", "test")
        store.save()
        assert deep_path.exists()

    def test_save_not_dirty_is_noop(self, store, store_path):
        store.save()
        assert not store_path.exists()

    def test_save_clears_dirty(self, store):
        store.add("alice", "test")
        assert store.dirty is True
        store.save()
        assert store.dirty is False

    def test_load_missing_file(self, tmp_path):
        store = AliasStore(path=tmp_path / "nonexistent.json")
        assert store.group_count == 0

    def test_load_corrupted_file(self, store_path):
        store_path.write_text("not valid json{{{")
        store = AliasStore(path=store_path)
        assert store.group_count == 0

    def test_merge(self, store):
        store.add("alice", "existing")
        store.merge({
            "alice": {"new-alias"},
            "bob": {"bob-alias"},
        })
        assert store.group_count == 2
        assert "new-alias" in store.all_aliases()["alice"]
        assert "existing" in store.all_aliases()["alice"]
        assert "bob-alias" in store.all_aliases()["bob"]

    def test_merge_duplicate_not_dirty(self, store):
        store.add("alice", "alias1")
        store.save()  # clears dirty
        store.merge({"alice": {"alias1"}})  # same alias
        assert store.dirty is False

    def test_json_format(self, store, store_path):
        store.add("alice", "b-alias")
        store.add("alice", "a-alias")
        store.save()

        with open(store_path) as f:
            data = json.load(f)
        # Aliases should be sorted in JSON output
        assert data["alice"] == ["a-alias", "b-alias"]
