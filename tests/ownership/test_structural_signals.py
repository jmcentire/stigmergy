"""Tests for complement-coded structural signal construction."""

import numpy as np
import pytest

from stigmergy.ownership.structural_signals import (
    complement_code,
    elliptic_transaction_to_structural_vector,
    _clip,
    _sigmoid,
    _tanh_norm,
)
from stigmergy.ownership.wallet_signals import (
    wallet_to_structural_vector,
    COEFF_NAMES,
)


class TestComplementCoding:
    def test_doubles_dimensionality(self):
        a = [0.3, 0.7, 0.5]
        cc = complement_code(a)
        assert len(cc) == 6

    def test_sum_to_one(self):
        a = [0.1, 0.5, 0.9, 0.0, 1.0]
        cc = complement_code(a)
        for i in range(len(a)):
            assert abs(cc[i] + cc[i + len(a)] - 1.0) < 1e-10

    def test_bounded(self):
        a = [0.0, 0.5, 1.0]
        cc = complement_code(a)
        assert all(0.0 <= v <= 1.0 for v in cc)

    def test_empty(self):
        assert complement_code([]) == []


class TestNormalization:
    def test_clip_bounds(self):
        assert _clip(-0.5) == 0.0
        assert _clip(1.5) == 1.0
        assert _clip(0.5) == 0.5

    def test_sigmoid_midpoint(self):
        assert abs(_sigmoid(5.0, midpoint=5.0) - 0.5) < 0.01

    def test_sigmoid_bounds(self):
        assert _sigmoid(-100, midpoint=0) < 0.01
        assert _sigmoid(100, midpoint=0) > 0.99

    def test_tanh_norm_center(self):
        assert abs(_tanh_norm(0.0) - 0.5) < 0.01


class TestTransactionVector:
    def _make_row(self, **overrides):
        row = {"txId": 1, "Time step": 1}
        for i in range(1, 95):
            row[f"Local_feature_{i}"] = 0.0
        for i in range(1, 73):
            row[f"Aggregate_feature_{i}"] = 0.0
        row.update(overrides)
        return row

    def test_output_length(self):
        sv = elliptic_transaction_to_structural_vector(self._make_row(), 0, 0)
        assert len(sv) == 7

    def test_bounded(self):
        sv = elliptic_transaction_to_structural_vector(self._make_row(), 5, 3)
        assert all(0.0 <= v <= 1.0 for v in sv)

    def test_complement_coded_full(self):
        sv = elliptic_transaction_to_structural_vector(self._make_row(), 2, 4)
        cc = complement_code(sv)
        assert len(cc) == 14
        for i in range(7):
            assert abs(cc[i] + cc[i + 7] - 1.0) < 1e-10

    def test_high_fanout(self):
        sv = elliptic_transaction_to_structural_vector(self._make_row(), 0, 20)
        # Fragmentation should be high with 20 outputs
        assert sv[0] > 0.5  # f_o
        # Fan ratio should indicate fan-out
        assert sv[5] > 0.5  # t_f

    def test_relay_pattern(self):
        sv = elliptic_transaction_to_structural_vector(self._make_row(), 3, 3)
        # Balanced in/out = relay pattern
        assert sv[1] > 0.1  # n_p should be nonzero for balanced relay

    def test_zero_degree(self):
        sv = elliptic_transaction_to_structural_vector(self._make_row(), 0, 0)
        assert sv[0] >= 0.0  # should not crash
        assert sv[5] == 0.5  # fan ratio = neutral when no edges


class TestWalletVector:
    def _make_wallet(self, **overrides):
        row = {
            "address": "1ABC",
            "Time step": 10,
            "num_txs_as_sender": 5,
            "num_txs_as receiver": 10,
            "first_block_appeared_in": 400000,
            "last_block_appeared_in": 500000,
            "lifetime_in_blocks": 100000,
            "total_txs": 15,
            "btc_transacted_total": 10.0,
            "btc_transacted_min": 0.01,
            "btc_transacted_max": 5.0,
            "btc_transacted_mean": 0.67,
            "btc_transacted_median": 0.1,
            "btc_sent_total": 3.0,
            "btc_received_total": 7.0,
            "fees_total": 0.01,
            "fees_mean": 0.001,
            "fees_as_share_mean": 0.001,
            "blocks_btwn_txs_mean": 5000,
            "blocks_btwn_txs_max": 20000,
            "blocks_btwn_txs_median": 4000,
            "transacted_w_address_total": 10,
            "num_addr_transacted_multiple": 2,
            "num_timesteps_appeared_in": 5,
        }
        row.update(overrides)
        return row

    def test_output_length(self):
        sv = wallet_to_structural_vector(self._make_wallet())
        assert len(sv) == 7

    def test_bounded(self):
        sv = wallet_to_structural_vector(self._make_wallet())
        assert all(0.0 <= v <= 1.0 for v in sv)

    def test_coefficient_names(self):
        assert len(COEFF_NAMES) == 7

    def test_pure_receiver(self):
        sv = wallet_to_structural_vector(self._make_wallet(
            num_txs_as_sender=0, total_txs=10
        ))
        assert sv[0] == 0.0  # asymmetry: pure receiver

    def test_pure_sender(self):
        sv = wallet_to_structural_vector(self._make_wallet(
            num_txs_as_sender=10, **{"num_txs_as receiver": 0}, total_txs=10
        ))
        assert sv[0] == 1.0  # asymmetry: pure sender

    def test_high_volume_concentration(self):
        sv = wallet_to_structural_vector(self._make_wallet(
            btc_transacted_max=100.0, btc_transacted_mean=0.1
        ))
        assert sv[1] > 0.5  # volume concentration should be high

    def test_regular_timing(self):
        sv = wallet_to_structural_vector(self._make_wallet(
            blocks_btwn_txs_median=9000, blocks_btwn_txs_max=10000
        ))
        assert sv[2] > 0.5  # temporal regularity should be high

    def test_no_repeat_counterparties(self):
        sv = wallet_to_structural_vector(self._make_wallet(
            num_addr_transacted_multiple=0
        ))
        assert sv[6] == 0.0  # repeat ratio = 0
