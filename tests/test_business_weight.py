"""Tests for business weight computation."""

import math

import pytest

from stigmergy.core.business_weight import (
    BusinessWeightConfig,
    BusinessWeightMetrics,
    MetricUpdateRequest,
    apply_metric_update,
    compute_business_weight,
    create_default_metrics,
    get_effective_business_weight,
    validate_metrics,
)


class TestCreateDefaultMetrics:
    def test_all_neutral(self):
        m = create_default_metrics()
        assert m.signal_frequency == 0.5
        assert m.author_seniority == 0.5
        assert m.channel_gravity == 0.5
        assert m.engagement_velocity == 0.5


class TestComputeBusinessWeight:
    def test_neutral_metrics_equal_weights(self):
        m = create_default_metrics()
        result = compute_business_weight(m)
        assert result.raw_weight == pytest.approx(0.5)
        assert result.effective_weight == pytest.approx(0.5)

    def test_all_high(self):
        m = BusinessWeightMetrics(
            signal_frequency=1.0,
            author_seniority=1.0,
            channel_gravity=1.0,
            engagement_velocity=1.0,
        )
        result = compute_business_weight(m)
        assert result.raw_weight == pytest.approx(1.0)
        assert result.effective_weight == pytest.approx(1.0)

    def test_all_low(self):
        m = BusinessWeightMetrics(
            signal_frequency=0.0,
            author_seniority=0.0,
            channel_gravity=0.0,
            engagement_velocity=0.0,
        )
        result = compute_business_weight(m)
        assert result.raw_weight == pytest.approx(0.0)
        # Floor at 0.1
        assert result.effective_weight == pytest.approx(0.1)

    def test_custom_weights(self):
        m = BusinessWeightMetrics(signal_frequency=1.0, author_seniority=0.0,
                                   channel_gravity=0.0, engagement_velocity=0.0)
        config = BusinessWeightConfig(
            weight_signal_frequency=1.0,
            weight_author_seniority=0.0,
            weight_channel_gravity=0.0,
            weight_engagement_velocity=0.0,
        )
        result = compute_business_weight(m, config)
        assert result.raw_weight == pytest.approx(1.0)

    def test_all_zero_weights(self):
        m = create_default_metrics()
        config = BusinessWeightConfig(
            weight_signal_frequency=0.0,
            weight_author_seniority=0.0,
            weight_channel_gravity=0.0,
            weight_engagement_velocity=0.0,
        )
        result = compute_business_weight(m, config)
        assert result.raw_weight == pytest.approx(0.5)  # Neutral default

    def test_policy_floor(self):
        m = BusinessWeightMetrics(
            signal_frequency=0.05,
            author_seniority=0.05,
            channel_gravity=0.05,
            engagement_velocity=0.05,
        )
        result = compute_business_weight(m)
        assert result.raw_weight == pytest.approx(0.05)
        assert result.effective_weight == pytest.approx(0.1)  # Floor

    def test_custom_policy_floor(self):
        m = BusinessWeightMetrics(
            signal_frequency=0.0,
            author_seniority=0.0,
            channel_gravity=0.0,
            engagement_velocity=0.0,
        )
        config = BusinessWeightConfig(policy_floor=0.3)
        result = compute_business_weight(m, config)
        assert result.effective_weight == pytest.approx(0.3)

    def test_nan_metric_raises(self):
        m = BusinessWeightMetrics()
        m.signal_frequency = float("nan")
        with pytest.raises(ValueError):
            compute_business_weight(m)

    def test_equal_metrics_equal_raw(self):
        """If all metrics equal X, raw_weight == X."""
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            m = BusinessWeightMetrics(
                signal_frequency=x,
                author_seniority=x,
                channel_gravity=x,
                engagement_velocity=x,
            )
            result = compute_business_weight(m)
            assert result.raw_weight == pytest.approx(x)


class TestApplyMetricUpdate:
    def test_partial_update(self):
        m = create_default_metrics()
        update = MetricUpdateRequest(signal_frequency=0.9)
        result = apply_metric_update(m, update)
        assert result.signal_frequency == 0.9
        assert result.author_seniority == 0.5  # unchanged
        assert result.channel_gravity == 0.5  # unchanged
        assert result.engagement_velocity == 0.5  # unchanged

    def test_no_mutation(self):
        m = create_default_metrics()
        update = MetricUpdateRequest(signal_frequency=0.9)
        apply_metric_update(m, update)
        assert m.signal_frequency == 0.5  # original unchanged

    def test_full_update(self):
        m = create_default_metrics()
        update = MetricUpdateRequest(
            signal_frequency=0.1,
            author_seniority=0.2,
            channel_gravity=0.3,
            engagement_velocity=0.4,
        )
        result = apply_metric_update(m, update)
        assert result.signal_frequency == 0.1
        assert result.author_seniority == 0.2
        assert result.channel_gravity == 0.3
        assert result.engagement_velocity == 0.4

    def test_out_of_range_raises(self):
        m = create_default_metrics()
        with pytest.raises(ValueError):
            apply_metric_update(m, MetricUpdateRequest(signal_frequency=1.5))

    def test_nan_raises(self):
        m = create_default_metrics()
        with pytest.raises(ValueError):
            apply_metric_update(m, MetricUpdateRequest(signal_frequency=float("nan")))


class TestGetEffectiveBusinessWeight:
    def test_above_floor(self):
        assert get_effective_business_weight(0.5) == 0.5

    def test_below_floor(self):
        assert get_effective_business_weight(0.05) == 0.1

    def test_at_floor(self):
        assert get_effective_business_weight(0.1) == 0.1

    def test_custom_floor(self):
        assert get_effective_business_weight(0.2, policy_floor=0.5) == 0.5

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            get_effective_business_weight(float("nan"))

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            get_effective_business_weight(1.5)


class TestValidateMetrics:
    def test_valid(self):
        assert validate_metrics(create_default_metrics()) is True

    def test_out_of_range(self):
        m = BusinessWeightMetrics()
        m.signal_frequency = 1.5
        with pytest.raises(ValueError):
            validate_metrics(m)
