"""Tests for the InquiryResult primitive and its factory functions."""

import json

import pytest
from pydantic import ValidationError

from stigmergy.primitives.inquiry import (
    InquiryDimension,
    InquiryResult,
    create_inquiry_result,
    deserialize_inquiry_result,
    get_active_dimensions,
    get_dimension_confidence,
    get_dimension_connections,
    get_overall_confidence,
    serialize_inquiry_result,
)


class TestInquiryDimension:
    def test_all_variants(self):
        assert set(InquiryDimension) == {
            InquiryDimension.PEOPLE,
            InquiryDimension.PROJECTS,
            InquiryDimension.TICKETS,
            InquiryDimension.KEYWORDS,
            InquiryDimension.SENTIMENT,
        }

    def test_string_values(self):
        assert InquiryDimension.PEOPLE == "people"
        assert InquiryDimension.PROJECTS == "projects"


class TestCreateInquiryResult:
    def test_basic_creation(self):
        result = create_inquiry_result(
            signal_id="sig-1",
            context_id="ctx-1",
            dimensions={"people": ["Alice", "Bob"]},
            confidence_scores={"people": 0.85},
        )
        assert result.signal_id == "sig-1"
        assert result.context_id == "ctx-1"
        assert result.dimensions["people"] == ["Alice", "Bob"]
        assert result.confidence_scores["people"] == 0.85

    def test_empty_dimensions(self):
        result = create_inquiry_result(
            signal_id="sig-1",
            context_id="ctx-1",
            dimensions={},
            confidence_scores={},
        )
        assert result.dimensions == {}
        assert result.confidence_scores == {}

    def test_multiple_dimensions(self):
        result = create_inquiry_result(
            signal_id="sig-1",
            context_id="ctx-1",
            dimensions={
                "people": ["Alice"],
                "projects": ["proj-1"],
                "keywords": ["auth", "login"],
            },
            confidence_scores={
                "people": 0.9,
                "projects": 0.7,
                "keywords": 0.6,
            },
        )
        assert len(result.dimensions) == 3
        assert len(result.confidence_scores) == 3

    def test_frozen(self):
        result = create_inquiry_result(
            signal_id="sig-1",
            context_id="ctx-1",
            dimensions={},
            confidence_scores={},
        )
        with pytest.raises(ValidationError):
            result.signal_id = "other"

    def test_invalid_confidence_range(self):
        with pytest.raises(ValidationError):
            create_inquiry_result(
                signal_id="sig-1",
                context_id="ctx-1",
                dimensions={"people": ["A"]},
                confidence_scores={"people": 1.5},
            )

    def test_negative_confidence(self):
        with pytest.raises(ValidationError):
            create_inquiry_result(
                signal_id="sig-1",
                context_id="ctx-1",
                dimensions={"people": ["A"]},
                confidence_scores={"people": -0.1},
            )

    def test_invalid_dimension_key(self):
        with pytest.raises(ValidationError):
            create_inquiry_result(
                signal_id="sig-1",
                context_id="ctx-1",
                dimensions={"invalid_dim": ["A"]},
                confidence_scores={"invalid_dim": 0.5},
            )

    def test_key_mismatch(self):
        with pytest.raises(ValidationError):
            create_inquiry_result(
                signal_id="sig-1",
                context_id="ctx-1",
                dimensions={"people": ["A"]},
                confidence_scores={"projects": 0.5},
            )

    def test_custom_id_and_timestamp(self):
        result = create_inquiry_result(
            signal_id="sig-1",
            context_id="ctx-1",
            dimensions={},
            confidence_scores={},
            id="custom-id",
            timestamp="2026-01-01T00:00:00Z",
        )
        assert result.id == "custom-id"
        assert result.timestamp == "2026-01-01T00:00:00Z"


class TestGetDimensionConnections:
    def test_existing_dimension(self):
        result = create_inquiry_result(
            signal_id="s", context_id="c",
            dimensions={"people": ["Alice", "Bob"]},
            confidence_scores={"people": 0.8},
        )
        assert get_dimension_connections(result, InquiryDimension.PEOPLE) == ["Alice", "Bob"]

    def test_missing_dimension(self):
        result = create_inquiry_result(
            signal_id="s", context_id="c",
            dimensions={}, confidence_scores={},
        )
        assert get_dimension_connections(result, InquiryDimension.PEOPLE) == []


class TestGetDimensionConfidence:
    def test_existing(self):
        result = create_inquiry_result(
            signal_id="s", context_id="c",
            dimensions={"projects": ["p1"]},
            confidence_scores={"projects": 0.75},
        )
        assert get_dimension_confidence(result, InquiryDimension.PROJECTS) == 0.75

    def test_missing(self):
        result = create_inquiry_result(
            signal_id="s", context_id="c",
            dimensions={}, confidence_scores={},
        )
        assert get_dimension_confidence(result, InquiryDimension.TICKETS) == 0.0


class TestGetActiveDimensions:
    def test_multiple_active(self):
        result = create_inquiry_result(
            signal_id="s", context_id="c",
            dimensions={"people": ["A"], "keywords": ["k1"]},
            confidence_scores={"people": 0.8, "keywords": 0.5},
        )
        active = get_active_dimensions(result)
        assert len(active) == 2
        assert InquiryDimension.PEOPLE in active
        assert InquiryDimension.KEYWORDS in active

    def test_empty_values_excluded(self):
        result = create_inquiry_result(
            signal_id="s", context_id="c",
            dimensions={"people": []},
            confidence_scores={"people": 0.0},
        )
        assert get_active_dimensions(result) == []


class TestGetOverallConfidence:
    def test_mean_confidence(self):
        result = create_inquiry_result(
            signal_id="s", context_id="c",
            dimensions={"people": ["A"], "keywords": ["k"]},
            confidence_scores={"people": 0.8, "keywords": 0.6},
        )
        assert get_overall_confidence(result) == pytest.approx(0.7)

    def test_empty_result(self):
        result = create_inquiry_result(
            signal_id="s", context_id="c",
            dimensions={}, confidence_scores={},
        )
        assert get_overall_confidence(result) == 0.0


class TestSerialization:
    def test_round_trip(self):
        original = create_inquiry_result(
            signal_id="sig-1",
            context_id="ctx-1",
            dimensions={"people": ["Alice"], "keywords": ["auth"]},
            confidence_scores={"people": 0.9, "keywords": 0.7},
            id="test-id",
            timestamp="2026-01-01T00:00:00Z",
        )
        data = serialize_inquiry_result(original)
        restored = deserialize_inquiry_result(data)
        assert restored.signal_id == original.signal_id
        assert restored.context_id == original.context_id
        assert restored.dimensions == original.dimensions
        assert restored.confidence_scores == original.confidence_scores

    def test_json_serializable(self):
        result = create_inquiry_result(
            signal_id="s", context_id="c",
            dimensions={"people": ["A"]},
            confidence_scores={"people": 0.5},
        )
        data = serialize_inquiry_result(result)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
