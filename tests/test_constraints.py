"""Tests for constraint evaluation.

Kill patterns → message null-routed.
Redact patterns → message delivered with sensitive content removed.
Pass → message delivered as-is.
"""

from __future__ import annotations

import pytest

from stigmergy.constraints.filter import ConstraintAction, ConstraintEvaluator


@pytest.fixture
def evaluator():
    return ConstraintEvaluator.from_config("config/constraints.yaml")


class TestKillPatterns:
    def test_kills_ssn(self, evaluator):
        result = evaluator.evaluate("User SSN is 123-45-6789")
        assert result.action == ConstraintAction.KILL
        assert result.category == "pii_ssn"

    def test_kills_email(self, evaluator):
        result = evaluator.evaluate("Contact them at john.doe@company.com")
        assert result.action == ConstraintAction.KILL
        assert result.category == "pii_email"

    def test_kills_credit_card(self, evaluator):
        result = evaluator.evaluate("Card number: 4111 1111 1111 1111")
        assert result.action == ConstraintAction.KILL
        assert result.category == "pii_credit_card"

    def test_kills_api_key(self, evaluator):
        result = evaluator.evaluate("api_key = sk_test_FAKE000000000000000000000")
        assert result.action == ConstraintAction.KILL
        assert result.category == "credentials"

    def test_kills_bearer_token(self, evaluator):
        result = evaluator.evaluate("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        assert result.action == ConstraintAction.KILL
        assert result.category == "credentials"

    def test_kills_compensation(self, evaluator):
        result = evaluator.evaluate("Their salary is: $150,000")
        assert result.action == ConstraintAction.KILL
        assert result.category == "financial_compensation"

    def test_kills_termination_reference(self, evaluator):
        result = evaluator.evaluate("Discussed termination of John")
        assert result.action == ConstraintAction.KILL
        assert result.category == "hr_sensitive"


class TestRedactPatterns:
    def test_redacts_phone(self, evaluator):
        result = evaluator.evaluate("Call me at 555-123-4567 for details")
        assert result.action == ConstraintAction.REDACT
        assert "[PHONE REDACTED]" in result.content
        assert "555-123-4567" not in result.content

    def test_redacts_address(self, evaluator):
        result = evaluator.evaluate("Office is at 123 Main St downtown")
        assert result.action == ConstraintAction.REDACT
        assert "[ADDRESS REDACTED]" in result.content


class TestPassthrough:
    def test_clean_content_passes(self, evaluator):
        result = evaluator.evaluate("The booking sync is failing for Guesty properties")
        assert result.action == ConstraintAction.PASS
        assert result.content == "The booking sync is failing for Guesty properties"

    def test_technical_content_passes(self, evaluator):
        result = evaluator.evaluate("PR #487 fixes the availability cache invalidation race condition")
        assert result.action == ConstraintAction.PASS

    def test_empty_content_passes(self, evaluator):
        result = evaluator.evaluate("")
        assert result.action == ConstraintAction.PASS


class TestPriority:
    def test_kill_takes_priority_over_redact(self, evaluator):
        """If content matches both kill and redact, kill wins."""
        result = evaluator.evaluate("SSN: 123-45-6789, call 555-123-4567")
        assert result.action == ConstraintAction.KILL


class TestEmptyConfig:
    def test_no_patterns_passes_everything(self):
        evaluator = ConstraintEvaluator()
        result = evaluator.evaluate("SSN: 123-45-6789")
        assert result.action == ConstraintAction.PASS
