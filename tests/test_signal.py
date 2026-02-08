"""Tests for Signal term extraction and filtering."""

from __future__ import annotations

from datetime import datetime, timezone

from stigmergy.primitives.signal import Signal, SignalSource, extract_terms


def _signal(content: str, **kwargs) -> Signal:
    defaults = dict(
        content=content,
        source=SignalSource.GITHUB,
        channel="test/repo",
        author="test.user",
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Signal(**defaults)


# ── extract_terms ─────────────────────────────────────────────


class TestExtractTerms:
    def test_basic_extraction(self):
        terms = extract_terms("pricing engine cache invalidation")
        assert "pricing" in terms
        assert "engine" in terms
        assert "cache" in terms
        assert "invalidation" in terms

    def test_lowercase(self):
        terms = extract_terms("Pricing Engine CACHE")
        assert "pricing" in terms
        assert "engine" in terms
        assert "cache" in terms

    def test_strips_punctuation(self):
        terms = extract_terms("error. (fix) [bug] {deploy}")
        assert "error" in terms
        assert "fix" in terms
        assert "bug" in terms
        assert "deploy" in terms

    def test_min_length(self):
        terms = extract_terms("a an to in of is it on")
        assert len(terms) == 0

    def test_preserves_domain_keywords(self):
        """Short but important domain keywords must survive filtering."""
        terms = extract_terms("bug fix api error deploy merge test code")
        assert "bug" in terms
        assert "fix" in terms
        assert "api" in terms
        assert "error" in terms
        assert "deploy" in terms

    def test_filters_stopwords(self):
        terms = extract_terms("the and for with this that from are")
        assert len(terms) == 0

    def test_filters_urls(self):
        terms = extract_terms(
            "see https://github.com/acme-org/backend/pull/17790 for details"
        )
        assert not any("github.com" in t for t in terms)
        assert not any("://" in t for t in terms)
        assert "details" in terms

    def test_filters_html(self):
        terms = extract_terms("<details> <summary>release notes</summary> </details>")
        assert "details" not in terms  # stripped to empty or too short
        assert "release" in terms
        assert "notes" in terms

    def test_filters_markdown_bold(self):
        terms = extract_terms("**high risk** **low risk**")
        assert "high" in terms
        assert "risk" in terms
        assert not any("**" in t for t in terms)

    def test_filters_code_fences(self):
        terms = extract_terms("```typescript const foo = bar ```")
        assert "typescript" in terms
        assert "const" in terms
        # backtick fences stripped
        assert "```" not in terms
        assert "```typescript" not in terms

    def test_filters_version_numbers(self):
        terms = extract_terms("bumps 5.102.1 to 5.105.0 and 7.7.3")
        assert "bumps" in terms
        assert "5.102.1" not in terms
        assert "5.105.0" not in terms
        assert "7.7.3" not in terms

    def test_filters_pure_numbers(self):
        terms = extract_terms("detected 372 errors in 104 files")
        assert "detected" in terms
        assert "errors" in terms
        assert "files" in terms
        assert "372" not in terms
        assert "104" not in terms

    def test_filters_backtick_refs(self):
        terms = extract_terms("`invoice_transactions` and `console.error`")
        assert "invoice_transactions" in terms  # stripped of backticks
        assert not any("`" in t for t in terms)

    def test_filters_html_comments(self):
        """HTML comments are stripped entirely — their content is not visible."""
        terms = extract_terms("<!-- Mark the appropriate option -->")
        assert "-->" not in terms
        assert "<!--" not in terms
        # Content inside HTML comments is stripped along with the tags
        assert len(terms) == 0

    def test_filters_markdown_rules(self):
        terms = extract_terms("--- section break ---")
        assert "---" not in terms
        assert "section" in terms
        assert "break" in terms

    def test_mixed_real_signal(self):
        """Real-world signal content should produce clean domain terms."""
        content = (
            "PR #17790: simple api for the poc ai page builder\n"
            "## Type of Change\n"
            "<!-- Mark the appropriate option -->"
        )
        terms = extract_terms(content)
        # Domain terms preserved
        assert "simple" in terms
        assert "api" in terms
        assert "page" in terms
        assert "builder" in terms
        # Junk filtered
        assert "the" not in terms
        assert "##" not in terms
        assert "<!--" not in terms
        assert "-->" not in terms

    def test_no_pure_punctuation(self):
        terms = extract_terms("--- ... *** ### >>>")
        assert len(terms) == 0


# ── Signal.terms property ────────────────────────────────────


class TestSignalTerms:
    def test_terms_from_content(self):
        sig = _signal(content="pricing engine cache invalidation strategy")
        assert "pricing" in sig.terms
        assert "engine" in sig.terms
        assert "cache" in sig.terms

    def test_terms_stable(self):
        """Frozen model — terms should be the same on repeated access."""
        sig = _signal(content="deploy release production")
        assert sig.terms == sig.terms

    def test_terms_clean_for_domain_inference(self):
        """Terms should match domain hint keywords without noise."""
        sig = _signal(content="fix critical bug in production deploy")
        terms = sig.terms
        assert "fix" in terms
        assert "bug" in terms
        assert "deploy" in terms
        assert "critical" in terms
        assert "production" in terms
        # stopword filtered
        assert "in" not in terms  # 2 chars, already filtered by length
