"""Linguistic compression detection (Paper Section 5.3).

Two paths to the same data structures:

1. AGENT PATH (primary): The correlator and indexer agents assess
   linguistic quality as part of their normal work. They read the text
   and exercise judgment — they know what hedging sounds like, what
   bureaucratic substitution is, whether language is forward-looking.
   Their assessments populate CompressionSignals via from_agent().

2. REGEX PATH (fallback): When no LLM is available (stub mode, budget
   exhausted), regex heuristics provide a baseline. These are the old
   approach — word lists, suffix matching, pattern detection. They run
   free and fast but miss context that agents understand.

The CompressionTracker doesn't care which path produced the values.
It tracks trends over time regardless of source. The output dashboard
renders whatever the tracker has.

Extended metrics from the variance compression research:
- Lexical Diversity (LD): unique tokens / total tokens (analysis.tex)
- Shannon Entropy (SE): base-2 information entropy (analysis.tex)
- Temporal Orientation: forward-looking vs backward-looking language
- Bureaucratic Substitution: euphemistic lexical narrowing
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ── Hedge phrases ──────────────────────────────────────────

_HEDGE_PHRASES = [
    "may", "might", "could potentially", "it appears", "seems to",
    "is expected to", "under investigation", "possibly", "perhaps",
    "it seems", "appears to", "likely", "unlikely", "probably",
    "to some extent", "in some cases", "tends to", "somewhat",
    "it is believed", "it is thought", "it is possible",
]

# Pre-compile a pattern that matches any hedge phrase as whole words
_HEDGE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(h) for h in _HEDGE_PHRASES) + r")\b",
    re.IGNORECASE,
)

# ── Passive voice ──────────────────────────────────────────

# Matches: was/were/been/being/is/are/has been/have been + past participle
# Past participle approximation: word ending in -ed, -en, -t (common patterns)
_PASSIVE_RE = re.compile(
    r"\b(?:was|were|been|being|is|are|has\s+been|have\s+been|had\s+been)"
    r"\s+(?:\w+ly\s+)?(\w+(?:ed|en|t|wn|ne))\b",
    re.IGNORECASE,
)

# ── Nominalization ─────────────────────────────────────────

# Words ending in -tion, -ment, -ness, -ity, -ence, -ance that likely
# derive from verbs. Simple suffix heuristic — no stemming needed.
_NOMINALIZATION_RE = re.compile(
    r"\b\w{4,}(?:tion|ment|ness|ity|ence|ance|sion)\b",
    re.IGNORECASE,
)

# ── Specificity ────────────────────────────────────────────

# Patterns indicating concrete, specific language
_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # 2024-01-15
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}"  # January 15
    r"|\d{1,2}/\d{1,2}/\d{2,4})\b",  # 01/15/2024
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_DIRECT_VERB_RE = re.compile(
    r"\b(?:will|must|fixed|deployed|completed|resolved|merged|shipped|released|closed|done)\b",
    re.IGNORECASE,
)

# ── Temporal orientation ──────────────────────────────────
# Forward-looking: commitment to future action (the aerospace paper's
# "specificity as commitment device" — Kennedy's concrete deadlines)
_FORWARD_PHRASES = [
    "will", "going to", "plan to", "intend to", "aim to",
    "next week", "next month", "next quarter", "next sprint",
    "tomorrow", "upcoming", "goal is", "target is", "deadline",
    "by friday", "by monday", "by end of", "ship by", "launch by",
    "we will", "we'll", "we plan", "we intend", "we aim",
    "scheduled for", "on track to", "expected to ship",
]

_BACKWARD_PHRASES = [
    "was", "were", "had been", "used to", "previously",
    "last week", "last month", "last quarter", "last sprint",
    "yesterday", "formerly", "historically", "in the past",
    "we had", "we used to", "we were", "it was",
    "back when", "at that time", "prior to",
]

_FORWARD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _FORWARD_PHRASES) + r")\b",
    re.IGNORECASE,
)

_BACKWARD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _BACKWARD_PHRASES) + r")\b",
    re.IGNORECASE,
)

# ── Bureaucratic substitution (aerospace paper, NASA vocabulary) ──
# Euphemistic replacements that narrow the semantic field and remove
# evaluative content. Each tuple: (bureaucratic_term, direct_equivalent)
# Detection: presence of the bureaucratic form indicates lexical narrowing.
_BUREAUCRATIC_TERMS = [
    # NASA vocabulary table (aerospace paper page 2)
    "mishap", "anomaly", "contingency", "re-plan", "nominal",
    # Extended bureaucratic euphemisms common in tech organizations
    "action item", "going forward", "circle back", "take offline",
    "align on", "synergize", "leverage", "operationalize",
    "socialize", "double-click on", "deep dive",
    "stakeholder", "deliverable", "bandwidth",
    "parking lot", "boil the ocean", "move the needle",
    # Softening language (Moore's Brown-Levinson politeness strategies)
    "it would be helpful if", "we might want to consider",
    "there may be an opportunity", "it could be beneficial",
    "we should perhaps", "it might make sense to",
]

_BUREAUCRATIC_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _BUREAUCRATIC_TERMS) + r")\b",
    re.IGNORECASE,
)

# ── Tokenization ──────────────────────────────────────────
# Simple word tokenizer for LD and SE computation.
# Matches sequences of word characters — no stemming, no lemmatization.
_TOKEN_RE = re.compile(r"\b[a-zA-Z]+\b")

# ── Sentence splitting ─────────────────────────────────────

_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Returns non-empty sentences."""
    parts = _SENTENCE_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


# ── Core data structure ────────────────────────────────────


@dataclass
class CompressionSignals:
    """Linguistic compression indicators for a text sample.

    Original four metrics (Paper Section 5.3):
        hedging_density, passive_voice_ratio, nominalization_rate, specificity_score

    Extended metrics (variance compression research):
        lexical_diversity: LD = unique tokens / total tokens (analysis.tex)
        shannon_entropy: SE in bits, base-2 (analysis.tex)
        temporal_orientation: -1 (backward) to +1 (forward), 0 = balanced
        bureaucratic_density: fraction of sentences with bureaucratic euphemisms

    Reference ranges from analysis.tex (SEC filings):
        S-1 filings (pre-cage): LD 0.16-0.19, SE 11-12.5
        10-K filings (caged):   LD 0.13-0.15, SE 10-11
        Apple Jobs aggregate:   LD 0.175 (High)
        Apple Cook aggregate:   LD 0.143 (Low)
    """

    hedging_density: float       # fraction of sentences with hedge words
    passive_voice_ratio: float   # fraction of clauses in passive voice
    nominalization_rate: float   # nominalizations per sentence
    specificity_score: float     # concrete commitments vs vague language
    compression_index: float     # weighted composite [0, 1]
    # Extended metrics
    lexical_diversity: float = 0.0    # unique tokens / total tokens
    shannon_entropy: float = 0.0      # bits (base 2)
    temporal_orientation: float = 0.0  # -1 backward to +1 forward
    bureaucratic_density: float = 0.0  # fraction of sentences with bureaucratic terms
    source: str = "regex"             # "regex" or "agent" — which path produced these values

    @classmethod
    def from_agent(
        cls,
        *,
        compression_level: float = 0.0,
        temporal_direction: float = 0.0,
        bureaucratic_index: float = 0.0,
        exploration_signal: float = 0.0,
        text: str = "",
    ) -> "CompressionSignals":
        """Create CompressionSignals from agent-assessed values.

        The agent provides batch-level assessments: compression_level,
        temporal_direction, bureaucratic_index. These map directly to our
        fields. The agent's judgment replaces regex heuristics.

        If text is provided, LD and SE are still computed mechanically
        (they're pure math, not judgment) to complement the agent's
        qualitative assessment with quantitative baselines.
        """
        # LD and SE are pure computation — always compute if text available
        ld = 0.0
        se = 0.0
        if text:
            tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
            ld = _compute_lexical_diversity(tokens)
            se = _compute_shannon_entropy(tokens)

        return cls(
            hedging_density=0.0,        # not decomposed by agent
            passive_voice_ratio=0.0,    # not decomposed by agent
            nominalization_rate=0.0,    # not decomposed by agent
            specificity_score=1.0 - compression_level,  # inverse proxy
            compression_index=round(compression_level, 4),
            lexical_diversity=round(ld, 4),
            shannon_entropy=round(se, 4),
            temporal_orientation=round(temporal_direction, 4),
            bureaucratic_density=round(bureaucratic_index, 4),
            source="agent",
        )


def _compute_lexical_diversity(tokens: list[str]) -> float:
    """Lexical Diversity: unique tokens / total tokens.

    Directly comparable to analysis.tex LD metric used on SEC filings.
    Reference: Amazon S-1 LD=0.1742, 10-K Y2 LD=0.1415 (17% compression).
    """
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _compute_shannon_entropy(tokens: list[str]) -> float:
    """Shannon Entropy in bits (base 2).

    Measures information unpredictability. High SE = varied, unpredictable
    vocabulary. Low SE = formulaic, dominated by recurring terms.
    Reference: Google S-1 SE=12.05, 10-K Y2 SE=11.10 (analysis.tex).

    Note: SE scales with vocabulary size. For Slack messages (~50-200 tokens)
    expect SE in range 4-8. For full documents (~5000+ tokens) expect 10-13.
    The relative change over time is the meaningful signal, not the absolute value.
    """
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_temporal_orientation(sentences: list[str]) -> float:
    """Temporal orientation: forward-looking vs backward-looking language.

    Returns value in [-1, +1]:
        +1 = all forward-looking (commitment, planning, future deadlines)
        -1 = all backward-looking (describing past, retrospective)
         0 = balanced or no temporal markers

    From the aerospace paper: organizations that increasingly describe past
    achievements rather than future goals are compressing. Kennedy's language
    was overwhelmingly forward-looking; post-Columbia NASA increasingly
    backward-looking.

    From LIWC methodology: temporal orientation analysis reveals whether
    NASA increasingly described past achievements rather than future goals.
    """
    forward = sum(1 for s in sentences if _FORWARD_RE.search(s))
    backward = sum(1 for s in sentences if _BACKWARD_RE.search(s))
    total = forward + backward
    if total == 0:
        return 0.0
    # Normalize to [-1, +1]: (forward - backward) / total
    return (forward - backward) / total


def analyze_text(text: str) -> CompressionSignals:
    """Analyze text for linguistic compression signals.

    Returns a CompressionSignals dataclass with individual metrics and
    a composite compression index (0 = direct/specific, 1 = compressed/vague).

    Extended metrics include Lexical Diversity, Shannon Entropy, temporal
    orientation, and bureaucratic density — connecting the tool's output
    directly to the published empirical data on variance compression.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return CompressionSignals(
            hedging_density=0.0,
            passive_voice_ratio=0.0,
            nominalization_rate=0.0,
            specificity_score=1.0,
            compression_index=0.0,
            lexical_diversity=0.0,
            shannon_entropy=0.0,
            temporal_orientation=0.0,
            bureaucratic_density=0.0,
        )

    n_sentences = len(sentences)

    # Hedging: fraction of sentences containing hedge phrases
    hedged_count = sum(1 for s in sentences if _HEDGE_RE.search(s))
    hedging_density = hedged_count / n_sentences

    # Passive voice: fraction of sentences with passive constructions
    passive_count = sum(1 for s in sentences if _PASSIVE_RE.search(s))
    passive_voice_ratio = passive_count / n_sentences

    # Nominalization: average nominalizations per sentence
    total_noms = sum(len(_NOMINALIZATION_RE.findall(s)) for s in sentences)
    nominalization_rate = min(total_noms / n_sentences, 1.0)  # cap at 1.0

    # Specificity: fraction of sentences with concrete markers
    specific_count = sum(
        1 for s in sentences
        if _DATE_RE.search(s) or _NUMBER_RE.search(s) or _DIRECT_VERB_RE.search(s)
    )
    specificity_score = specific_count / n_sentences

    # Compression index: weighted composite (original four-component formula)
    compression_index = (
        0.3 * hedging_density
        + 0.3 * passive_voice_ratio
        + 0.2 * nominalization_rate
        + 0.2 * (1.0 - specificity_score)
    )
    compression_index = max(0.0, min(1.0, compression_index))

    # ── Extended metrics ──────────────────────────────────

    # Tokenize the full text for LD and SE
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]

    lexical_diversity = _compute_lexical_diversity(tokens)
    shannon_entropy = _compute_shannon_entropy(tokens)
    temporal_orientation = _compute_temporal_orientation(sentences)

    # Bureaucratic density: fraction of sentences with bureaucratic terms
    bureaucratic_count = sum(1 for s in sentences if _BUREAUCRATIC_RE.search(s))
    bureaucratic_density = bureaucratic_count / n_sentences

    return CompressionSignals(
        hedging_density=round(hedging_density, 4),
        passive_voice_ratio=round(passive_voice_ratio, 4),
        nominalization_rate=round(nominalization_rate, 4),
        specificity_score=round(specificity_score, 4),
        compression_index=round(compression_index, 4),
        lexical_diversity=round(lexical_diversity, 4),
        shannon_entropy=round(shannon_entropy, 4),
        temporal_orientation=round(temporal_orientation, 4),
        bureaucratic_density=round(bureaucratic_density, 4),
    )


# ── Rolling average tracker ───────────────────────────────


@dataclass
class CompressionTrend:
    """Compression trend for a single topic or channel.

    Extended with "Born Caged" detection (analysis.tex Section 3):
    initial_index captures the first observed compression level. If
    initial_index is already high, the channel was "born caged" —
    it started with compressed language, like modern S-1 filings
    that already embed defensive boilerplate before the IPO.
    """

    key: str                      # topic term or channel name
    sample_count: int
    current_index: float          # most recent compression index
    rolling_avg: float            # exponential moving average
    trend: str                    # "increasing", "stable", "decreasing"
    peak_index: float             # highest compression seen
    initial_index: float = 0.0    # first observed value ("Born Caged" baseline)
    current_ld: float = 0.0       # most recent lexical diversity
    current_se: float = 0.0       # most recent Shannon entropy

    @property
    def compression_delta(self) -> float:
        """Change from initial to current. Positive = compressing."""
        return self.current_index - self.initial_index

    @property
    def is_born_caged(self) -> bool:
        """True if the initial observation was already compressed.

        Threshold 0.3 based on analysis.tex: mature 10-K filings
        show compression_index ~0.3+. Channels that start above
        this were "born caged" — their language was already formalized.
        """
        return self.initial_index >= 0.3


class CompressionTracker:
    """Tracks per-topic and per-channel compression over time.

    Uses exponential moving average (EMA) to smooth noise. Detects
    trends by comparing recent EMA to historical EMA.

    Extended with:
    - LD/SE tracking alongside compression_index
    - "Born Caged" baseline (initial observation recorded)
    """

    def __init__(self, *, alpha: float = 0.3) -> None:
        self._alpha = alpha  # EMA smoothing factor (higher = more recent weight)
        self._ema: dict[str, float] = {}        # key -> current EMA
        self._counts: dict[str, int] = {}       # key -> sample count
        self._peaks: dict[str, float] = {}      # key -> peak index
        self._prev_ema: dict[str, float] = {}   # key -> previous EMA (for trend)
        self._initial: dict[str, float] = {}    # key -> first observed value
        self._ld: dict[str, float] = {}         # key -> most recent LD
        self._se: dict[str, float] = {}         # key -> most recent SE

    def update(self, key: str, compression_index: float,
               *, ld: float = 0.0, se: float = 0.0) -> None:
        """Record a new compression measurement for a key (topic or channel).

        Args:
            key: topic term or channel name
            compression_index: composite compression score [0, 1]
            ld: lexical diversity for this sample (optional)
            se: Shannon entropy for this sample (optional)
        """
        self._ld[key] = ld
        self._se[key] = se

        if key not in self._ema:
            self._ema[key] = compression_index
            self._counts[key] = 1
            self._peaks[key] = compression_index
            self._prev_ema[key] = compression_index
            self._initial[key] = compression_index  # "Born Caged" baseline
        else:
            self._prev_ema[key] = self._ema[key]
            self._ema[key] = self._alpha * compression_index + (1 - self._alpha) * self._ema[key]
            self._counts[key] += 1
            self._peaks[key] = max(self._peaks[key], compression_index)

    def get_trend(self, key: str) -> CompressionTrend | None:
        """Get compression trend for a key. Returns None if no data."""
        if key not in self._ema:
            return None

        ema = self._ema[key]
        prev = self._prev_ema[key]
        delta = ema - prev

        if delta > 0.02:
            trend = "increasing"
        elif delta < -0.02:
            trend = "decreasing"
        else:
            trend = "stable"

        return CompressionTrend(
            key=key,
            sample_count=self._counts[key],
            current_index=round(ema, 4),
            rolling_avg=round(ema, 4),
            trend=trend,
            peak_index=round(self._peaks[key], 4),
            initial_index=round(self._initial.get(key, 0.0), 4),
            current_ld=round(self._ld.get(key, 0.0), 4),
            current_se=round(self._se.get(key, 0.0), 4),
        )

    def all_trends(self, *, min_samples: int = 2) -> list[CompressionTrend]:
        """Get all trends with enough samples, sorted by current_index descending."""
        trends = []
        for key in self._ema:
            if self._counts[key] >= min_samples:
                t = self.get_trend(key)
                if t is not None:
                    trends.append(t)
        trends.sort(key=lambda t: t.current_index, reverse=True)
        return trends

    def increasing_trends(self, *, min_samples: int = 2) -> list[CompressionTrend]:
        """Get only increasing trends — these are the concerning ones."""
        return [t for t in self.all_trends(min_samples=min_samples)
                if t.trend == "increasing"]

    def born_caged_channels(self, *, min_samples: int = 2) -> list[CompressionTrend]:
        """Get channels that started with already-compressed language.

        These channels were "born caged" — their language was formalized
        from the first observation. This is the analysis.tex "Born Caged"
        effect: modern organizations whose communication is pre-compressed
        before any external event triggers the compression.
        """
        return [t for t in self.all_trends(min_samples=min_samples)
                if t.is_born_caged]
