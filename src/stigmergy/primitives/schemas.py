"""Output schemas: contracts that shape agent generation.

These are not validators checking work after the fact — they're
contracts that constrain the LLM's output at generation time via
structured outputs (tool use). The schema IS the guardrail.

Each schema represents one agent decision. The LLM fills it in.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SignalFacts(BaseModel):
    """What an indexer agent extracts from a raw signal.

    This replaces extract_terms(), _infer_domain(), and all the regex/
    stopword/heuristic machinery. The agent reads the signal and tells
    us what it's about — the way a human would.
    """

    concepts: list[str] = Field(
        description=(
            "3-8 domain-relevant concepts from this signal. "
            "These should be specific technical or business terms that "
            "distinguish this signal's domain — not generic words. "
            "Examples: 'pricing engine', 'calendar sync', 'webhook retry', "
            "'booking cancellation'. Include multi-word phrases when they "
            "form a single concept."
        ),
    )
    domain: str = Field(
        description=(
            "Primary domain: 'engineering', 'infrastructure', 'product', "
            "'support', 'organizational', or a self-describing string if "
            "none of those fit."
        ),
    )
    summary: str = Field(
        description=(
            "One sentence: what is this signal about? Be specific. "
            "'PR adding seasonal pricing multipliers to the rate engine' "
            "not 'code change'."
        ),
    )
    actors: list[str] = Field(
        default_factory=list,
        description="People or systems mentioned (usernames, service names).",
    )
    urgency: str = Field(
        default="normal",
        description="'high' if this needs immediate attention, 'normal' otherwise.",
    )
    linguistic_tone: str = Field(
        default="direct",
        description=(
            "The tone of this signal's language. "
            "'direct' = concrete, specific, names things plainly. "
            "'hedged' = softened, qualified, lots of 'may', 'might', 'could'. "
            "'bureaucratic' = process-heavy, euphemistic, standardized phrasing. "
            "'compressed' = vague, nominal, passive — avoids direct statements. "
            "Most engineering signals are 'direct'. Flag the exceptions."
        ),
    )


class SurfaceDecision(BaseModel):
    """Whether a signal is worth interrupting a human about.

    This replaces the `if confidence > 0.7 and domain_weight > 0.5`
    heuristic in Agent.evaluate(). The agent reads the signal in the
    context of what the worker has seen and decides what action to take.
    """

    action: str = Field(
        description=(
            "'surface' if a human should see this now, "
            "'store' if it should be indexed for later retrieval, "
            "'ignore' if it adds no value."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident are you in this decision?",
    )
    reasoning: str = Field(
        description=(
            "Why this action? One sentence. "
            "'Two engineers are unknowingly building the same feature' "
            "not 'high confidence score'."
        ),
    )


class RecommendedArtifact(BaseModel):
    """A concrete, copy-pasteable action output.

    Instead of advisory text ("create a parent ticket"), produce the
    actual artifact: the draft ticket body, the Slack message, or the
    specific people-to-connect instruction. The system becomes a drafter,
    not an advisor.
    """

    artifact_type: str = Field(
        description=(
            "'draft_ticket' (ready to paste into Linear/Jira), "
            "'draft_message' (ready to paste into Slack/email), "
            "'connect_people' (specific intro: who, with whom, about what)."
        ),
    )
    target: str = Field(
        description=(
            "Who or where this artifact is directed: a person (@alice), "
            "a channel (#dev-platform), or a team (Platform)."
        ),
    )
    content: str = Field(
        description=(
            "The actual draft text. This should be copy-pasteable — "
            "a complete ticket body, a complete Slack message, or a "
            "complete 'connect A with B about C' instruction. "
            "Not a summary of what to write. The actual text."
        ),
    )


class CorrelationInsight(BaseModel):
    """A cross-signal pattern worth reporting — with action structure.

    Four-part finding: what was found, what we don't know yet,
    what to do about it, and what happens if we don't. The difference
    between a finding someone acknowledges and a finding someone acts on.

    The quality test: would a tech lead know exactly what to do
    Monday morning after reading this?
    """

    pattern_type: str = Field(
        description=(
            "What kind of pattern: 'overlap' (two people building the "
            "same thing), 'contradiction' (conflicting decisions), "
            "'risk' (something that could break), 'trend' (recurring "
            "theme), 'dependency' (cross-team coupling), or a "
            "self-describing string."
        ),
    )
    summary: str = Field(
        description=(
            "What was found. Be specific about who and what. "
            "Name people, ticket numbers, repos. "
            "'@alice and @bob both modifying the availability cache "
            "without a shared ticket (INT-332, INT-543)' not "
            "'multiple engineers working on related things'."
        ),
    )
    unknowns: str = Field(
        default="",
        description=(
            "1-3 specific unanswered questions that require investigation. "
            "These complete the conversion from 'nobody knew' to 'someone "
            "can investigate.' Not 'more research needed' but 'Do INT-332 "
            "and INT-543 share the same sync code path? When was the last "
            "change to the shared pipeline?' Each question should be "
            "answerable by a specific person looking at specific data."
        ),
    )
    recommended_action: str = Field(
        default="",
        description=(
            "An opinionated, specific, assignable next step. If you're "
            "confident enough to report a pattern, be confident enough "
            "to recommend what to do. 'Create a parent ticket linking "
            "these 8 sync bugs and assign one person to triage root "
            "cause' or 'Disable price matching immediately — the tax "
            "calculation is known broken per @otavio's Jan 12 comment.' "
            "Include who should do it when possible."
        ),
    )
    inaction_risk: str = Field(
        default="",
        description=(
            "Concrete stakes of doing nothing, using evidence from the "
            "signals. Not 'this could cause problems' but '$13K in owner "
            "payouts already failed to reach Stripe' or 'A double-booking "
            "went undetected for 2 months; 8 open sync bugs suggest more "
            "are accumulating.' Use numbers, ticket references, dollar "
            "amounts, and timelines from the signals."
        ),
    )
    severity: str = Field(
        description="'high' (act now), 'medium' (worth knowing), 'low' (background signal).",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description=(
            "How confident are you in this pattern? Use the full range: "
            "0.3 = weak signal, might be coincidence. "
            "0.5 = plausible but circumstantial. "
            "0.7 = strong evidence from multiple signals. "
            "0.85 = high confidence, clear pattern. "
            "0.95 = unambiguous, would bet on it."
        ),
    )
    actors: list[str] = Field(
        default_factory=list,
        description="People or systems involved in this pattern.",
    )
    signal_ids: list[str] = Field(
        default_factory=list,
        description="IDs of the signals that form this pattern.",
    )
    classification: str = Field(
        default="correlation",
        description=(
            "What did the system actually do to produce this finding? "
            "'retrieval': re-describing a known ticket/issue from a single source. "
            "'amplification': a known problem being quantified or escalated with new evidence. "
            "'correlation': connecting signals from different sources/people that nobody had assembled. "
            "'discovery': surfacing something nobody knew to ask about — true 2OI→1OI. "
            "If all signals reference the same existing ticket, this is retrieval. "
            "If signals span 2+ sources or 2+ unconnected authors, it's at least correlation."
        ),
    )
    temporal_relevance: str = Field(
        default="active",
        description=(
            "Is this finding about something currently happening or historical? "
            "'active': at least one referenced signal is open/in-progress. "
            "'historical': all referenced signals are resolved (merged PRs, closed tickets) "
            "but the pattern is still informative. "
            "'stale': the pattern was correct but the underlying code/state has moved past it."
        ),
    )
    artifacts: list[RecommendedArtifact] = Field(
        default_factory=list,
        description=(
            "Concrete, copy-pasteable action artifacts. For each finding, produce "
            "at least one: a draft Linear ticket body, a draft Slack message, or "
            "a 'connect person A with person B about topic C' instruction. "
            "The artifact should be ready to paste — not a description of what to write."
        ),
    )
    # ── Normalized Deviance assessment (variance compression research) ──
    # The correlator assesses these by reading the batch language, not by
    # regex or word lists. The agent KNOWS what hedging, bureaucratic
    # substitution, and temporal retreat look like — it reads the text.
    compression_level: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "How linguistically compressed is the language across signals in "
            "this batch? 0.0 = direct, specific, concrete (names, dates, "
            "numbers, commitments). 1.0 = hedged, passive, nominalized, vague. "
            "Assess the OVERALL language quality of the signals you reviewed, "
            "not just the finding itself. Are people speaking directly about "
            "problems or hiding behind process language?"
        ),
    )
    temporal_direction: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Is the language forward-looking or backward-looking? "
            "-1.0 = entirely retrospective (describing past, reviewing history). "
            "+1.0 = entirely future-oriented (commitments, deadlines, plans). "
            "0.0 = balanced or no temporal markers. "
            "Organizations that increasingly describe past achievements rather "
            "than future goals are compressing. Track the direction."
        ),
    )
    bureaucratic_index: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "How much bureaucratic substitution is present? Euphemistic "
            "replacements that narrow the semantic field: 'anomaly' for "
            "'failure', 'action item' for 'task', 'circle back' for 'discuss', "
            "'socialize' for 'tell people'. 0.0 = direct language. "
            "1.0 = saturated with process euphemisms. This measures lexical "
            "narrowing — the replacement of evaluative language with "
            "standardized terms that remove judgment."
        ),
    )
    exploration_signal: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Is the work in these signals exploring new territory or "
            "exploiting known territory? 0.0 = pure exploitation (same "
            "systems, same patterns, incremental refinement). "
            "1.0 = pure exploration (new systems, new approaches, novel "
            "territory). A team with high action but zero exploration is "
            "in the March trap — refining what they know while the "
            "environment shifts."
        ),
    )


class CorrelationBatch(BaseModel):
    """Batch output from the correlator agent.

    The correlator examines a set of recent signals and reports all
    cross-signal patterns it finds. An empty list is a valid answer —
    it means no meaningful patterns in this batch.

    The ND fields assess the batch's LANGUAGE quality — they're reported
    even when insights is empty, because the way an organization
    communicates is informative independent of specific findings.
    """

    insights: list[CorrelationInsight] = Field(
        default_factory=list,
        description=(
            "Cross-signal patterns found. Empty list if no meaningful "
            "patterns. Only include patterns you're confident about."
        ),
    )
    # Batch-level Normalized Deviance assessment — always reported
    batch_compression: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Overall linguistic compression of this batch. "
            "0.0 = direct, specific language. 1.0 = hedged, vague, passive. "
            "Assess the signals as a whole — how is this organization talking?"
        ),
    )
    batch_temporal: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Overall temporal direction. -1.0 = backward-looking. "
            "+1.0 = forward-looking. 0.0 = balanced."
        ),
    )
    batch_bureaucratic: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Overall bureaucratic substitution level. "
            "0.0 = plain language. 1.0 = saturated with process euphemisms."
        ),
    )
    batch_exploration: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Overall exploration vs exploitation. "
            "0.0 = all refinement of known systems. "
            "1.0 = all new territory."
        ),
    )


class CorroborationAssessment(BaseModel):
    """A worker's verdict on a single finding from Phase 2 corroboration.

    The worker evaluates a finding against everything it has seen —
    its accumulated context of signals, terms, and patterns. This is
    the same worker that processed the original signals, now asked
    "does this finding match what you've observed?"
    """

    finding_index: int = Field(
        description="Zero-based index of the finding being assessed.",
    )
    verdict: str = Field(
        description=(
            "'corroborate' if your accumulated context supports this finding. "
            "'contradict' if your context suggests the opposite. "
            "'partial' if you see partial evidence but not full support. "
            "'no_evidence' if your context has nothing relevant."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How confident are you in this verdict? "
            "0.3 = weak signal. 0.5 = moderate. 0.7 = strong. 0.9 = certain."
        ),
    )
    reasoning: str = Field(
        description=(
            "What specific signals or patterns from your context support "
            "this verdict? Reference specific things you've seen. "
            "'I've processed 3 Hostaway sync failures in the last week, "
            "which supports the bidirectional sync risk finding' not "
            "'the finding seems plausible'."
        ),
    )
    related_context: str = Field(
        default="",
        description=(
            "What additional context from your domain is relevant to this "
            "finding? This is new information the original finding might "
            "not have captured. 'The Escapia integration (SERV-198) also "
            "had similar sync issues last month' adds value."
        ),
    )


class CorroborationBatch(BaseModel):
    """Batch output from the corroborator agent.

    A worker receives a set of findings and assesses each one against
    its accumulated context. One assessment per finding.
    """

    assessments: list[CorroborationAssessment] = Field(
        default_factory=list,
        description=(
            "One assessment per finding presented. Assess every finding, "
            "even if your verdict is 'no_evidence'. The absence of evidence "
            "from your domain is itself informative."
        ),
    )


class MetaModelRecommendation(BaseModel):
    """A single recommendation from the meta-modeler.

    Recommendations are advisory — deposited to the discovery store
    for other agents to sense. Only inject_noise and competency
    adjustments have mechanical effects (when perturbation is enabled).
    """

    action: str = Field(
        description=(
            "What to do: 'inject_noise' (synthetic signal to unstick mesh), "
            "'adjust_vigilance' (tweak worker thresholds), "
            "'suggest_fork' (worker should split), "
            "'suggest_merge' (workers should combine), "
            "'boost_competency' (increase agent competency weight), "
            "'dampen_competency' (decrease agent competency weight), "
            "'no_action' (observation only, no intervention needed)."
        ),
    )
    target: str = Field(
        default="mesh",
        description=(
            "What the recommendation targets: a worker ID prefix, "
            "a competency name, or 'mesh' for system-wide."
        ),
    )
    detail: str = Field(
        default="",
        description="Specific description of the recommendation.",
    )
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How strongly to apply this recommendation. 0.0 = minimal, 1.0 = maximum.",
    )


class MetaModelAssessment(BaseModel):
    """Full output from the meta-modeler agent.

    The meta-modeler observes the mesh itself — not the signals it
    processes, but HOW it processes them. This assessment captures
    mesh state, diagnosis, and advisory recommendations.
    """

    mesh_state: str = Field(
        description=(
            "Overall mesh state: 'healthy' (V(x) decreasing, workers differentiated, "
            "routing stable), 'thrashing' (change rates too high, oscillating), "
            "'stuck' (change rates near zero, no learning), "
            "'fragmented' (too many underfed workers), "
            "'collapsed' (too few overfull workers)."
        ),
    )
    diagnosis: str = Field(
        description=(
            "One paragraph explaining why the mesh is in this state. "
            "Reference specific metrics: V(x) trend, worker count changes, "
            "familiarity rates, spectral high-frequency energy."
        ),
    )
    observations: list[str] = Field(
        default_factory=list,
        description=(
            "2-5 one-sentence observations about rates and trends. "
            "Each should reference a specific metric and its direction. "
            "'V(x) has decreased 12% over last 3 pulses — mesh is converging.' "
            "'Worker count increased from 5 to 8 but avg familiarity dropped — "
            "new workers are underfed.'"
        ),
    )
    recommendations: list[MetaModelRecommendation] = Field(
        default_factory=list,
        description=(
            "0-3 specific recommendations. Empty list is valid — healthy "
            "meshes don't need intervention. Only recommend when there's "
            "a clear problem with a clear remedy."
        ),
    )
    happy_place: str = Field(
        default="",
        description=(
            "If the mesh is healthy, describe WHY it's healthy in natural "
            "language. This description persists across sessions as a target "
            "state. 'Mesh is well-differentiated with 5 workers, each serving "
            "a distinct domain. V(x) is stable at 0.3. Familiarity averaging "
            "0.7 across workers.' Empty if not healthy."
        ),
    )
    health_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Overall health score. 0.0 = completely degraded, "
            "0.5 = functional but suboptimal, 1.0 = ideal state."
        ),
    )


class WorkerProfile(BaseModel):
    """A worker's self-description of its specialization.

    This replaces the 'top 3 terms by bloom frequency' label.
    The agent looks at what a worker has processed and describes
    what it's become expert in.
    """

    specialization: str = Field(
        description=(
            "2-5 word description of what this worker handles. "
            "'PMS sync and availability' not 'webhooks/potential/suggested'."
        ),
    )
    domains: list[str] = Field(
        description="The 2-3 most relevant domain areas.",
    )
    strength: str = Field(
        description="What this worker is best at — one sentence.",
    )
