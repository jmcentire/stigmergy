"""Pydantic models for .stigmergy/config.yaml."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GitHubSourceConfig(BaseModel):
    enabled: bool = True
    mode: str = "live"
    repos: list[str] = Field(default_factory=list)
    poll_interval: int = 300


class LinearTeamConfig(BaseModel):
    name: str
    id: str


class LinearSourceConfig(BaseModel):
    enabled: bool = True
    mode: str = "mock"
    api_key: str = ""  # or set LINEAR_API_KEY env var
    organization_id: str = ""
    teams: list[LinearTeamConfig] = Field(default_factory=list)
    poll_interval: int = 300


class SlackSourceConfig(BaseModel):
    enabled: bool = False
    mode: str = "mock"
    bot_token: str = ""  # xoxb-... or set SLACK_BOT_TOKEN env var
    channels: list[str] = Field(default_factory=list)  # channel names to monitor
    poll_interval: int = 300


class SourcesConfig(BaseModel):
    github: GitHubSourceConfig = Field(default_factory=GitHubSourceConfig)
    linear: LinearSourceConfig = Field(default_factory=LinearSourceConfig)
    slack: SlackSourceConfig = Field(default_factory=SlackSourceConfig)


class ContextConfig(BaseModel):
    description: str = ""  # self-describing: what this context observes and why
    terms: list[str] = Field(default_factory=list)  # seed terms — agents expand these
    business_weight: float = 1.0
    relevance_threshold: float = 0.15


class AgentConfig(BaseModel):
    description: str = ""  # self-describing: this agent's role and perspective
    contexts: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.85
    weights: dict[str, float] = Field(default_factory=dict)
    competencies: dict[str, float] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    provider: str = "stub"
    model: str = "claude-haiku-4-5-20251001"


class BudgetConfig(BaseModel):
    daily_cap_usd: float = 5.00
    hourly_cap_usd: float = 1.00
    max_agents: int = 10
    max_contexts: int = 20
    warn_at_percent: int = 80
    input_cost_per_million: float = 0.0  # 0 = auto-detect from model
    output_cost_per_million: float = 0.0  # 0 = auto-detect from model


class FamiliarityWeightsConfig(BaseModel):
    embedding_similarity: float = 0.0
    keyword_overlap: float = 0.5
    source_affinity: float = 0.2
    temporal_proximity: float = 0.2
    author_affinity: float = 0.1


class PipelineConfig(BaseModel):
    consensus_threshold: float = 0.4
    uncertainty_threshold: float = 0.15
    dedup_enabled: bool = True
    familiarity_weights: FamiliarityWeightsConfig = Field(
        default_factory=FamiliarityWeightsConfig
    )


class OutputConfig(BaseModel):
    surface_only: bool = False
    show_traces: bool = False


class AttentionConfig(BaseModel):
    enabled: bool = False  # opt-in: existing behavior unchanged
    target_persons: list[str] = Field(default_factory=list)  # canonical_ids to rank for
    decay_half_life_days: float = 14.0
    channel_size_reference: int = 5
    volume_discount_half: int = 50
    importance_weight: float = 1.0
    annoyance_weight: float = 0.3
    calibration_lr_up: float = 0.05    # "already_knew" — small upward nudge
    calibration_lr_down: float = 0.20  # "had_no_idea" — aggressive correction


class MetaModelerConfig(BaseModel):
    enabled: bool = False  # opt-in: no meta-modeler unless explicitly enabled
    fire_interval: int = 1  # every Nth health pulse
    max_output_tokens: int = 400
    perturbation_enabled: bool = False  # opt-in: allow inject_noise, competency nudges
    competency_reinforcement: bool = True  # close the open competency loop


class UnityConfig(BaseModel):
    enabled: bool = False  # opt-in: no field equations unless explicitly enabled
    # CERTX weights
    w_coherence: float = 1.0
    w_entropy: float = 1.0
    w_resonance: float = 1.0
    w_temperature: float = 1.0
    w_substrate: float = 1.0
    # PID
    pid_kp: float = 0.3
    pid_ki: float = 0.05
    pid_kd: float = 0.1
    pid_setpoint: float = 0.7
    # Breathing
    breathing_period: int = 200
    breathing_amplitude: float = 0.1
    accumulate_ratio: float = 0.6
    # Diversity
    min_diversity_index: float = 0.3
    max_specialization: float = 0.9
    # Calibration
    babbling_threshold: float = 0.3
    calibration_window: int = 50
    # Bridge
    bridge_enabled: bool = False
    bridge_vigilance_gain: float = 0.1
    bridge_learning_gain: float = 0.05
    bridge_quorum_gain: float = 0.05
    # Eigenvalue
    eigenvalue_healthy_min: float = 0.8
    eigenvalue_healthy_max: float = 1.2
    jacobian_window: int = 5
    # Logging
    log_field_state: bool = True


class NormalizedDevianceConfig(BaseModel):
    enabled: bool = True
    linguistic_enabled: bool = True       # cheapest — regex only
    action_correlation_enabled: bool = True  # cheap — signal classification
    solo_ownership_enabled: bool = False   # more expensive — needs per-signal actor tracking
    channel_migration_enabled: bool = False  # needs channel metadata from Slack
    phase_lock_enabled: bool = False       # depends on action-correlation data
    decoupling_threshold: float = 0.1      # action/discussion ratio below this = decoupling
    migration_threshold: float = 0.3       # migration score above this = alert
    compression_warning: float = 0.3       # compression index above this = warn
    compression_critical: float = 0.5      # compression index above this = critical


class MeshWorkerConfig(BaseModel):
    capacity: int = 200
    base_threshold: float = 0.15
    max_threshold: float = 0.8
    threshold_curve: float = 2.0
    high_relevance_offset: float = 0.2


class MeshTemporalConfig(BaseModel):
    min_step_seconds: float = 60
    max_step_seconds: float = 86400
    step_growth_factor: float = 1.5
    max_depth_seconds: float = 604800  # 1 week
    fresh_window_seconds: float = 300  # 5 minutes


class MeshConfig(BaseModel):
    enabled: bool = True
    max_hops: int = 10
    max_workers: int = 50
    worker: MeshWorkerConfig = Field(default_factory=MeshWorkerConfig)
    temporal: MeshTemporalConfig = Field(default_factory=MeshTemporalConfig)


class WorkspaceConfig(BaseModel):
    root: str = ""
    config_dir: str = ""
    repos_dir: str = ""


class IdentityConfig(BaseModel):
    github_org: str = ""  # for GitHubOrgProvider
    persist_learned_aliases: bool = True
    sensitivity_level: str = "internal"  # max PII level in output


class StigmergyConfig(BaseModel):
    goal: str = "Monitor engineering signals"
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    contexts: dict[str, ContextConfig] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    normalized_deviance: NormalizedDevianceConfig = Field(default_factory=NormalizedDevianceConfig)
    meta_modeler: MetaModelerConfig = Field(default_factory=MetaModelerConfig)
    unity: UnityConfig = Field(default_factory=UnityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def default_config() -> StigmergyConfig:
    """Return a fully-populated default config."""
    return StigmergyConfig(
        goal="Monitor engineering signals for the platform team",
        sources=SourcesConfig(
            github=GitHubSourceConfig(
                enabled=True,
                mode="live",
                repos=[
                    "acme-org/backend",
                    "acme-org/pricing",
                    "acme-org/property",
                    "acme-org/payment",
                    "acme-org/sync-service",
                    "acme-org/events",
                    "acme-org/availability",
                    "acme-org/frontend",
                    "acme-org/admin-portal",
                    "acme-org/mobile",
                    "acme-org/marketing-site",
                    "acme-org/orchestrator",
                    "acme-org/data-pipelines",
                    "acme-org/event-schemas",
                    "acme-org/media-services",
                    "acme-org/infrastructure",
                ],
                poll_interval=300,
            ),
            linear=LinearSourceConfig(
                enabled=True,
                mode="mock",
                organization_id="00000000-0000-0000-0000-000000000000",
                teams=[
                    LinearTeamConfig(name="Platform", id="11111111-1111-1111-1111-111111111111"),
                    LinearTeamConfig(name="Integrations", id="22222222-2222-2222-2222-222222222222"),
                ],
                poll_interval=300,
            ),
            slack=SlackSourceConfig(enabled=False),
        ),
        contexts={
            "engineering": ContextConfig(
                description="Technical work: code changes, bug fixes, deployments, API work",
                terms=["bug", "error", "fix", "pr", "merge", "deploy", "sync", "api", "webhook", "cache"],
                business_weight=3.0,
                relevance_threshold=0.15,
            ),
            "support": ContextConfig(
                description="Customer-facing: issues, complaints, bookings, support tickets",
                terms=["customer", "issue", "booking", "complaint", "ticket", "support"],
                business_weight=2.0,
                relevance_threshold=0.15,
            ),
            "leadership": ContextConfig(
                description="Business strategy: revenue, market positioning, growth decisions",
                terms=["revenue", "market", "strategy", "growth", "expansion"],
                business_weight=5.0,
                relevance_threshold=0.2,
            ),
        },
        agents={
            "eng_watcher": AgentConfig(
                description="Engineering-focused: watches for technical patterns, code conflicts, deployment risks",
                contexts={"engineering": 0.95},
                confidence=0.85,
                weights={"technical": 0.9, "business": 0.2, "operations": 0.7},
                competencies={"similarity_detection": 0.8, "temporal_analysis": 0.6, "cross_context_propagation": 0.3},
            ),
            "cross_cutter": AgentConfig(
                description="Cross-cutting: bridges engineering, leadership, and support to surface business-relevant technical patterns",
                contexts={"engineering": 0.7, "leadership": 0.9, "support": 0.5},
                confidence=0.85,
                weights={"technical": 0.8, "business": 0.9, "operations": 0.6},
                competencies={"cross_context_propagation": 0.9, "similarity_detection": 0.5, "temporal_analysis": 0.4},
            ),
        },
        llm=LLMConfig(provider="stub", model="claude-haiku-4-5-20251001"),
        budget=BudgetConfig(
            daily_cap_usd=5.00,
            hourly_cap_usd=1.00,
            max_agents=10,
            max_contexts=20,
            warn_at_percent=80,
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
        ),
        pipeline=PipelineConfig(
            consensus_threshold=0.4,
            uncertainty_threshold=0.15,
            dedup_enabled=True,
            familiarity_weights=FamiliarityWeightsConfig(
                embedding_similarity=0.0,
                keyword_overlap=0.5,
                source_affinity=0.2,
                temporal_proximity=0.2,
                author_affinity=0.1,
            ),
        ),
        output=OutputConfig(surface_only=False, show_traces=False),
    )
