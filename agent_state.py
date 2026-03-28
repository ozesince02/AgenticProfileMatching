"""
Agent State Schema

Defines the complete state machine for the LangGraph hiring agent.
Uses Pydantic for validation and JSON serialization.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, validator


@dataclass
class Message:
    """Single message in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RefinementAction:
    """Record of a single refinement action."""
    action: str  # "add_skill", "remove_skill", "set_experience_range", "adjust_weight", etc.
    description: str
    before_state: Dict[str, Any]  # snapshot of state before
    after_state: Dict[str, Any]   # snapshot of state after
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CandidateScoreBreakdown:
    """Score components for a single candidate."""
    candidate_name: str
    resume_path: str
    semantic_score: float  # 0-1
    skills_score: float    # 0-1
    must_have_score: float # 0-1
    match_score: int  # final 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateProfile:
    """Enriched candidate info after retrieval."""
    candidate_name: str
    resume_path: str
    skills: List[str]
    experience_years: float
    education: List[str]
    chunks: List[str]  # relevant resume snippets
    distances: List[float]  # embedding distances
    scores: Optional[CandidateScoreBreakdown] = None


@dataclass
class Round2Analysis:
    """Deep analysis results for Round 2."""
    candidate_name: str
    resume_path: str
    strengths: List[str]
    gaps: List[str]
    red_flags: List[str]
    suggested_questions: List[str]
    confidence_score: float  # 0-100
    interview_focus: List[str]


@dataclass
class FinalRecommendation:
    """Final decision for Round 3."""
    candidate_name: str
    resume_path: str
    final_score: float  # 0-100
    recommendation: str  # "Hire", "Hold", "No-Hire"
    scoreboard: Dict[str, float]  # {semantic_fit, skill_coverage, experience_alignment, must_have_satisfaction, interview_signals}
    rationale: str
    strengths: List[str]
    gaps: List[str]
    suggested_improvements: List[str]
    interview_focus: List[str]


class AgentState(BaseModel):
    """Complete state for the LangGraph agent."""
    
    # === Conversation & Context ===
    messages: List[Message] = Field(default_factory=list, description="Conversation history")
    current_job_description: str = Field(default="", description="Raw job description text")
    
    # === Parsed Requirements ===
    parsed_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed JD: {must_have_skills, good_to_have_skills, all_skills, min_experience_years}"
    )
    
    # === Retrieval Results ===
    aggregated_candidates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="resume_path → {name, skills, exp_years, chunks, distances}"
    )
    
    # === Ranking & Scoring ===
    shortlist: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top-K candidates with scores, currently ranked"
    )
    ranking_breakdown: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="candidate_path → {semantic_score, skills_score, must_have_score, match_score}"
    )
    
    # === Refinement & History ===
    refinement_history: List[RefinementAction] = Field(
        default_factory=list,
        description="Audit trail of refinements applied"
    )
    explanation_traces: List[str] = Field(
        default_factory=list,
        description="Decision explanations for transparency"
    )
    
    # === Multi-Round Screening ===
    current_round: int = Field(default=1, description="Round 1 (initial), 2 (deep), or 3 (final)")
    round2_analyses: Dict[str, Round2Analysis] = Field(
        default_factory=dict,
        description="Per-candidate deep analysis results (Round 2)"
    )
    final_recommendations: Dict[str, FinalRecommendation] = Field(
        default_factory=dict,
        description="Per-candidate final decision (Round 3)"
    )
    
    # === Scoring Parameters (User-Tunable) ===
    weights: Tuple[float, float, float] = Field(
        default=(0.60, 0.25, 0.15),
        description="(semantic_weight, skills_weight, must_have_weight) for Round 1 scoring"
    )
    top_k: int = Field(default=10, description="Number of candidates to return")
    top_chunks: int = Field(default=120, description="Number of chunks to retrieve before aggregation")
    
    # === Active Filters (Soft Constraints) ===
    skill_filter: List[str] = Field(default_factory=list, description="Additional required skills (soft filter)")
    experience_range: Tuple[float, float] = Field(default=(0.0, 100.0), description="(min, max) years experience")
    
    # === Node Transitions & Status ===
    last_node_executed: str = Field(default="", description="For debugging and audit")
    error_state: Optional[str] = Field(default=None, description="Error message if node failed")
    
    # === Session Metadata ===
    session_id: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Unique session identifier")
    
    class Config:
        """Pydantic config for JSON serialization."""
        arbitrary_types_allowed = True
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))
    
    def add_refinement(self, action: str, description: str, before: Dict, after: Dict) -> None:
        """Record a refinement action."""
        self.refinement_history.append(
            RefinementAction(
                action=action,
                description=description,
                before_state=before,
                after_state=after
            )
        )
    
    def add_explanation(self, text: str) -> None:
        """Add an explanation trace."""
        self.explanation_traces.append(f"[{datetime.now().isoformat()}] {text}")
    
    def get_current_shortlist(self) -> List[Dict[str, Any]]:
        """Get current ranked shortlist."""
        return self.shortlist
    
    def get_candidate_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find candidate in shortlist by name."""
        for cand in self.shortlist:
            if cand.get("candidate_name", "").lower() == name.lower():
                return cand
        return None
    
    def get_candidate_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Find candidate in shortlist by resume path."""
        for cand in self.shortlist:
            if cand.get("resume_path") == path:
                return cand
        return None
    
    def reset_for_new_job(self) -> None:
        """Reset state for a new job description."""
        self.current_job_description = ""
        self.parsed_requirements = {}
        self.aggregated_candidates = {}
        self.shortlist = []
        self.ranking_breakdown = {}
        self.skill_filter = []
        self.current_round = 1
        self.round2_analyses = {}
        self.final_recommendations = {}
        self.error_state = None
        # Keep messages for context, but clear refinement history for new job
        self.refinement_history = []
        self.explanation_traces = []
    
    def to_json(self) -> str:
        """Serialize state to JSON (safe for export)."""
        return json.dumps(self.dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> "AgentState":
        """Deserialize state from JSON."""
        data = json.loads(json_str)
        return cls(**data)


# Utility functions for state management

def create_initial_state() -> AgentState:
    """Create a fresh agent state."""
    return AgentState()


def get_state_summary(state: AgentState) -> str:
    """Get a concise summary of current state."""
    summary = []
    summary.append(f"Session: {state.session_id[:8]}...")
    summary.append(f"Job: {state.current_job_description[:50]}..." if state.current_job_description else "Job: Not loaded")
    summary.append(f"Round: {state.current_round}")
    summary.append(f"Shortlist: {len(state.shortlist)} candidates")
    summary.append(f"Messages: {len(state.messages)}")
    summary.append(f"Refinements: {len(state.refinement_history)}")
    if state.error_state:
        summary.append(f"⚠️ Error: {state.error_state}")
    return " | ".join(summary)
