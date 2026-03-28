"""
LangGraph Matching Agent

Core orchestration layer for candidate matching with state machine workflow.

Graph Structure:
  START → parse_jd → extract_requirements → search_resumes → 
  rank_candidates → generate_report → human_feedback_loop → END
  
  With loop-backs:
  - human_feedback_loop → extract_requirements (for refinement)
  - human_feedback_loop → rank_candidates (for comparison)
  - human_feedback_loop → END (finalize)
"""

import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from agent_state import AgentState, create_initial_state, get_state_summary
from agent_tools import (
    extract_requirements,
    rag_search_candidates,
    compare_candidates,
    generate_interview_questions,
    rerank_with_constraints,
    find_candidate_by_pattern,
)
from fs_tools import read_file, list_files
from job_matcher import normalize_text


# ============================================================================
# NODE HANDLERS
# ============================================================================

def node_parse_jd(state: AgentState) -> AgentState:
    """
    Parse job description node.
    
    Input: Assumes user has provided job_description (via CLI).
    Output: Validates and normalizes job_description in state.
    """
    state.add_explanation("Parsing job description...")
    
    if not state.current_job_description.strip():
        state.error_state = "No job description provided. Use: search for [job text] or /load_job [file]"
        state.last_node_executed = "parse_jd"
        return state
    
    # Normalize
    state.current_job_description = normalize_text(state.current_job_description)
    state.add_message("assistant", f"✅ Job description loaded ({len(state.current_job_description)} chars)")
    state.last_node_executed = "parse_jd"
    
    return state


def node_extract_requirements(state: AgentState) -> AgentState:
    """
    Extract requirements from job description.
    
    Uses agent_tools.extract_requirements() which applies regex + optional Gemini.
    """
    state.add_explanation("Extracting requirements from job description...")
    
    try:
        parsed = extract_requirements(state.current_job_description)
        state.parsed_requirements = parsed
        
        summary = f"""
Must-have: {', '.join(parsed['must_have_skills'][:3])}...
Min experience: {parsed['min_experience_years']:.0f}+ years
"""
        state.add_message("assistant", f"✅ Requirements extracted:\n{summary}")
        state.add_explanation(f"Parsed {len(parsed['all_skills'])} unique skills")
        
    except Exception as e:
        state.error_state = f"Failed to parse requirements: {str(e)}"
        state.add_message("assistant", f"❌ Error: {state.error_state}")
    
    state.last_node_executed = "extract_requirements"
    return state


def node_search_resumes(state: AgentState) -> AgentState:
    """
    Search resumes using RAG.
    
    Populates aggregated_candidates and initial shortlist.
    """
    state.add_explanation("Searching resume database...")
    
    if not state.parsed_requirements:
        state.error_state = "Requirements not parsed. Try again."
        state.last_node_executed = "search_resumes"
        return state
    
    try:
        result = rag_search_candidates(
            jd_text=state.current_job_description,
            top_k=state.top_k,
            top_chunks=state.top_chunks,
        )
        
        state.aggregated_candidates = result["aggregated_candidates"]
        state.shortlist = result["top_matches"]
        
        # Build ranking breakdown
        for match in state.shortlist:
            resume_path = match.get("resume_path", "")
            state.ranking_breakdown[resume_path] = {
                "semantic_score": match.get("semantic_score", 0.0),
                "skills_score": match.get("skills_score", 0.0),
                "must_have_score": match.get("must_have_score", 0.0),
                "match_score": match.get("match_score", 0),
            }
        
        msg = f"✅ Retrieved {len(state.shortlist)} candidates (Round 1: Initial Shortlist)"
        state.add_message("assistant", msg)
        state.add_explanation(f"Found {len(state.aggregated_candidates)} total candidates")
        state.current_round = 1
        
    except Exception as e:
        state.error_state = f"Search failed: {str(e)}"
        state.add_message("assistant", f"❌ Search error: {state.error_state}")
    
    state.last_node_executed = "search_resumes"
    return state


def node_rank_candidates(state: AgentState) -> AgentState:
    """
    Rank candidates (may refine based on refinement_history).
    
    For now, maintains existing ranking. Will be called again during refinements.
    """
    state.add_explanation("Ranking candidates...")
    
    if not state.aggregated_candidates:
        state.error_state = "No candidates to rank"
        state.last_node_executed = "rank_candidates"
        return state
    
    # If refinements were applied, rerank
    if state.refinement_history:
        last_refinement = state.refinement_history[-1]
        # Extract refinement params
        refinements = {
            k: v for k, v in last_refinement.after_state.items()
            if k in ["skill_filter", "experience_range", "weights"]
        }
        
        if refinements:
            new_ranked, delta_explanation = rerank_with_constraints(state, refinements)
            
            if new_ranked:
                state.shortlist = new_ranked
                state.add_message("assistant", delta_explanation)
                state.add_explanation("Applied refinement constraints and re-ranked")
    
    # Display current ranking
    msg = "**Top Candidates:**\n"
    for i, cand in enumerate(state.shortlist[:5], 1):
        name = cand.get("candidate_name", "Unknown")
        score = cand.get("match_score", 0)
        skills = ", ".join(cand.get("matched_skills", [])[:2])
        msg += f"{i}. {name} ({score}/100) - {skills}\n"
    
    state.add_message("assistant", msg)
    state.last_node_executed = "rank_candidates"
    
    return state


def node_generate_report(state: AgentState) -> AgentState:
    """
    Generate human-readable report of current findings.
    
    Triggered after initial search and after user requests finalization.
    """
    state.add_explanation("Generating report...")
    
    if not state.shortlist:
        state.add_message("assistant", "No candidates to report.")
        state.last_node_executed = "generate_report"
        return state
    
    # Build report
    msg = f"\n📋 **Matching Report** (Round {state.current_round})\n\n"
    
    top_candidate = state.shortlist[0] if state.shortlist else None
    if top_candidate:
        msg += f"**Top Match:** {top_candidate.get('candidate_name')}\n"
        msg += f"- Score: {top_candidate.get('match_score')}/100\n"
        msg += f"- Skills: {', '.join(top_candidate.get('matched_skills', [])[:4])}\n"
        msg += f"- Relevance: {top_candidate.get('reasoning', 'See details for reasoning')}\n\n"
    
    msg += f"**Total Candidates Reviewed:** {len(state.aggregated_candidates)}\n"
    msg += f"**Shortlist:** {len(state.shortlist)} candidates\n\n"
    
    msg += "**Next Steps:**\n"
    if state.current_round == 1:
        msg += "- Type `/compare [names]` to see side-by-side comparison\n"
        msg += "- Type `/explain [name]` for detailed reasoning\n"
        msg += "- Type `/round2` for deep analysis on top 5\n"
    elif state.current_round == 2:
        msg += "- Type `show interview questions for [name]`\n"
        msg += "- Type `/round3` for final Hire/Hold/No-Hire decisions\n"
    else:
        msg += "- Final recommendations ready\n"
        msg += "- Type `/export` to save results\n"
    
    state.add_message("assistant", msg)
    state.last_node_executed = "generate_report"
    
    return state


def node_human_feedback_loop(state: AgentState) -> AgentState:
    """
    Wait for human feedback and route to appropriate next action.
    
    This node is responsible for user intent routing but the actual
    command handling happens in the CLI. This node prepares for routing.
    """
    state.add_explanation("Waiting for user feedback...")
    
    # This is a marker node. Actual routing happens in chat_cli.py
    # which decides whether to:
    # - Loop back to extract_requirements (for refinement)
    # - Loop back to rank_candidates (for comparison)
    # - Continue to END (for finalization)
    
    state.last_node_executed = "human_feedback_loop"
    
    return state


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_matching_agent_graph() -> StateGraph:
    """
    Construct the LangGraph state machine.
    
    Nodes: parse_jd, extract_requirements, search_resumes, rank_candidates,
           generate_report, human_feedback_loop
    
    Transitions: Linear flow + loop-back for refinements
    """
    
    graph_builder = StateGraph(AgentState)
    
    # Add nodes
    graph_builder.add_node("parse_jd", node_parse_jd)
    graph_builder.add_node("extract_requirements", node_extract_requirements)
    graph_builder.add_node("search_resumes", node_search_resumes)
    graph_builder.add_node("rank_candidates", node_rank_candidates)
    graph_builder.add_node("generate_report", node_generate_report)
    graph_builder.add_node("human_feedback_loop", node_human_feedback_loop)
    
    # Add edges (linear flow)
    graph_builder.add_edge(START, "parse_jd")
    graph_builder.add_edge("parse_jd", "extract_requirements")
    graph_builder.add_edge("extract_requirements", "search_resumes")
    graph_builder.add_edge("search_resumes", "rank_candidates")
    graph_builder.add_edge("rank_candidates", "generate_report")
    graph_builder.add_edge("generate_report", "human_feedback_loop")
    
    # Default end
    graph_builder.add_edge("human_feedback_loop", END)
    
    return graph_builder


class MatchingAgent:
    """
    High-level interface to the LangGraph matching agent.
    
    Provides methods to:
    - Run the main workflow
    - Refine requirements
    - Compare candidates
    - Generate questions
    - Finalize recommendations
    """
    
    def __init__(self):
        """Initialize agent with fresh state and compiled graph."""
        self.state = create_initial_state()
        self.graph_builder = build_matching_agent_graph()
        self.graph = self.graph_builder.compile()
    
    def load_job_description(self, job_text: str) -> None:
        """Load a job description (from text or file)."""
        if not job_text:
            raise ValueError("Job description cannot be empty")
        
        self.state.current_job_description = job_text
        self.state.add_message("user", f"Loaded job: {job_text[:50]}...")
    
    def run_initial_search(self) -> Dict[str, Any]:
        """
        Run the complete initial search workflow.
        
        Executes: parse_jd → extract_requirements → search_resumes →
                  rank_candidates → generate_report
        """
        # Use invoke for compatibility across LangGraph stream output formats.
        result_state = self.graph.invoke(self.state, {"recursion_limit": 25})
        if isinstance(result_state, AgentState):
            self.state = result_state
        elif isinstance(result_state, dict):
            self.state = AgentState(**result_state)
        else:
            raise TypeError(f"Unexpected graph output type: {type(result_state)}")
        
        return {
            "success": not self.state.error_state,
            "state": self.state,
            "shortlist": self.state.shortlist,
            "error": self.state.error_state,
        }
    
    def refine_requirements(self, refinements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine requirements and re-rank.
        
        Refinements dict:
        - skill_filter: List[str]
        - experience_range: Tuple[float, float]
        - weights: Tuple[float, float, float]
        """
        # Record refinement
        before = {
            "skill_filter": self.state.skill_filter,
            "experience_range": self.state.experience_range,
            "weights": self.state.weights,
        }
        
        self.state.add_refinement(
            action="refinement",
            description=f"Applied: {list(refinements.keys())}",
            before=before,
            after=refinements,
        )
        
        # Re-rank directly from current cached candidates to avoid unnecessary graph traversal.
        new_ranked, delta_explanation = rerank_with_constraints(self.state, refinements)
        if new_ranked:
            self.state.shortlist = new_ranked

        self.state.add_message("assistant", delta_explanation)
        self.state.add_explanation("Applied refinement constraints and re-ranked")
        self.state.last_node_executed = "rank_candidates"

        # Regenerate summary after refinement.
        self.state = node_generate_report(self.state)
        
        return {
            "success": True,
            "shortlist": self.state.shortlist,
            "messages": [m.content for m in self.state.messages[-3:]],
        }
    
    def compare_top_candidates(self, candidate_ids: List[str]) -> Dict[str, Any]:
        """Compare specified candidates."""
        result = compare_candidates(candidate_ids, self.state)
        return result
    
    def explain_candidate(self, candidate_id: str) -> Dict[str, Any]:
        """Explain why a candidate ranked where they did."""
        cand = find_candidate_by_pattern(candidate_id, self.state)
        
        if not cand:
            return {"error": f"Candidate '{candidate_id}' not found"}
        
        explanation = f"""
**Explanation for {cand.get('candidate_name')}**

Match Score: {cand.get('match_score')}/100

**Score Breakdown:**
- Semantic Fit: {cand.get('semantic_score', 0):.0%}
- Skill Overlap: {cand.get('skills_score', 0):.0%}
- Must-Have Satisfaction: {cand.get('must_have_score', 0):.0%}

**Matched Skills:** {', '.join(cand.get('matched_skills', []))}

**Reasoning:** {cand.get('reasoning', 'Standard hybrid scoring applied')}

**Top Matching Sections:**
{chr(10).join(['- ' + excerpt[:80] + '...' for excerpt in cand.get('relevant_excerpts', [])[:2]])}
"""
        
        return {
            "candidate_name": cand.get("candidate_name"),
            "explanation": explanation,
        }
    
    def generate_interview_for_candidate(self, candidate_id: str) -> Dict[str, Any]:
        """Generate interview questions for a candidate."""
        return generate_interview_questions(candidate_id, self.state, num_questions=5)
    
    def start_round2_analysis(self) -> Dict[str, Any]:
        """
        Transition to Round 2: Deep analysis.
        
        For each top candidate, generate interview questions and annotate
        with strengths, gaps, red flags.
        """
        if self.state.current_round != 1:
            return {"error": "Already past Round 1"}
        
        self.state.current_round = 2
        self.state.add_message("assistant", "🔍 Entering Round 2: Deep Analysis")
        self.state.add_explanation("Started Round 2 deep analysis")
        
        # For each top 5 candidate, generate questions
        for i, cand in enumerate(self.state.shortlist[:5]):
            candidate_name = cand.get("candidate_name")
            q_result = generate_interview_questions(candidate_name, self.state)
            
            if "questions" in q_result:
                self.state.add_message(
                    "assistant",
                    f"\n**Interview Questions for {candidate_name}:**\n" +
                    "\n".join([f"- {q}" for q in q_result.get("questions", [])[:3]])
                )
        
        return {
            "success": True,
            "current_round": self.state.current_round,
            "top_candidates_analysis": len(self.state.shortlist[:5]),
        }
    
    def start_round3_final(self) -> Dict[str, Any]:
        """
        Transition to Round 3: Final recommendations.
        
        Apply balanced rubric and output Hire / Hold / No-Hire.
        """
        if self.state.current_round != 2:
            return {"error": "Must complete Round 2 first. Type: /round2"}
        
        self.state.current_round = 3
        self.state.add_message("assistant", "✅ Entering Round 3: Final Recommendations")
        self.state.add_explanation("Started Round 3 final scoring")
        
        # Apply Round 3 rubric
        rubric_msg = "**Round 3 Scoring Rubric (0-100):**\n"
        rubric_msg += "- 30% Semantic Fit\n"
        rubric_msg += "- 25% Skill Coverage\n"
        rubric_msg += "- 20% Experience Alignment\n"
        rubric_msg += "- 15% Must-Have Satisfaction\n"
        rubric_msg += "- 10% Interview Signals\n\n"
        rubric_msg += "**Recommendations:**\n"
        rubric_msg += "- **Hire** (>80): Strong match, interview immediately\n"
        rubric_msg += "- **Hold** (60-80): Good match, needs clarification\n"
        rubric_msg += "- **No-Hire** (<60): Misalignment\n"
        
        self.state.add_message("assistant", rubric_msg)
        
        # Score each candidate
        for cand in self.state.shortlist[:5]:
            score = cand.get("match_score", 0)
            if score > 80:
                rec = "🟢 **HIRE**"
            elif score >= 60:
                rec = "🟡 **HOLD**"
            else:
                rec = "🔴 **NO-HIRE**"
            
            msg = f"{rec} - {cand.get('candidate_name')} ({score}/100)\n"
            msg += f"   Skills: {', '.join(cand.get('matched_skills', [])[:3])}\n"
            msg += f"   Reasoning: {cand.get('reasoning', 'See details')}"
            
            self.state.add_message("assistant", msg)
        
        return {
            "success": True,
            "current_round": self.state.current_round,
            "final_recommendations_ready": True,
        }
    
    def reset_session(self) -> None:
        """Reset for a new job."""
        self.state.reset_for_new_job()
        self.state.add_message("assistant", "✅ Session reset. Ready for new job description.")
    
    def get_state_snapshot(self) -> str:
        """Get human-readable state summary."""
        return get_state_summary(self.state)
    
    def export_state(self, filepath: str = None) -> str:
        """Export state to JSON."""
        if not filepath:
            filename = f"agent_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join("exports", filename)

        export_dir = os.path.dirname(filepath)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.state.to_json())
        
        return filepath


# Convenience function
def create_agent() -> MatchingAgent:
    """Factory function to create a new MatchingAgent."""
    return MatchingAgent()
