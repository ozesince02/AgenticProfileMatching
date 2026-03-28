"""
Agent Tools Module

Wraps existing RAG functions and adds new tools for LangGraph agent.
Integrates Gemini for enrichment (with offline fallback).
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

from job_matcher import (
    parse_job_description,
    match_candidates,
    compute_score,
    aggregate_candidates,
    extract_skills_from_text,
    normalize_text,
)
from resume_rag import (
    SKILL_VOCAB,
    extract_name,
    extract_skills,
    extract_experience_years,
)
from agent_state import AgentState, CandidateScoreBreakdown

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Keep runtime resilient if dotenv is unavailable.
    pass

# Global cache for embedder (avoid re-instantiation per query)
_embedder_cache: Dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Get or create embedder (cached)."""
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def init_gemini_client():
    """Initialize Gemini client if API key available, return None otherwise."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
        client = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )
        return client
    except Exception as e:
        print(f"⚠️ Gemini initialization failed: {e}. Will use offline fallback.")
        return None


# Global Gemini client (lazy init)
_gemini_client = None


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """Parse JSON from plain or fenced LLM output."""
    if not text:
        return {}

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", candidate)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}

def get_gemini_client():
    """Get Gemini client (lazy initialization)."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = init_gemini_client()
    return _gemini_client


# ============================================================================
# TOOL 1: Extract Requirements
# ============================================================================

def extract_requirements(jd_text: str) -> Dict[str, Any]:
    """
    Extract structured requirements from job description.
    
    Uses existing regex parser + optional Gemini refinement.
    
    Returns:
        {
            "must_have_skills": [...],
            "good_to_have_skills": [...],
            "all_skills": [...],
            "min_experience_years": float,
            "raw_parsed": {...}  # full parse_job_description output
        }
    """
    # Use existing parser
    parsed = parse_job_description(jd_text)
    
    # Optional Gemini refinement (normalize and enrich skill names)
    gemini = get_gemini_client()
    
    if gemini:
        try:
            # Ask Gemini to normalize skill names to match our vocab
            prompt = f"""
Given these skills extracted from a job description:
- Must-have: {', '.join(parsed['must_have_skills'][:5])}
- Good-to-have: {', '.join(parsed['good_to_have_skills'][:5])}

Normalize these to common industry terms (Python, Java, React, etc). 
Return ONLY a JSON with "must_have_normalized" and "good_to_have_normalized" lists.
"""
            response = gemini.invoke(prompt)
            enriched = _parse_json_from_text(response.content)
            
            parsed['must_have_skills'] = enriched.get('must_have_normalized', parsed['must_have_skills'])
            parsed['good_to_have_skills'] = enriched.get('good_to_have_normalized', parsed['good_to_have_skills'])
            parsed['all_skills'] = sorted(set(parsed['must_have_skills'] + parsed['good_to_have_skills']))
        except Exception as e:
            print(f"⚠️ Gemini skill normalization failed, using raw extraction: {e}")
    
    return {
        "must_have_skills": parsed['must_have_skills'],
        "good_to_have_skills": parsed['good_to_have_skills'],
        "all_skills": parsed['all_skills'],
        "min_experience_years": parsed['min_experience_years'],
        "raw_parsed": parsed,
    }


# ============================================================================
# TOOL 2: RAG Search Candidates
# ============================================================================

def rag_search_candidates(
    jd_text: str,
    top_k: int = 10,
    top_chunks: int = 120,
    persist_dir: str = "data/vector_db",
    collection_name: str = "resumes",
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Retrieve top candidates using RAG with existing match_candidates logic.
    
    Returns indexed aggregated candidates ready for ranking.
    """
    result = match_candidates(
        job_description=jd_text,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
        top_k=top_k,
        top_chunks=top_chunks,
    )
    
    # Return with metadata for agent
    return {
        "aggregated_candidates": aggregate_candidates(
            chromadb.PersistentClient(path=persist_dir)
            .get_collection(collection_name)
            .query(
                query_embeddings=[
                    get_embedder(model_name).encode([jd_text], normalize_embeddings=True)[0].tolist()
                ],
                n_results=min(top_chunks, 500),  # Clamp to reasonable max
                include=["documents", "metadatas", "distances"],
            )
        ),
        "top_matches": result["top_matches"],
        "retrieval_metadata": {
            "top_chunks_queried": top_chunks,
            "top_k_returned": len(result["top_matches"]),
        }
    }


# ============================================================================
# TOOL 3: Compare Candidates
# ============================================================================

def compare_candidates(candidate_ids: List[str], state: AgentState) -> Dict[str, Any]:
    """
    Side-by-side comparison of candidates.
    
    Args:
        candidate_ids: List of resume paths or candidate names to compare
        state: Current agent state
    
    Returns:
        {
            "comparison": [
                {
                    "candidate_name": "...",
                    "resume_path": "...",
                    "skills": [...],
                    "experience_years": float,
                    "match_score": int,
                    "matched_skills": [...],
                    "strengths": [...],
                    "gaps": [...],
                }
            ]
        }
    """
    comparison = []
    
    for cand_id in candidate_ids:
        # Find candidate in state
        cand = None
        cand_id = (cand_id or "").strip()
        if not cand_id:
            continue
        
        # Try by name
        cand = state.get_candidate_by_name(cand_id)
        if not cand:
            # Try by path
            cand = state.get_candidate_by_path(cand_id)
        if not cand:
            # Fallback: allow partial/case-insensitive name/path matches.
            cand = find_candidate_by_pattern(cand_id, state)
        
        if not cand:
            continue
        
        # Extract details from aggregated candidates if available
        agg_cand = state.aggregated_candidates.get(cand.get("resume_path", ""), {})
        
        comparison.append({
            "candidate_name": cand.get("candidate_name", "Unknown"),
            "resume_path": cand.get("resume_path", ""),
            "skills": agg_cand.get("skills", []),
            "experience_years": agg_cand.get("experience_years", 0.0),
            "match_score": cand.get("match_score", 0),
            "matched_skills": cand.get("matched_skills", []),
            "relevant_excerpts": cand.get("relevant_excerpts", [])[:2],
            "semantic_score": cand.get("semantic_score", 0.0),
            "skills_score": cand.get("skills_score", 0.0),
            "must_have_score": cand.get("must_have_score", 0.0),
        })
    
    if not comparison:
        return {"error": "No candidates found to compare", "comparison": []}
    
    return {"comparison": comparison}


# ============================================================================
# TOOL 4: Generate Interview Questions
# ============================================================================

INTERVIEW_QUESTION_TEMPLATES = {
    "skills_deep_dive": "Tell us about your most complex {skill} project and the challenges you faced.",
    "experience_trajectory": "Walk us through your progression from {role1} to {role2}. What drove that change?",
    "domain_expertise": "Describe your experience with {industry} and how you'd apply it to this role.",
    "problem_solving": "Give an example of a technical problem you solved with {skill}. How did you approach it?",
    "collaboration": "Tell us about a time you collaborated with {role} professionals. How did you communicate technical concepts?",
    "growth_mindset": "Which skill in this job description is new to you? How would you learn it?",
    "achievement": "What's your biggest achievement using {skill}? How does it relate to this role?",
}


def generate_interview_questions(
    candidate_id: str,
    state: AgentState,
    num_questions: int = 5,
) -> Dict[str, Any]:
    """
    Generate interview questions tailored to candidate and job.
    
    Uses Gemini if available, otherwise template-based fallback.
    """
    # Find candidate
    cand = state.get_candidate_by_name(candidate_id) or state.get_candidate_by_path(candidate_id)
    if not cand:
        return {"error": f"Candidate {candidate_id} not found"}
    
    resume_path = cand.get("resume_path", "")
    agg_cand = state.aggregated_candidates.get(resume_path, {})
    candidate_name = cand.get("candidate_name", "Candidate")
    candidate_skills = agg_cand.get("skills", [])
    
    # Try Gemini first
    gemini = get_gemini_client()
    if gemini:
        try:
            prompt = f"""
Generate {num_questions} targeted interview questions for:
- Role: {state.parsed_requirements.get('role', 'Software Engineer')}
- Candidate: {candidate_name}
- Candidate Skills: {', '.join(candidate_skills[:5])}
- Job Requirements: {', '.join(state.parsed_requirements.get('must_have_skills', [])[:5])}

Return ONLY a JSON with "questions" as a list of strings.
"""
            response = gemini.invoke(prompt)
            result = _parse_json_from_text(response.content)
            return {
                "candidate_name": candidate_name,
                "resume_path": resume_path,
                "questions": result.get("questions", []),
                "source": "gemini",
            }
        except Exception as e:
            print(f"⚠️ Gemini question generation failed: {e}. Using template fallback.")
    
    # Template-based fallback
    questions = []
    job_must_have = state.parsed_requirements.get("must_have_skills", [])
    job_good_to_have = state.parsed_requirements.get("good_to_have_skills", [])
    
    # Build from templates
    if candidate_skills and job_must_have:
        skill = candidate_skills[0]
        q = INTERVIEW_QUESTION_TEMPLATES["skills_deep_dive"].format(skill=skill)
        questions.append(q)
    
    if job_must_have:
        q = INTERVIEW_QUESTION_TEMPLATES["problem_solving"].format(skill=job_must_have[0])
        questions.append(q)
    
    if job_good_to_have:
        q = INTERVIEW_QUESTION_TEMPLATES["growth_mindset"]
        questions.append(q)
    
    questions.append(f"Why are you interested in joining a team focused on this role?")
    questions.append(f"What's your approach to staying current with {job_must_have[0] if job_must_have else 'your domain'}?")
    
    return {
        "candidate_name": candidate_name,
        "resume_path": resume_path,
        "questions": questions[:num_questions],
        "source": "template",
    }


# ============================================================================
# TOOL 5: Rerank with Constraints
# ============================================================================

def rerank_with_constraints(
    state: AgentState,
    refinements: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Re-score and re-rank aggregated candidates based on refined constraints.
    
    Refinements dict can contain:
    - "skill_filter": List[str] — additional required skills
    - "min_experience": float
    - "max_experience": float
    - "weights": Tuple[float, float, float]
    
    Returns:
        (new_ranked_list, delta_explanation)
    """
    if not state.aggregated_candidates:
        return [], "No candidates to rerank. Try searching first."
    
    # Capture before state
    before_ranked = state.shortlist.copy()
    before_scores = {item["resume_path"]: item["match_score"] for item in before_ranked}
    
    # Apply refinements to state
    if "skill_filter" in refinements:
        state.skill_filter = refinements["skill_filter"]
    
    if "min_experience" in refinements:
        state.experience_range = (refinements["min_experience"], state.experience_range[1])
    
    if "max_experience" in refinements:
        state.experience_range = (state.experience_range[0], refinements["max_experience"])
    
    if "weights" in refinements:
        state.weights = refinements["weights"]
    
    # Re-score all candidates with updated constraints
    new_ranked = []
    for resume_path, agg_cand in state.aggregated_candidates.items():
        # Apply soft filters
        if state.skill_filter:
            candidate_skills = set([s.lower() for s in agg_cand.get("skills", [])])
            filter_skills = set([s.lower() for s in state.skill_filter])
            missing_skills = filter_skills - candidate_skills
            if missing_skills:
                # Soft penalty: reduce score
                continue  # For now, skip candidates with missing filtered skills
        
        if agg_cand.get("experience_years", 0) < state.experience_range[0]:
            continue
        if agg_cand.get("experience_years", 0) > state.experience_range[1]:
            continue
        
        # Recompute score with new weights
        score_info = compute_score(agg_cand, state.parsed_requirements, state.weights)
        
        new_ranked.append({
            "candidate_name": agg_cand.get("candidate_name", "Unknown"),
            "resume_path": resume_path,
            "match_score": score_info["match_score"],
            "matched_skills": score_info["matched_skills"],
            "relevant_excerpts": agg_cand.get("chunks", [])[:3],
            "reasoning": f"Refined scoring applied with constraints: {list(refinements.keys())}",
            "semantic_score": score_info["semantic_score"],
            "skills_score": score_info["skills_score"],
            "must_have_score": score_info["must_have_score"],
        })
    
    # Sort by match score descending
    new_ranked.sort(key=lambda x: x["match_score"], reverse=True)
    new_ranked = new_ranked[:state.top_k]
    
    # Build delta explanation
    delta_lines = ["📊 **Reranking Summary**\n"]
    for i, (old, new_candidate) in enumerate(zip(before_ranked[:3], new_ranked[:3])):
        old_score = before_scores.get(new_candidate["resume_path"], 0)
        new_score = new_candidate["match_score"]
        delta = new_score - old_score
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        delta_lines.append(
            f"{arrow} {new_candidate['candidate_name']}: "
            f"{old_score} → {new_score} ({delta:+d})"
        )
    
    delta_explanation = "\n".join(delta_lines)
    
    return new_ranked, delta_explanation


# ============================================================================
# Helper: Candidate Lookup
# ============================================================================

def build_candidate_lookup_index(state: AgentState) -> Dict[str, str]:
    """Build fast lookup: candidate_name/path -> resume_path."""
    index = {}
    for cand in state.shortlist:
        name = cand.get("candidate_name", "")
        path = cand.get("resume_path", "")
        if name:
            index[name.lower()] = path
        if path:
            index[path] = path
    return index


def find_candidate_by_pattern(pattern: str, state: AgentState) -> Optional[Dict[str, Any]]:
    """Find candidate by name pattern or partial match."""
    pattern_lower = pattern.lower()
    
    for cand in state.shortlist:
        name = cand.get("candidate_name", "").lower()
        path = cand.get("resume_path", "").lower()
        
        if pattern_lower in name or pattern_lower in path:
            return cand
    
    return None


# ============================================================================
# Helper: Format output for display
# ============================================================================

def format_candidate_summary(cand: Dict[str, Any]) -> str:
    """Format candidate for display."""
    name = cand.get("candidate_name", "Unknown")
    score = cand.get("match_score", 0)
    skills = ", ".join(cand.get("matched_skills", [])[:3])
    reasoning = cand.get("reasoning", "")
    
    return f"{name} (Score: {score}/100, Skills: {skills})"


def format_comparison_table(comparison: List[Dict[str, Any]]) -> str:
    """Format comparison as text table."""
    lines = ["| Candidate | Skills | Experience | Score |\n", "|---|---|---|---|\n"]
    
    for cand in comparison:
        name = cand.get("candidate_name", "Unknown")
        skills = ", ".join(cand.get("matched_skills", [])[:2])
        exp = f"{cand.get('experience_years', 0):.1f}y"
        score = cand.get("match_score", 0)
        lines.append(f"| {name} | {skills} | {exp} | {score}/100 |\n")
    
    return "".join(lines)
