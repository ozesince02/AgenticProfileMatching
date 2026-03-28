# Agentic Profile Matching

Agentic Profile Matching is a LangGraph-powered hiring assistant built on top of a RAG resume-matching pipeline.

It combines:
- semantic retrieval over resume chunks,
- hybrid scoring for shortlist quality,
- conversational refinement through a CLI,
- multi-round screening with explainable Hire/Hold/No-Hire outputs.

## What This Project Solves

Given a job description, the system:
1. parses requirements,
2. searches indexed resumes,
3. ranks candidates with transparent scoring,
4. allows iterative refinements from recruiter feedback,
5. performs deeper interview-oriented analysis,
6. generates final recommendations.

## Architecture Overview

### Core RAG Layer
- Resume ingestion and chunk indexing in ChromaDB
- Embeddings via sentence-transformers (`all-MiniLM-L6-v2`)
- Hybrid candidate scoring in the existing matcher

### Agentic Orchestration Layer
- LangGraph state machine for deterministic workflow
- Pydantic state for auditability and state snapshots
- CLI intent routing for interactive refinement loops

### High-Level Flow

```text
START
  -> parse_jd
  -> extract_requirements
  -> search_resumes
  -> rank_candidates
  -> generate_report
  -> human_feedback_loop
       -> (refine) back to extract_requirements/rank_candidates
       -> (compare/explain/questions) report path
       -> (round2) deep analysis
       -> (round3/finalize) final recommendations
  -> END
```

## Project Structure

```text
AgenticProfileMatching/
├── data/
│   ├── jobs/                       # Job descriptions
│   ├── resumes/                    # Candidate resumes
│   └── vector_db/                  # Chroma persisted index
├── fs_tools.py                     # Safe file operations (sandboxed to data/)
├── resume_rag.py                   # Index build/rebuild
├── job_matcher.py                  # RAG retrieval + hybrid scoring
├── agent_state.py                  # Agent state schema (Pydantic)
├── agent_tools.py                  # Tool wrappers + utility helpers
├── matching_agent.py               # LangGraph orchestration
├── chat_cli.py                     # Interactive conversational CLI
├── state_machine.md                # State machine documentation
├── test_scenarios.md               # Test flows and expected behavior
├── demo_runbook.md                 # 5-6 minute demo script
└── requirements.txt
```

## Setup

### 1. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Gemini Configuration (Optional but Recommended)

Gemini is used for richer requirement normalization and interview-question generation.

### Option A: Environment variable (PowerShell)

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
$env:GEMINI_MODEL="gemini-pro"
```

### Option B: `.env` file in project root

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-pro
```

### Fallback behavior

If Gemini key is missing or API call fails, the system still works:
- requirement parsing falls back to rule-based extraction,
- interview questions fall back to deterministic templates,
- ranking and multi-round logic remain available.

## Usage

### A) Build or rebuild vector index

```powershell
python resume_rag.py --rebuild
```

### B) Baseline matcher (non-agent path)

```powershell
python job_matcher.py --job-file jobs/19_ml_engineer_saas.txt
```

Or pass raw text:

```powershell
python job_matcher.py --job-text "Looking for a Python ML engineer with 4+ years and AWS experience"
```

### C) Start the agentic CLI (recommended)

```powershell
python chat_cli.py
```

## Interactive CLI Capabilities

### Natural-language intents
- Search candidates for a role
- Refine filters (skills, constraints, preferences)
- Compare candidates side-by-side
- Explain ranking decisions
- Generate interview questions
- Trigger Round 2 and Round 3 workflows

### Operator commands
- `/help` : show available commands
- `/new_jd` : reset session and load a new job description
- `/show_state` : print state summary
- `/top N` : show top N candidates
- `/history` : show refinement history
- `/round2` : run deep analysis
- `/round3` : run final recommendation stage
- `/export` : export state/results to JSON
- `/exit` : close session

## Multi-Round Screening Logic

### Round 1: Initial shortlist
- Retrieves top candidates via semantic + hybrid scoring
- Standard score blend:

```text
final_score = 0.60 * semantic_similarity
            + 0.25 * skill_overlap_ratio
            + 0.15 * must_have_satisfaction
```

### Round 2: Deep analysis
- Generates candidate-specific interview focus areas
- Surfaces strengths, gaps, and possible risk flags
- Supports recruiter-driven follow-up

### Round 3: Final recommendation
- Applies balanced rubric for decisioning:

```text
30% semantic fit
25% skill coverage
20% experience alignment
15% must-have satisfaction
10% interview signals
```

- Recommendation thresholds:
  - Hire: score > 80
  - Hold: 60 <= score <= 80
  - No-Hire: score < 60

## Output Contract (Example)

```json
{
  "candidate_name": "Sneha Nair",
  "resume_path": "resumes/14_sneha_nair_ml_engineer.txt",
  "match_score": 84,
  "matched_skills": ["Pytorch", "Tensorflow", "Aws"],
  "relevant_excerpts": ["..."],
  "reasoning": "Matched core skills and strong semantic relevance.",
  "recommendation": "Hire"
}
```

## Explainability and Audit Trail

The system keeps transparent decision context in state:
- conversation history,
- refinement history,
- scoring breakdowns,
- explanation traces,
- round transitions.

This enables reproducibility for grading, debugging, and stakeholder review.

## Testing and Validation

Use these artifacts:
- `test_scenarios.md` for scenario-by-scenario checks,
- `state_machine.md` for transition and node behavior,
- `demo_runbook.md` for a timed walkthrough,
- `notebook_experiments.ipynb` for RAG quality and latency analysis.

## Submission Artifacts Checklist

- `agent_state.py`
- `agent_tools.py`
- `matching_agent.py`
- `chat_cli.py`
- `state_machine.md`
- `test_scenarios.md`
- `demo_runbook.md`
- `README.md`

## Known Limitations

- Gemini enrichment depends on network/API availability.
- Retrieval quality depends on resume text quality and skill normalization.
- Current workflow is CLI-first; UI/web front-end is not included.

## Quick Start (Minimal)

```powershell
pip install -r requirements.txt
python resume_rag.py --rebuild
python chat_cli.py
```

Then try:

```text
Search for ML engineers in SaaS with PyTorch and AWS
Filter by Docker and 5+ years
Compare top 3 candidates
/round2
/round3
/export
```

---

Built for the Agentic Profile Matching assignment: architecture + interactive refinement + advanced multi-round decision support.
