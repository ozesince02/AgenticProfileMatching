import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from fs_tools import list_files, read_file
from resume_rag import SKILL_VOCAB, normalize_text


def parse_list_section(text: str, heading: str) -> List[str]:
    pattern = rf"(?is){re.escape(heading)}\s*:\s*(.+?)(?:\n\s*[A-Za-z][A-Za-z\s\-/&]+\s*:|$)"
    match = re.search(pattern, text)
    if not match:
        return []

    body = match.group(1)
    items: List[str] = []
    for line in body.splitlines():
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if cleaned:
            items.append(cleaned)
    return items


def extract_skills_from_text(text: str) -> List[str]:
    lowered = text.lower()
    found = [skill for skill in sorted(SKILL_VOCAB) if skill in lowered]
    return sorted(set(found))


def parse_job_description(job_text: str) -> Dict:
    must_have = [normalize_text(item).lower() for item in parse_list_section(job_text, "Must-Have Skills")]
    good_to_have = [normalize_text(item).lower() for item in parse_list_section(job_text, "Good-to-Have Skills")]
    inferred = extract_skills_from_text(job_text)

    experience_match = re.search(r"(\d{1,2})\s*\+?\s*years", job_text, flags=re.IGNORECASE)
    min_experience = float(experience_match.group(1)) if experience_match else 0.0

    return {
        "must_have_skills": sorted(set(must_have)),
        "good_to_have_skills": sorted(set(good_to_have)),
        "all_skills": sorted(set(must_have + good_to_have + inferred)),
        "min_experience_years": min_experience,
    }


def load_job_description(job_text: str = "", job_file: str = "") -> str:
    if job_text.strip():
        return job_text.strip()

    if job_file:
        result = read_file(job_file)
        if not result.get("success"):
            raise ValueError(result.get("error") or "Failed to read job file")
        content = (result.get("content") or "").strip()
        if not content:
            raise ValueError(f"Empty job description file: {job_file}")
        return content

    jobs = list_files("jobs")
    if not jobs:
        raise ValueError("No job descriptions found in data/jobs and no --job-text/--job-file provided")

    first_job = jobs[0].get("name")
    result = read_file(f"jobs/{first_job}")
    if not result.get("success"):
        raise ValueError(result.get("error") or "Failed to read default job file")
    return (result.get("content") or "").strip()


def cosine_like_from_distance(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def parse_candidate_skills(value) -> List[str]:
    if isinstance(value, list):
        return [normalize_text(skill).lower() for skill in value if normalize_text(skill)]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [normalize_text(skill).lower() for skill in parsed if normalize_text(skill)]
        except json.JSONDecodeError:
            pass

    return []


def aggregate_candidates(query_result: Dict) -> Dict[str, Dict]:
    by_candidate: Dict[str, Dict] = defaultdict(lambda: {
        "candidate_name": "Unknown",
        "resume_path": "",
        "distances": [],
        "chunks": [],
        "skills": set(),
        "experience_years": 0.0,
    })

    docs = query_result.get("documents", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]
    distances = query_result.get("distances", [[]])[0]

    for doc, metadata, distance in zip(docs, metadatas, distances):
        candidate_key = metadata.get("resume_path", metadata.get("filename", "unknown"))
        candidate = by_candidate[candidate_key]

        candidate["candidate_name"] = metadata.get("candidate_name", candidate["candidate_name"])
        candidate["resume_path"] = metadata.get("resume_path", candidate["resume_path"])
        candidate["distances"].append(float(distance))
        candidate["chunks"].append(str(doc))
        candidate["experience_years"] = max(candidate["experience_years"], float(metadata.get("experience_years", 0.0)))

        for skill in parse_candidate_skills(metadata.get("skills", [])):
            candidate["skills"].add(skill)

    return by_candidate


def compute_score(candidate: Dict, job_meta: Dict, weights: Tuple[float, float, float]) -> Dict:
    semantic_score = 0.0
    if candidate["distances"]:
        semantic_score = sum(cosine_like_from_distance(d) for d in candidate["distances"]) / len(candidate["distances"])

    candidate_skills = candidate["skills"]
    required_skills = set(job_meta["all_skills"])
    must_have = set(job_meta["must_have_skills"])
    min_experience = float(job_meta["min_experience_years"])

    if required_skills:
        skill_overlap = len(candidate_skills.intersection(required_skills)) / len(required_skills)
    else:
        skill_overlap = 0.0

    if must_have:
        must_have_match_ratio = len(candidate_skills.intersection(must_have)) / len(must_have)
    else:
        must_have_match_ratio = 1.0

    if min_experience > 0:
        exp_ratio = min(candidate.get("experience_years", 0.0) / min_experience, 1.0)
    else:
        exp_ratio = 1.0

    must_have_score = (0.7 * must_have_match_ratio) + (0.3 * exp_ratio)

    final_score = (
        (weights[0] * semantic_score)
        + (weights[1] * skill_overlap)
        + (weights[2] * must_have_score)
    ) * 100

    matched_skills = sorted(candidate_skills.intersection(required_skills))
    return {
        "semantic_score": semantic_score,
        "skills_score": skill_overlap,
        "must_have_score": must_have_score,
        "match_score": max(0, min(100, round(final_score))),
        "matched_skills": [skill.title() for skill in matched_skills],
    }


def build_reasoning(candidate: Dict, score_info: Dict, job_meta: Dict) -> str:
    matched = score_info["matched_skills"]
    semantic_pct = round(score_info["semantic_score"] * 100)
    exp = candidate.get("experience_years", 0.0)
    min_exp = job_meta.get("min_experience_years", 0.0)

    if matched:
        skills_part = f"Matched core skills: {', '.join(matched[:5])}."
    else:
        skills_part = "Limited direct skill overlap detected."

    exp_part = f" Experience profile: {exp:.1f} years vs required {min_exp:.1f} years."
    semantic_part = f" Semantic relevance is {semantic_pct}% based on top resume chunks."

    return f"{skills_part}{exp_part}{semantic_part}".strip()


def _filter_and_score(
    aggregated: Dict[str, Dict],
    job_meta: Dict,
    weights: Tuple[float, float, float],
    strict: bool = True,
) -> List[Dict]:
    """Score candidates, applying hard must-have filter in strict mode only.

    Experience is always a soft penalty, never a hard filter.
    """
    hard_required = set(job_meta["must_have_skills"]) if strict else set()

    ranked = []
    for candidate in aggregated.values():
        candidate_skills = candidate["skills"]

        # Hard filter: candidate must possess ALL must-have skills (strict mode only)
        if hard_required and not hard_required.issubset(candidate_skills):
            continue

        score_info = compute_score(candidate, job_meta, weights)
        entry = {
            "candidate_name": candidate["candidate_name"],
            "resume_path": candidate["resume_path"],
            "match_score": score_info["match_score"],
            "matched_skills": score_info["matched_skills"],
            "relevant_excerpts": candidate["chunks"][:3],
            "reasoning": build_reasoning(candidate, score_info, job_meta),
        }
        if not strict:
            entry["fallback_mode"] = True
        ranked.append(entry)

    ranked.sort(key=lambda item: item["match_score"], reverse=True)
    return ranked


def match_candidates(
    job_description: str,
    persist_dir: str = "data/vector_db",
    collection_name: str = "resumes",
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 10,
    top_chunks: int = 120,
    weights: Tuple[float, float, float] = (0.60, 0.25, 0.15),
) -> Dict:
    job_meta = parse_job_description(job_description)

    embedder = SentenceTransformer(model_name)
    query_embedding = embedder.encode([job_description], normalize_embeddings=True)[0].tolist()

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)

    # Clamp top_chunks to actual collection size to avoid Chroma errors
    total_chunks = collection.count()
    n_results = min(top_chunks, total_chunks)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    aggregated = aggregate_candidates(result)

    # --- Strict pass: must-have hard filter, experience as soft penalty ---
    ranked = _filter_and_score(aggregated, job_meta, weights, strict=True)

    # --- Fallback: if strict returns nothing, score everyone without hard filter ---
    if not ranked:
        ranked = _filter_and_score(aggregated, job_meta, weights, strict=False)

    return {
        "job_description": job_description,
        "top_matches": ranked[:top_k],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match job descriptions against indexed resumes")
    parser.add_argument("--job-file", default="", help="Relative job file path under data/, e.g. jobs/01_ml_engineer_saas.txt")
    parser.add_argument("--job-text", default="", help="Raw job description text")
    parser.add_argument("--persist-dir", default="data/vector_db", help="Chroma persistence directory")
    parser.add_argument("--collection", default="resumes", help="Chroma collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--top-k", type=int, default=10, help="Top candidates to return")
    parser.add_argument("--top-chunks", type=int, default=120, help="Top chunks to retrieve before candidate aggregation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_description = load_job_description(job_text=args.job_text, job_file=args.job_file)

    result = match_candidates(
        job_description=job_description,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        top_k=args.top_k,
        top_chunks=args.top_chunks,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
