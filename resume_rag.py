import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from fs_tools import list_files, read_file

SECTION_HEADERS = [
    "Summary",
    "Professional Summary",
    "Experience",
    "Work Experience",
    "Skills",
    "Projects",
    "Education",
    "Certifications",
]

SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "react", "next.js", "node.js",
    "fastapi", "django", "flask", "sql", "postgresql", "mongodb", "redis", "spark",
    "airflow", "kafka", "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "jenkins", "pytorch", "tensorflow", "scikit-learn", "nlp", "rag", "llms",
    "power bi", "tableau", "excel", "selenium", "cypress", "figma", "dart", "flutter",
    "kotlin", "ios", "android", "linux", "bash", "siem", "owasp", "agile",
    "requirements gathering", "stakeholder management", "jira", "confluence",
    "user stories", "a/b testing", "roadmapping", "wireframing", "prototyping",
    "user research", "design systems", "test planning", "api testing", "postman",
    "network security", "soc", "incident response", "spacy", "nltk",
    "transformers", "vector databases", "etl", "ci/cd", "mlops", "statistics",
    "pandas", "numpy", "communication",
]

DEGREE_KEYWORDS = [
    "b.tech", "b.e.", "m.tech", "mca", "b.sc", "m.sc", "mba", "phd", "bachelor", "master",
]


@dataclass
class ResumeDocument:
    resume_path: str
    filename: str
    raw_text: str


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def infer_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].isdigit():
        name_tokens = parts[1:3]
    else:
        name_tokens = parts[:2]
    return " ".join(token.capitalize() for token in name_tokens)


def load_resumes() -> List[ResumeDocument]:
    resumes = []
    for item in list_files("resumes"):
        filename = item.get("name", "")
        if not filename:
            continue

        relative_path = f"resumes/{filename}"
        result = read_file(relative_path)
        if not result.get("success"):
            continue

        content = (result.get("content") or "").strip()
        if not content:
            continue

        resumes.append(
            ResumeDocument(
                resume_path=relative_path,
                filename=filename,
                raw_text=content,
            )
        )
    return resumes


def split_sections(text: str) -> List[Tuple[str, str]]:
    lines = [line.rstrip() for line in text.splitlines()]
    sections: List[Tuple[str, List[str]]] = []
    current_header = "General"
    current_lines: List[str] = []

    header_pattern = re.compile(r"^([A-Za-z][A-Za-z\s/&-]{1,40})\s*:\s*$")
    known_headers = {h.lower() for h in SECTION_HEADERS}

    for line in lines:
        stripped = line.strip()
        header_match = header_pattern.match(stripped)
        if header_match and header_match.group(1).strip().lower() in known_headers:
            if current_lines:
                sections.append((current_header, current_lines))
            current_header = header_match.group(1).strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_header, current_lines))

    return [(header, "\n".join(content).strip()) for header, content in sections if "\n".join(content).strip()]


def chunk_text(text: str, chunk_size: int = 750, overlap: int = 120) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - overlap)

    return chunks


def extract_name(text: str, fallback_filename: str) -> str:
    match = re.search(r"(?im)^name\s*:\s*(.+)$", text)
    if match:
        return normalize_text(match.group(1))
    return infer_name_from_filename(fallback_filename)


def extract_skills(text: str) -> List[str]:
    lowered_text = text.lower()
    found = {skill for skill in SKILL_VOCAB if skill in lowered_text}

    skills_section_match = re.search(r"(?is)skills\s*:\s*(.+?)(\n\s*\n|$)", text)
    if skills_section_match:
        section_body = skills_section_match.group(1)
        for token in re.split(r"[,;\n]", section_body):
            cleaned = normalize_text(token).lower()
            if cleaned:
                found.add(cleaned)

    return sorted({skill.title() for skill in found})


def extract_experience_years(text: str) -> float:
    matches = re.findall(r"(\d{1,2})\s*\+?\s*years", text, flags=re.IGNORECASE)
    if not matches:
        return 0.0
    values = [float(match) for match in matches]
    return max(values)


def extract_education(text: str) -> List[str]:
    found = []
    lowered = text.lower()
    for keyword in DEGREE_KEYWORDS:
        if keyword in lowered:
            found.append(keyword.upper())
    return sorted(set(found))


def extract_resume_metadata(text: str, filename: str) -> Dict:
    return {
        "candidate_name": extract_name(text, filename),
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
    }


def build_chunks(resume: ResumeDocument, chunk_size: int, overlap: int) -> List[Dict]:
    sections = split_sections(resume.raw_text)
    base_metadata = extract_resume_metadata(resume.raw_text, resume.filename)
    records: List[Dict] = []

    for section_name, section_text in sections:
        section_chunks = chunk_text(section_text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(section_chunks):
            records.append(
                {
                    "id": f"{Path(resume.filename).stem}::{section_name.lower()}::{idx}",
                    "document": chunk,
                    "metadata": {
                        **base_metadata,
                        "resume_path": resume.resume_path,
                        "filename": resume.filename,
                        "section": section_name,
                        "chunk_index": idx,
                    },
                }
            )

    return records


def get_collection(client: chromadb.PersistentClient, collection_name: str, rebuild: bool):
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(name=collection_name)


def build_index(
    model_name: str = "all-MiniLM-L6-v2",
    persist_dir: str = "data/vector_db",
    collection_name: str = "resumes",
    chunk_size: int = 750,
    overlap: int = 120,
    rebuild: bool = False,
) -> Dict:
    resumes = load_resumes()
    if not resumes:
        raise RuntimeError("No resumes found in data/resumes")

    all_records: List[Dict] = []
    for resume in resumes:
        all_records.extend(build_chunks(resume, chunk_size=chunk_size, overlap=overlap))

    if not all_records:
        raise RuntimeError("No chunks generated from resumes")

    embedder = SentenceTransformer(model_name)
    documents = [record["document"] for record in all_records]
    embeddings = embedder.encode(documents, normalize_embeddings=True, show_progress_bar=True)

    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = get_collection(client, collection_name, rebuild=rebuild)

    collection.add(
        ids=[record["id"] for record in all_records],
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                **record["metadata"],
                "skills": json.dumps(record["metadata"].get("skills", [])),
                "education": json.dumps(record["metadata"].get("education", [])),
            }
            for record in all_records
        ],
    )

    return {
        "resumes_indexed": len(resumes),
        "chunks_indexed": len(all_records),
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "embedding_model": model_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build resume RAG vector index.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--persist-dir", default="data/vector_db", help="Chroma persistence directory")
    parser.add_argument("--collection", default="resumes", help="Chroma collection name")
    parser.add_argument("--chunk-size", type=int, default=750, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild collection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_index(
        model_name=args.model,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        rebuild=args.rebuild,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
