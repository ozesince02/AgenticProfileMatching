import os
from datetime import datetime
from typing import List, Dict, Optional

from PyPDF2 import PdfReader
from docx import Document

DATA_DIR = "data"


def _is_safe_path(filepath: str) -> bool:
    filepath = os.path.normpath(filepath)
    if os.path.sep not in filepath:
        return True
    if ".." in filepath:
        return False
    if filepath.startswith(os.path.sep) or os.path.isabs(filepath):
        return False
    return True


def _get_data_path(filepath: str) -> Optional[str]:
    if not _is_safe_path(filepath):
        return None
    full_path = os.path.join(DATA_DIR, filepath)
    full_path = os.path.normpath(full_path)
    try:
        abs_data = os.path.abspath(DATA_DIR)
        abs_full = os.path.abspath(full_path)
        if not abs_full.startswith(abs_data):
            return None
    except Exception:
        return None
    return full_path


def read_file(filepath: str) -> Dict:
    try:
        actual_path = _get_data_path(filepath)
        if not actual_path or not os.path.isfile(actual_path):
            return {"success": False, "error": f"Access denied or file not found: '{filepath}'."}

        ext = os.path.splitext(actual_path)[1].lower()
        content = ""

        if ext == ".txt":
            with open(actual_path, "r", encoding="utf-8") as f:
                content = f.read()

        elif ext == ".pdf":
            reader = PdfReader(actual_path)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif ext == ".docx":
            doc = Document(actual_path)
            content = "\n".join(para.text for para in doc.paragraphs)

        else:
            return {"success": False, "error": f"Unsupported file type: '{ext}'."}

        metadata = {
            "filename": os.path.basename(actual_path),
            "size_bytes": os.path.getsize(actual_path),
            "modified": datetime.fromtimestamp(os.path.getmtime(actual_path)).isoformat(),
            "extension": ext,
        }
        return {"success": True, "content": content, "metadata": metadata}

    except Exception as e:
        return {"success": False, "error": str(e)}


def list_files(directory: str, extension: Optional[str] = None) -> List[Dict]:
    safe_dir = _get_data_path(directory) if directory else os.path.join(DATA_DIR)
    if not safe_dir or not os.path.isdir(safe_dir):
        return []

    results = []
    for file in os.listdir(safe_dir):
        path = os.path.join(safe_dir, file)
        if not os.path.isfile(path):
            continue
        if extension and not file.lower().endswith(extension.lower()):
            continue
        results.append({
            "name": file,
            "size_bytes": os.path.getsize(path),
            "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
            "path": path,
        })
    return results


def write_file(filepath: str, content: str) -> Dict:
    try:
        safe_path = _get_data_path(filepath)
        if not safe_path:
            return {"success": False, "error": f"Access denied: '{filepath}'."}

        parent_dir = os.path.dirname(safe_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {"success": True, "path": safe_path}

    except Exception as e:
        return {"success": False, "error": str(e)}


def search_in_file(filepath: str, keyword: str) -> Dict:
    result = read_file(filepath)
    if not result.get("success"):
        return result

    text = result["content"]
    keyword_lower = keyword.lower()
    matches = []

    for i, line in enumerate(text.splitlines()):
        if keyword_lower in line.lower():
            matches.append({"line_number": i + 1, "line": line.strip()})

    return {
        "success": True,
        "matches_found": len(matches),
        "matches": matches,
    }