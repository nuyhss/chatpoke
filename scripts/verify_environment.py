#!/usr/bin/env python3
"""
Verify local project setup for the Tilon modular FastAPI app.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"
ENV_EXAMPLE_FILE = ROOT_DIR / ".env.example"

REQUIRED_PACKAGES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
    "requests": "requests",
    "langchain_core": "langchain-core",
    "langchain_text_splitters": "langchain-text-splitters",
    "langchain_chroma": "langchain-chroma",
    "langchain_huggingface": "langchain-huggingface",
    "sentence_transformers": "sentence-transformers",
    "chromadb": "chromadb",
    "fitz": "PyMuPDF",
    "PIL": "Pillow",
    "pytesseract": "pytesseract",
    "pdf2image": "pdf2image",
    "langdetect": "langdetect",
}

OPTIONAL_PACKAGES = {
    "FlagEmbedding": "FlagEmbedding (reranker)",
    "tavily": "tavily-python (web search)",
}


def _status(ok: bool, label: str, detail: str = "") -> str:
    prefix = "[ok]" if ok else "[missing]"
    suffix = f" - {detail}" if detail else ""
    return f"{prefix} {label}{suffix}"


def _find_command(command: str) -> str | None:
    venv_path = Path(sys.executable).resolve().parent / command
    if venv_path.exists():
        return str(venv_path)
    return shutil.which(command)


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _python_check() -> list[str]:
    lines = []
    version_ok = sys.version_info >= (3, 10)
    version_text = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    lines.append(_status(version_ok, "Python", version_text))

    for module_name, package_name in REQUIRED_PACKAGES.items():
        found = importlib.util.find_spec(module_name) is not None
        lines.append(_status(found, package_name))

    for module_name, label in OPTIONAL_PACKAGES.items():
        found = importlib.util.find_spec(module_name) is not None
        lines.append(_status(found, label))

    return lines


def _binary_check() -> list[str]:
    lines = []
    for command, label in (
        ("ollama", "Ollama CLI"),
        ("pdftoppm", "Poppler pdftoppm"),
        ("tesseract", "Tesseract OCR"),
        ("marker_single", "marker-pdf CLI"),
    ):
        path = _find_command(command)
        lines.append(_status(path is not None, label, path or "not found in PATH"))
    return lines


def _resolve_expected_models() -> list[str]:
    env_values = _load_env_file(ENV_FILE)
    if not env_values:
        env_values = _load_env_file(ENV_EXAMPLE_FILE)

    model = env_values.get("OLLAMA_MODEL", "qwen2.5:7b").strip()
    available = env_values.get(
        "AVAILABLE_MODELS",
        "qwen2.5:7b,llama3.1:latest,llama3.2-vision:11b",
    )
    items = [model]
    items.extend(part.strip() for part in available.split(",") if part.strip())

    ordered: list[str] = []
    seen = set()
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _ollama_check() -> list[str]:
    ollama_path = _find_command("ollama")
    if ollama_path is None:
        return [_status(False, "Ollama models", "ollama binary not installed")]

    try:
        result = subprocess.run(
            [ollama_path, "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return [_status(False, "Ollama models", str(exc))]

    output = result.stdout
    return [_status(model in output, f"Ollama model {model}") for model in _resolve_expected_models()]


def main() -> int:
    print("Tilon environment verification")
    print(f"Project root: {ROOT_DIR}")
    print()

    env_path = ENV_FILE if ENV_FILE.exists() else ENV_EXAMPLE_FILE
    print(_status(env_path.exists(), "Environment file", str(env_path)))
    print()

    print("Python packages")
    for line in _python_check():
        print(line)
    print()

    print("System binaries")
    for line in _binary_check():
        print(line)
    print()

    print("Ollama models")
    for line in _ollama_check():
        print(line)

    missing_required = []
    if sys.version_info < (3, 10):
        missing_required.append("python")

    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing_required.append(package_name)

    for command in ("ollama", "pdftoppm"):
        if _find_command(command) is None:
            missing_required.append(command)

    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
