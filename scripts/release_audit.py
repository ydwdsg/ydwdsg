"""Audit redistributable files; no network, scientific solver or secret access.

This is an allowlist plus conservative pattern/size scan, not a security proof.
Review the final archive and owner authorization before any public push.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SKIP={".git",".venv","__pycache__",".pytest_cache","build","dist","local-results"}
TOP_FILES={"README.md","LICENSE","NOTICE","THIRD_PARTY_NOTICES.md","CHANGELOG.md","MANIFEST.in","pyproject.toml",".gitignore"}
TOP_DIRS={"src","tests","examples","benchmarks","scripts","docs","licenses","requirements",".github"}
FORBIDDEN_SUFFIXES={".h5",".hdf5",".npy",".npz",".so",".dll",".pyd",".exe",".key",".pem"}
PATTERNS=[
    ("private absolute user path",re.compile(r"[A-Z]:[/\\](?:Users|xwechat_files)[/\\]",re.I)),
    ("Unix account home",re.compile(r"/home/[A-Za-z0-9_-]+/")),
    ("private network address",re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")),
    ("SSH account",re.compile(r"\b[A-Za-z0-9_-]+@(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("private key",re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("GitHub credential",re.compile(r"\bgh[pousr]_[A-Za-z0-9]{25,}\b|github_pat_[A-Za-z0-9_]{30,}")),
]


def public_files(root=ROOT):
    for path in sorted(root.rglob("*")):
        rel=path.relative_to(root)
        if any(p in SKIP or p.endswith(".egg-info") for p in rel.parts):continue
        if path.is_file():yield path


def audit(root=ROOT):
    errors=[]; manifest=[]
    for path in public_files(root):
        rel=path.relative_to(root)
        if (len(rel.parts)==1 and rel.name not in TOP_FILES) or (len(rel.parts)>1 and rel.parts[0] not in TOP_DIRS):
            errors.append(f"Unlisted release path: {rel.as_posix()}")
        if path.is_symlink():errors.append(f"Symlink forbidden: {rel.as_posix()}")
        blob=path.read_bytes()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or len(blob)>2*1024**2:
            errors.append(f"Binary/scientific asset or oversized file: {rel.as_posix()}")
        allowed_binary=rel.parts[:2]==("docs","figures") and path.suffix in {".png",".pdf"}
        if not allowed_binary:
            try:content=blob.decode("utf-8")
            except UnicodeDecodeError:
                errors.append(f"Unexpected binary: {rel.as_posix()}");content=""
            for name,pattern in PATTERNS:
                if pattern.search(content):errors.append(f"{name}: {rel.as_posix()}")
        manifest.append(dict(path=rel.as_posix(),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    return dict(passed=not errors,errors=errors,file_count=len(manifest),files=manifest,
                scope="allowlist and text pattern/size scan; not a security guarantee")


if __name__=="__main__":
    report=audit()
    print(json.dumps({k:v for k,v in report.items() if k!="files"},indent=2))
    if not report["passed"]:raise SystemExit(2)
