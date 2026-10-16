"""Build a LOCAL release directory from audited public files. Never push or publish."""
import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from release_audit import ROOT, PATTERNS, FORBIDDEN_SUFFIXES, audit, public_files


def inspect_archive(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members=[(x.filename,archive.read(x)) for x in archive.infolist() if not x.is_dir()]
    else:
        with tarfile.open(path,"r:gz") as archive:
            members=[]
            for x in archive.getmembers():
                if x.issym() or x.islnk():raise ValueError("Archive links forbidden")
                if x.isfile():members.append((x.name,archive.extractfile(x).read()))
    for name,blob in members:
        rel=Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.suffix in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Forbidden archive member: {name}")
        if len(blob)>2*1024**2:raise ValueError(f"Oversized member: {name}")
        if rel.suffix not in {".png",".pdf"}:
            content=blob.decode("utf-8")
            if any(pattern.search(content) for _,pattern in PATTERNS):
                raise ValueError(f"Sensitive pattern in archive: {name}")
    return len(members)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True,help="New, nonexistent directory outside repository")
    args=p.parse_args()
    out=args.out.resolve()
    if out==ROOT or ROOT in out.parents or out.exists():p.error("Output must be new and outside repository")
    report=audit()
    if not report["passed"]:raise RuntimeError(report["errors"])
    artifacts=[ROOT/"dist/phono3py_cupy_accelerator-0.1.0-py3-none-any.whl",
               ROOT/"dist/phono3py_cupy_accelerator-0.1.0.tar.gz"]
    for a in artifacts:
        if not a.is_file():raise FileNotFoundError("Run python -m build first")
        inspect_archive(a)
    out.mkdir(parents=True)
    for a in artifacts:shutil.copy2(a,out/a.name)
    source=out/"phono3py-cupy-accelerator-0.1.0-source.zip"
    with zipfile.ZipFile(source,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for path in public_files():
            z.write(path,"phono3py-cupy-accelerator/"+path.relative_to(ROOT).as_posix())
    inspect_archive(source)
    manifest=dict(version="0.1.0",status="LOCAL_EXPERIMENTAL_NOT_PUBLISHED",
                  public_source=report,production_modified=False,public_push=False,
                  omitted="private data, accounts, paths, models, FC, HDF5, environments, Git metadata")
    (out/"RELEASE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    lines=[]
    for a in sorted(out.iterdir()):
        if a.is_file():lines.append(hashlib.sha256(a.read_bytes()).hexdigest()+"  "+a.name)
    (out/"SHA256SUMS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="LOCAL_ONLY",public_file_count=report["file_count"],
                         artifacts=[a.name for a in sorted(out.iterdir())]),indent=2))


if __name__=="__main__":main()
