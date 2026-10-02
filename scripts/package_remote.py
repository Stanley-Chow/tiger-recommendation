"""Package source and prepared Beauty artifacts, excluding credentials and caches."""

import hashlib
import json
import tarfile
from pathlib import Path


def package(project_dir: Path, output_path: Path) -> dict:
    project_dir = project_dir.resolve()
    roots = [
        project_dir / name
        for name in ("src", "tests", "configs", "docs", "scripts", "data/processed")
    ]
    files = [
        project_dir / name
        for name in (
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "requirements-lock.txt",
            ".gitignore",
            ".gitattributes",
        )
    ]
    files.extend(
        project_dir / "artifacts/beauty" / name
        for name in ("embeddings.npy", "semantic_ids.json", "embedding_config.json")
    )
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and not path.name.endswith(".pyc")
        )
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("Prepare the project and Beauty artifacts before packaging.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tarfile.open(output_path, "w:gz") as archive:
        for path in sorted(set(files)):
            relative = path.relative_to(project_dir).as_posix()
            archive.add(path, arcname=relative, recursive=False)
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "files": manifest,
        "archive_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "archive_bytes": output_path.stat().st_size,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    result = package(project, project / ".remote/project.tar.gz")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
