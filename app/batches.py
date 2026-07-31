"""Batch intake: unpack upload, validate the manifest, queue the work.

The brief's contract: audio files at the folder root plus one CSV manifest
with `name` (exact filename) and `result_json` (expected labels; may be
empty). Validation reports, without failing the batch: rows without files,
files without rows, unparseable manifest rows, unsupported extensions.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import uuid
import zipfile
from pathlib import Path

from . import db, worker

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


def _safe_name(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def _parse_manifest(raw: bytes) -> tuple[dict[str, str], list[str]]:
    """filename -> expected result_json string ('' when unlabeled)."""
    warnings: list[str] = []
    text = raw.decode("utf-8-sig", errors="replace")
    rows: dict[str, str] = {}
    reader = csv.DictReader(io.StringIO(text))
    cols = [c.strip().lower() for c in reader.fieldnames or []]
    if "name" not in cols:
        return {}, [f"manifest has no 'name' column (found: {cols}); "
                    "all audio files will be processed without labels"]
    for i, row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
        name = _safe_name(row.get("name", "").strip())
        if not name:
            warnings.append(f"manifest line {i}: empty name, skipped")
            continue
        rj = row.get("result_json", "").strip()
        if rj:
            try:
                json.loads(rj)
            except json.JSONDecodeError:
                warnings.append(f"manifest line {i} ({name}): result_json is not "
                                "valid JSON; treated as unlabeled")
                rj = ""
        rows[name] = rj
    return rows, warnings


def create_batch_from_upload(files: list[tuple[str, bytes]], batch_name: str) -> str:
    """`files` = (filename, content) pairs: either one ZIP or a loose set of
    audio files + manifest.csv. Returns the batch id."""
    staged: dict[str, bytes] = {}
    for fname, content in files:
        name = _safe_name(fname)
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                for info in z.infolist():
                    if info.is_dir() or info.file_size > 500_000_000:
                        continue
                    inner = _safe_name(info.filename)
                    if inner and not inner.startswith("."):
                        staged[inner] = z.read(info)
        else:
            staged[name] = content

    manifests = [n for n in staged if n.lower().endswith(".csv")]
    warnings: list[str] = []
    manifest: dict[str, str] = {}
    if not manifests:
        warnings.append("no CSV manifest found; processing every audio file unlabeled")
    else:
        if len(manifests) > 1:
            warnings.append(f"multiple CSVs found ({manifests}); using {manifests[0]}")
        rows, w = _parse_manifest(staged[manifests[0]])
        manifest, warnings = rows, warnings + w

    audio = {n: c for n, c in staged.items()
             if Path(n).suffix.lower() in worker.AUDIO_EXT}
    ignored = [n for n in staged
               if n not in audio and not n.lower().endswith(".csv")]
    if ignored:
        warnings.append(f"ignored non-audio files: {sorted(ignored)[:10]}")

    for name in sorted(manifest):
        if name not in audio:
            warnings.append(f"manifest row '{name}' has no matching audio file")

    batch_id = db.create_batch(batch_name or f"batch-{uuid.uuid4().hex[:6]}", warnings)
    bdir = DATA_DIR / "batches" / batch_id
    bdir.mkdir(parents=True, exist_ok=True)

    for name in sorted(audio):
        p = bdir / name
        p.write_bytes(audio[name])
        expected = manifest.get(name, "")
        if manifest and name not in manifest:
            warnings.append(f"audio file '{name}' not present in manifest; processed anyway")
        fid = db.add_file(batch_id, name, str(p), expected or None)
        worker.submit(batch_id, fid, str(p))

    if not audio:
        db.con().execute("UPDATE batches SET status = 'done', warnings = ? WHERE id = ?",
                         (json.dumps(warnings + ["no audio files in upload"]), batch_id))
        db.con().commit()
    else:
        db.con().execute("UPDATE batches SET warnings = ? WHERE id = ?",
                         (json.dumps(warnings), batch_id))
        db.con().commit()
    return batch_id
