"""CLI: local analysis of audio files.

    python -m autoace_pipeline.cli resources/audio/*.ogg [--no-vad] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .ingest import IngestError
from .pipeline import analyze_local


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    out = []
    for fp in args.files:
        t0 = time.perf_counter()
        try:
            feats, dec = analyze_local(fp, use_vad=not args.no_vad)
        except IngestError as e:
            print(f"=== {Path(fp).name}: FAILED ({e}) ===\n")
            out.append({"file": Path(fp).name, "error": str(e)})
            continue
        dt = time.perf_counter() - t0
        rtf = dt / max(feats.duration_s, 1e-6)
        print(f"=== {Path(fp).name}  ({feats.duration_s}s, {dt:.1f}s wall, RTF {rtf:.3f}) ===")
        for k, v in feats.model_dump().items():
            print(f"  {k:>26}: {v}")
        print("  --- local decisions ---")
        for k, v in vars(dec).items():
            print(f"  {k:>26}: {v}")
        print()
        out.append({"file": Path(fp).name, "features": feats.model_dump(),
                    "decisions": vars(dec), "wall_s": round(dt, 2)})

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
