"""Background batch processing.

A small thread pool runs the full analysis per file. One malformed file
fails alone — its row records the error and the batch continues; that
behavior is a stated requirement. Local torch models are inference-only
singletons shared across two workers; the LLM call inside each analysis
overlaps I/O so two workers keep both CPU and API busy.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback

# duplicate OpenMP runtimes (torch + onnx + funasr) abort the process on
# Windows without a traceback; this is the documented workaround
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from autoace_pipeline.ingest import IngestError

from . import db

_q: "queue.Queue[tuple[str, int, str]]" = queue.Queue()
_started = False

AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".webm", ".aiff"}


def submit(batch_id: str, file_id: int, path: str) -> None:
    _q.put((batch_id, file_id, path))


def _process(batch_id: str, file_id: int, path: str) -> None:
    from autoace_pipeline.pipeline import analyze_full

    db.set_file_status(file_id, "processing")
    t0 = time.perf_counter()
    try:
        r = analyze_full(path)
        wall = time.perf_counter() - t0
        detail = {
            "features": r["features"].model_dump(),
            "local": vars(r["local"]),
            "llm": vars(r["llm"]),
            "overlap": r["overlap"],
            "ser": {k: v for k, v in (r["ser"] or {}).items() if k != "segments"},
            "trace": vars(r["trace"]),
        }
        db.set_file_status(
            file_id, "done",
            result_json=json.dumps(r["result"].model_dump()),
            detail_json=json.dumps(detail),
            duration_s=r["features"].duration_s,
            wall_s=round(wall, 2),
            cost_usd=r["llm"].cost_usd,
        )
    except IngestError as e:
        db.set_file_status(file_id, "failed", error=f"unreadable audio: {e}",
                           wall_s=round(time.perf_counter() - t0, 2))
    except Exception as e:  # noqa: BLE001 - any per-file crash must not kill the batch
        traceback.print_exc()
        db.set_file_status(file_id, "failed", error=str(e)[:300],
                           wall_s=round(time.perf_counter() - t0, 2))
    finally:
        db.finish_batch_if_done(batch_id)


def _loop() -> None:
    while True:
        batch_id, file_id, path = _q.get()
        try:
            _process(batch_id, file_id, path)
        finally:
            _q.task_done()


def _prewarm() -> None:
    """Load every model once, sequentially, before accepting work.
    Concurrent first-loads of the torch stacks crash natively on some
    platforms, and pre-warming moves the ~60 s load cost off the first file."""
    def _rss_mb() -> int:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        except Exception:  # noqa: BLE001 - windows dev box
            return 0

    # sensevoice first: it has the largest load spike, so it runs while the
    # memory baseline is smallest
    loaders = []
    if os.environ.get("DISABLE_SER") != "1":
        loaders.append(("sensevoice",
                        lambda: __import__("autoace_pipeline.ser", fromlist=["_model"])._model()))
    loaders += [
        ("silero", lambda: __import__("autoace_pipeline.vad", fromlist=["_model"])._model()),
        ("pyannote", lambda: __import__("autoace_pipeline.overlap", fromlist=["_pipeline"])._pipeline()),
    ]
    print(f"[prewarm] start (peak rss {_rss_mb()} MB)", flush=True)
    for name, fn in loaders:
        try:
            fn()
            print(f"[prewarm] {name} ready (peak rss {_rss_mb()} MB)", flush=True)
        except Exception as e:  # noqa: BLE001 - degraded but alive is better than down
            print(f"[prewarm] {name} unavailable: {str(e)[:120]}", flush=True)


def _recover_orphans() -> None:
    """Files left queued/processing by a previous run are re-queued so a
    restart resumes the batch instead of stranding it."""
    rows = db.con().execute(
        "SELECT id, batch_id, path FROM files WHERE status IN ('queued', 'processing')"
    ).fetchall()
    for r in rows:
        db.set_file_status(r["id"], "queued")
        submit(r["batch_id"], r["id"], r["path"])
    if rows:
        print(f"[recover] re-queued {len(rows)} orphaned file(s)")


def start(workers: int = 1) -> None:
    global _started
    if _started:
        return
    _started = True

    def _boot() -> None:
        _prewarm()
        _recover_orphans()
        for i in range(workers):
            threading.Thread(target=_loop, daemon=True, name=f"worker-{i}").start()

    threading.Thread(target=_boot, daemon=True, name="worker-boot").start()
