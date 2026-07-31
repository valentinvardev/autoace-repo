"""Concurrent access to the shared model singletons must not corrupt state.

WORKERS>1 crashed the TorchScript interpreter in production: Silero VAD
carries RNN state across calls and was being driven from two threads at
once. These tests hammer the shared models from several threads and require
identical results to a single-threaded run.

Skipped automatically when the models are unavailable (no torch, no network,
no HF token), so the fast unit suite still runs anywhere.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

SR = 16000


def _signal(seed: int) -> np.ndarray:
    """Speech-like bursts separated by silence; distinct per seed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(8 * SR)) / SR
    tone = np.sin(2 * np.pi * (180 + 40 * seed) * t)
    env = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(np.float32)
    return (0.2 * tone * env + rng.normal(0, 0.001, len(t))).astype(np.float32)


def test_vad_concurrent_matches_sequential():
    vad = pytest.importorskip("autoace_pipeline.vad")
    try:
        vad._model()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"silero unavailable: {e}")

    signals = [_signal(i) for i in range(4)]
    expected = [vad.speech_spans(s, len(s) / SR) for s in signals]

    with ThreadPoolExecutor(max_workers=4) as ex:
        got = list(ex.map(lambda s: vad.speech_spans(s, len(s) / SR), signals))

    assert got == expected, "concurrent VAD results diverged from sequential"


def test_vad_repeated_concurrent_rounds():
    """State corruption is intermittent; several rounds make it likely."""
    vad = pytest.importorskip("autoace_pipeline.vad")
    try:
        vad._model()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"silero unavailable: {e}")

    sig = _signal(1)
    expected = vad.speech_spans(sig, len(sig) / SR)
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=3) as ex:
            for got in ex.map(lambda _: vad.speech_spans(sig, len(sig) / SR), range(6)):
                assert got == expected
