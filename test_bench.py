"""Tests for silicon-bench.

Covers hardware detection, result formatting, and graceful backend skipping.
Run with: python -m pytest tests/ or python -m pytest test_bench.py
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from bench import (
    BenchmarkResult,
    bench_coreml,
    bench_embedding,
    bench_mlx,
    bench_ollama,
    detect_hardware,
    print_results,
    run_suite,
)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

class TestDetectHardware:
    """Tests for detect_hardware()."""

    def test_returns_dict(self) -> None:
        hw = detect_hardware()
        assert isinstance(hw, dict)

    def test_required_keys(self) -> None:
        hw = detect_hardware()
        required = {"chip", "cpu_cores", "gpu_cores", "ane", "memory_gb", "platform"}
        assert required.issubset(hw.keys())

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_macos_chip_not_empty(self) -> None:
        hw = detect_hardware()
        assert hw["chip"] != ""

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_macos_cpu_cores_positive(self) -> None:
        hw = detect_hardware()
        assert hw["cpu_cores"] > 0

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_macos_platform_string(self) -> None:
        hw = detect_hardware()
        assert "Darwin" in hw["platform"] or "macOS" in hw["platform"]

    def test_ane_is_bool(self) -> None:
        hw = detect_hardware()
        assert isinstance(hw["ane"], bool)

    def test_memory_gb_non_negative(self) -> None:
        hw = detect_hardware()
        assert hw["memory_gb"] >= 0


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation_minimal(self) -> None:
        r = BenchmarkResult(name="test", backend="test")
        assert r.name == "test"
        assert r.backend == "test"
        assert r.tokens_per_sec == 0.0
        assert r.error is None

    def test_creation_full(self) -> None:
        r = BenchmarkResult(
            name="ollama-inference",
            backend="ollama",
            tokens_per_sec=42.5,
            memory_mb=256.0,
            latency_ms=120.0,
            throughput=42.5,
            model="llama3.2",
            error=None,
            extra={"elapsed_s": 1.5},
        )
        assert r.tokens_per_sec == 42.5
        assert r.model == "llama3.2"
        assert r.extra["elapsed_s"] == 1.5

    def test_error_result(self) -> None:
        r = BenchmarkResult(
            name="ollama-inference",
            backend="ollama",
            error="ollama package not installed",
        )
        assert r.error is not None
        assert "not installed" in r.error

    def test_extra_defaults_to_empty_dict(self) -> None:
        r = BenchmarkResult(name="x", backend="y")
        assert r.extra == {}
        # Verify it's not shared across instances
        r2 = BenchmarkResult(name="a", backend="b")
        r.extra["key"] = "val"
        assert "key" not in r2.extra


# ---------------------------------------------------------------------------
# Graceful backend skipping (unavailable backends)
# ---------------------------------------------------------------------------

class TestGracefulSkip:
    """Benchmarks return a result with error set when backend is unavailable."""

    def test_ollama_skips_when_not_installed(self) -> None:
        with patch.dict(sys.modules, {"ollama": None}):
            result = bench_ollama("llama3.2")
        assert result.error is not None
        assert result.backend == "ollama"
        assert result.tokens_per_sec == 0.0

    def test_mlx_skips_when_not_installed(self) -> None:
        with patch.dict(sys.modules, {"mlx.core": None, "mlx_lm": None}):
            result = bench_mlx("some-model")
        assert result.error is not None
        assert result.backend == "mlx"

    def test_coreml_skips_when_not_installed(self) -> None:
        with patch.dict(sys.modules, {"coremltools": None}):
            result = bench_coreml("bge-small")
        assert result.error is not None
        assert result.backend == "coreml"

    def test_embedding_returns_result_on_failure(self) -> None:
        # Force all backends to fail by pointing to nonexistent backend
        result = bench_embedding(backend="nonexistent_backend_xyz")
        assert result.error is not None

    def test_run_suite_returns_list(self) -> None:
        # run_suite should always return a list, even if everything fails
        results = run_suite()
        assert isinstance(results, list)

    def test_run_suite_all_results_are_benchmark_results(self) -> None:
        results = run_suite()
        for r in results:
            assert isinstance(r, BenchmarkResult)


# ---------------------------------------------------------------------------
# print_results formatting
# ---------------------------------------------------------------------------

class TestPrintResults:
    """Tests for print_results() output formatting."""

    def test_prints_table_headers(self, capsys: pytest.CaptureFixture) -> None:
        results = [
            BenchmarkResult(
                name="ollama-inference",
                backend="ollama",
                tokens_per_sec=55.3,
                memory_mb=512.0,
                latency_ms=230.0,
                model="llama3.2",
            )
        ]
        print_results(results)
        captured = capsys.readouterr()
        assert "Benchmark" in captured.out
        assert "Backend" in captured.out
        assert "Tokens/s" in captured.out

    def test_prints_error_as_skip(self, capsys: pytest.CaptureFixture) -> None:
        results = [
            BenchmarkResult(
                name="mlx-inference",
                backend="mlx",
                error="mlx not installed",
            )
        ]
        print_results(results)
        captured = capsys.readouterr()
        assert "SKIP" in captured.out
        assert "mlx not installed" in captured.out

    def test_empty_results(self, capsys: pytest.CaptureFixture) -> None:
        print_results([])
        captured = capsys.readouterr()
        assert "No results" in captured.out

    def test_prints_model_name(self, capsys: pytest.CaptureFixture) -> None:
        results = [
            BenchmarkResult(
                name="coreml-embedding",
                backend="coreml",
                throughput=652.0,
                model="BAAI/bge-small-en-v1.5",
                tokens_per_sec=652.0,
            )
        ]
        print_results(results)
        captured = capsys.readouterr()
        assert "bge-small" in captured.out

    def test_multiple_results(self, capsys: pytest.CaptureFixture) -> None:
        results = [
            BenchmarkResult(name="bench-a", backend="x", tokens_per_sec=10.0),
            BenchmarkResult(name="bench-b", backend="y", error="unavailable"),
            BenchmarkResult(name="bench-c", backend="z", tokens_per_sec=200.0),
        ]
        print_results(results)
        captured = capsys.readouterr()
        assert "bench-a" in captured.out
        assert "bench-b" in captured.out
        assert "bench-c" in captured.out
        assert "SKIP" in captured.out


# ---------------------------------------------------------------------------
# run_suite filtering
# ---------------------------------------------------------------------------

class TestRunSuite:
    """Tests for run_suite() backend filtering."""

    def test_backend_filter_single(self) -> None:
        results = run_suite(backends=["coreml"])
        # Should only contain coreml results
        backends = {r.backend for r in results}
        # All results should be coreml (or sentence_transformers as proxy)
        assert "ollama" not in backends
        assert "mlx" not in backends

    def test_embedding_only_excludes_inference(self) -> None:
        results = run_suite(embedding_only=True)
        inference_names = {"ollama-inference", "mlx-inference"}
        result_names = {r.name for r in results}
        assert not inference_names.intersection(result_names)
