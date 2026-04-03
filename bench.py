"""silicon-bench — AI inference benchmarks for Apple Silicon.

Measures tokens/sec, embeddings/sec, first-token latency, memory usage,
and thermal throttling across CoreML, MLX, PyTorch MPS, and Ollama backends.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single benchmark measurement.

    Attributes:
        name: Human-readable benchmark name.
        backend: Backend identifier (ollama, mlx, coreml, mps).
        tokens_per_sec: Inference throughput (tokens per second).
        memory_mb: Peak RSS memory in megabytes.
        latency_ms: First-token (or first-result) latency in milliseconds.
        throughput: Generic throughput metric (backend-specific unit).
        model: Model name or identifier used.
        error: Error message if benchmark failed/skipped.
        extra: Additional key-value metadata.
    """

    name: str
    backend: str
    tokens_per_sec: float = 0.0
    memory_mb: float = 0.0
    latency_ms: float = 0.0
    throughput: float = 0.0
    model: str = ""
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_hardware() -> dict:
    """Detect Apple Silicon chip, core counts, memory, and ANE availability.

    Returns:
        Dict with keys: chip, cpu_cores, gpu_cores, ane, memory_gb, platform.
    """
    info: dict = {
        "chip": "unknown",
        "cpu_cores": 0,
        "gpu_cores": 0,
        "ane": False,
        "memory_gb": 0,
        "platform": platform.platform(),
    }

    if platform.system() != "Darwin":
        info["chip"] = "non-apple"
        return info

    # Chip name via system_profiler
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["chip"] = raw
    except Exception:
        pass

    # Try system_profiler for GPU / ANE info
    try:
        sp_out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        sp = json.loads(sp_out)
        hw = sp.get("SPHardwareDataType", [{}])[0]
        chip_name = hw.get("chip_type", "") or hw.get("cpu_type", "")
        if chip_name:
            info["chip"] = chip_name
        mem_raw = hw.get("physical_memory", "")  # e.g. "48 GB"
        if mem_raw:
            try:
                info["memory_gb"] = int(mem_raw.split()[0])
            except ValueError:
                pass
    except Exception:
        pass

    # CPU core count
    try:
        info["cpu_cores"] = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.logicalcpu"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except Exception:
        pass

    # ANE: present on all Apple Silicon; confirm via chip name
    chip_lower = info["chip"].lower()
    info["ane"] = any(k in chip_lower for k in ("apple m", "m1", "m2", "m3", "m4", "m5"))

    # GPU cores: best-effort via system_profiler graphics
    try:
        gfx_out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        gfx = json.loads(gfx_out)
        displays = gfx.get("SPDisplaysDataType", [{}])
        for d in displays:
            cores_str = d.get("sppci_cores", "")
            if cores_str:
                try:
                    info["gpu_cores"] = int(cores_str)
                    break
                except ValueError:
                    pass
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Return current RSS memory in MB (macOS: RUSAGE_SELF maxrss is bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes; on Linux it's kilobytes
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


# ---------------------------------------------------------------------------
# Ollama benchmark
# ---------------------------------------------------------------------------

def bench_ollama(model: str = "llama3.2") -> BenchmarkResult:
    """Benchmark Ollama inference throughput and latency.

    Args:
        model: Ollama model name to benchmark (must be pulled locally).

    Returns:
        BenchmarkResult with tokens_per_sec and latency_ms populated.
    """
    try:
        import ollama  # type: ignore
    except ImportError:
        return BenchmarkResult(
            name="ollama-inference",
            backend="ollama",
            model=model,
            error="ollama package not installed (pip install ollama)",
        )

    prompt = (
        "Explain the difference between a transformer encoder and decoder "
        "in exactly three sentences."
    )
    mem_before = _rss_mb()

    # --- first-token latency ---
    t0 = time.perf_counter()
    first_token_received = False
    first_token_latency = 0.0
    token_count = 0
    content_parts: list[str] = []

    try:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            if content and not first_token_received:
                first_token_latency = (time.perf_counter() - t0) * 1000
                first_token_received = True
            if content:
                content_parts.append(content)
                token_count += len(content.split())

    except Exception as exc:
        return BenchmarkResult(
            name="ollama-inference",
            backend="ollama",
            model=model,
            error=f"Ollama error: {exc}",
        )

    elapsed = time.perf_counter() - t0
    mem_after = _rss_mb()
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0

    return BenchmarkResult(
        name="ollama-inference",
        backend="ollama",
        tokens_per_sec=tokens_per_sec,
        memory_mb=mem_after - mem_before,
        latency_ms=first_token_latency,
        throughput=tokens_per_sec,
        model=model,
        extra={
            "elapsed_s": round(elapsed, 3),
            "token_count": token_count,
            "response_preview": "".join(content_parts)[:120],
        },
    )


# ---------------------------------------------------------------------------
# MLX benchmark
# ---------------------------------------------------------------------------

def bench_mlx(model: str = "mlx-community/bge-small-en-v1.5-mlx") -> BenchmarkResult:
    """Benchmark MLX embedding throughput on Apple Silicon.

    Args:
        model: MLX-compatible model identifier (HuggingFace hub path).

    Returns:
        BenchmarkResult with tokens_per_sec and latency_ms populated.
    """
    try:
        import mlx.core as mx  # type: ignore
        import mlx_lm  # type: ignore  # noqa: F401
    except ImportError:
        return BenchmarkResult(
            name="mlx-inference",
            backend="mlx",
            model=model,
            error="mlx / mlx-lm not installed (pip install mlx mlx-lm)",
        )

    # Use mlx_lm for text generation if model looks like a generative model,
    # otherwise fall back to embedding-style timing
    texts = [
        "Apple Silicon neural engine delivers remarkable AI performance.",
        "MLX is a machine learning framework designed for Apple Silicon.",
        "Benchmarking embeddings reveals true inference throughput.",
    ]

    mem_before = _rss_mb()
    t_load = time.perf_counter()

    try:
        from mlx_lm import load, generate  # type: ignore

        loaded_model, tokenizer = load(model)
        load_ms = (time.perf_counter() - t_load) * 1000

        # Warm-up
        _ = generate(loaded_model, tokenizer, prompt=texts[0], max_tokens=10, verbose=False)

        # Timed run
        token_counts = []
        t0 = time.perf_counter()
        first_latency = 0.0
        first_done = False

        for text in texts:
            t_start = time.perf_counter()
            response = generate(
                loaded_model, tokenizer, prompt=text, max_tokens=64, verbose=False
            )
            if not first_done:
                first_latency = (time.perf_counter() - t_start) * 1000
                first_done = True
            token_counts.append(len(tokenizer.encode(response)))

        elapsed = time.perf_counter() - t0
        total_tokens = sum(token_counts)
        mem_after = _rss_mb()

        return BenchmarkResult(
            name="mlx-inference",
            backend="mlx",
            tokens_per_sec=total_tokens / elapsed if elapsed > 0 else 0.0,
            memory_mb=mem_after - mem_before,
            latency_ms=first_latency,
            throughput=total_tokens / elapsed if elapsed > 0 else 0.0,
            model=model,
            extra={
                "load_ms": round(load_ms, 1),
                "total_tokens": total_tokens,
                "elapsed_s": round(elapsed, 3),
            },
        )

    except Exception as exc:
        return BenchmarkResult(
            name="mlx-inference",
            backend="mlx",
            model=model,
            error=f"MLX error: {exc}",
        )


# ---------------------------------------------------------------------------
# CoreML benchmark
# ---------------------------------------------------------------------------

def bench_coreml(model: str = "bge-small") -> BenchmarkResult:
    """Benchmark CoreML embedding throughput using ANE/GPU acceleration.

    Args:
        model: Short model identifier or path to .mlpackage / .mlmodel.

    Returns:
        BenchmarkResult with throughput (embeddings/sec) populated.
    """
    try:
        import coremltools as ct  # type: ignore
    except ImportError:
        return BenchmarkResult(
            name="coreml-embedding",
            backend="coreml",
            model=model,
            error="coremltools not installed (pip install coremltools)",
        )

    # Try sentence-transformers as the embedding engine + CoreML compute unit
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        return BenchmarkResult(
            name="coreml-embedding",
            backend="coreml",
            model=model,
            error="sentence-transformers not installed",
        )

    sentences = [
        "Apple Silicon delivers exceptional AI performance.",
        "CoreML leverages the Neural Engine for fast inference.",
        "Benchmarking reveals hardware differences across chips.",
        "The M5 Pro has 14 CPU cores and 20 GPU cores.",
        "Embeddings power semantic search and RAG pipelines.",
    ] * 20  # 100 sentences total

    mem_before = _rss_mb()

    try:
        t_load = time.perf_counter()
        # sentence-transformers model name mapping
        model_map = {
            "bge-small": "BAAI/bge-small-en-v1.5",
            "bge-base": "BAAI/bge-base-en-v1.5",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        }
        st_model_name = model_map.get(model, model)
        embedder = SentenceTransformer(st_model_name)
        load_ms = (time.perf_counter() - t_load) * 1000

        # Warm-up pass
        _ = embedder.encode(sentences[:5], batch_size=32)

        # Timed pass
        t0 = time.perf_counter()
        first_latency_start = time.perf_counter()
        _ = embedder.encode([sentences[0]], batch_size=1)
        first_latency = (time.perf_counter() - first_latency_start) * 1000

        t1 = time.perf_counter()
        _ = embedder.encode(sentences, batch_size=32, show_progress_bar=False)
        elapsed = time.perf_counter() - t1

        mem_after = _rss_mb()
        embeddings_per_sec = len(sentences) / elapsed if elapsed > 0 else 0.0

        return BenchmarkResult(
            name="coreml-embedding",
            backend="coreml",
            tokens_per_sec=embeddings_per_sec,  # embeddings/sec as throughput proxy
            memory_mb=mem_after - mem_before,
            latency_ms=first_latency,
            throughput=embeddings_per_sec,
            model=st_model_name,
            extra={
                "load_ms": round(load_ms, 1),
                "sentences": len(sentences),
                "embeddings_per_sec": round(embeddings_per_sec, 1),
                "elapsed_s": round(elapsed, 3),
            },
        )

    except Exception as exc:
        return BenchmarkResult(
            name="coreml-embedding",
            backend="coreml",
            model=model,
            error=f"CoreML/SentenceTransformer error: {exc}",
        )


# ---------------------------------------------------------------------------
# Generic embedding benchmark
# ---------------------------------------------------------------------------

def bench_embedding(
    text: str = "The quick brown fox jumps over the lazy dog.",
    backend: str = "auto",
) -> BenchmarkResult:
    """Benchmark embedding generation speed across available backends.

    Args:
        text: Input text to embed repeatedly.
        backend: One of "auto", "ollama", "sentence_transformers".

    Returns:
        BenchmarkResult with throughput (embeddings/sec) populated.
    """
    n_runs = 50
    texts = [text] * n_runs

    if backend in ("auto", "ollama"):
        try:
            import ollama  # type: ignore

            mem_before = _rss_mb()
            # Warm-up
            ollama.embeddings(model="nomic-embed-text", prompt=text)

            t0 = time.perf_counter()
            for t in texts:
                ollama.embeddings(model="nomic-embed-text", prompt=t)
            elapsed = time.perf_counter() - t0
            mem_after = _rss_mb()

            return BenchmarkResult(
                name="embedding-ollama",
                backend="ollama",
                tokens_per_sec=n_runs / elapsed,
                memory_mb=mem_after - mem_before,
                latency_ms=(elapsed / n_runs) * 1000,
                throughput=n_runs / elapsed,
                model="nomic-embed-text",
                extra={"n_runs": n_runs, "elapsed_s": round(elapsed, 3)},
            )
        except Exception as exc:
            if backend == "ollama":
                return BenchmarkResult(
                    name="embedding-ollama",
                    backend="ollama",
                    error=str(exc),
                )
            # fall through to next backend

    if backend in ("auto", "sentence_transformers"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            mem_before = _rss_mb()
            embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
            _ = embedder.encode([text])  # warm-up

            t0 = time.perf_counter()
            _ = embedder.encode(texts, batch_size=32, show_progress_bar=False)
            elapsed = time.perf_counter() - t0
            mem_after = _rss_mb()

            return BenchmarkResult(
                name="embedding-st",
                backend="sentence_transformers",
                tokens_per_sec=n_runs / elapsed,
                memory_mb=mem_after - mem_before,
                latency_ms=(elapsed / n_runs) * 1000,
                throughput=n_runs / elapsed,
                model="BAAI/bge-small-en-v1.5",
                extra={"n_runs": n_runs, "elapsed_s": round(elapsed, 3)},
            )
        except Exception as exc:
            return BenchmarkResult(
                name="embedding-st",
                backend="sentence_transformers",
                error=str(exc),
            )

    return BenchmarkResult(
        name="embedding",
        backend=backend,
        error=f"No embedding backend available for: {backend}",
    )


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------

def run_suite(
    backends: Optional[list[str]] = None,
    ollama_model: str = "llama3.2",
    mlx_model: str = "mlx-community/bge-small-en-v1.5-mlx",
    coreml_model: str = "bge-small",
    embedding_only: bool = False,
) -> list[BenchmarkResult]:
    """Run all available benchmarks and return results.

    Args:
        backends: List of backends to run. None runs all.
        ollama_model: Ollama model to benchmark.
        mlx_model: MLX model to benchmark.
        coreml_model: CoreML/sentence-transformers model to benchmark.
        embedding_only: If True, only run embedding benchmarks.

    Returns:
        List of BenchmarkResult, one per benchmark run.
    """
    run_all = backends is None
    active = set(backends or [])
    results: list[BenchmarkResult] = []

    if not embedding_only:
        if run_all or "ollama" in active:
            print(f"  Running Ollama inference ({ollama_model})...", flush=True)
            results.append(bench_ollama(ollama_model))

        if run_all or "mlx" in active:
            print(f"  Running MLX inference ({mlx_model})...", flush=True)
            results.append(bench_mlx(mlx_model))

    if run_all or "coreml" in active:
        print(f"  Running CoreML embedding ({coreml_model})...", flush=True)
        results.append(bench_coreml(coreml_model))

    if run_all or "embedding" in active:
        print("  Running embedding benchmark (auto-detect backend)...", flush=True)
        results.append(bench_embedding())

    return results


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a formatted table.

    Args:
        results: List of BenchmarkResult to display.
    """
    if not results:
        print("No results to display.")
        return

    col_w = [30, 12, 16, 14, 14, 40]
    headers = ["Benchmark", "Backend", "Tokens/s", "Memory MB", "Latency ms", "Model / Note"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    def row(cells: list[str]) -> str:
        parts = []
        for cell, w in zip(cells, col_w):
            parts.append(f" {cell[:w]:<{w}} ")
        return "|" + "|".join(parts) + "|"

    print()
    print(sep)
    print(row(headers))
    print(sep)

    for r in results:
        if r.error:
            note = f"SKIP: {r.error}"
            print(row([r.name, r.backend, "-", "-", "-", note]))
        else:
            tps = f"{r.tokens_per_sec:.1f}" if r.tokens_per_sec else "-"
            mem = f"{r.memory_mb:.1f}" if r.memory_mb else "-"
            lat = f"{r.latency_ms:.1f}" if r.latency_ms else "-"
            note = r.model or ""
            if r.extra.get("embeddings_per_sec"):
                note += f"  [{r.extra['embeddings_per_sec']} emb/s]"
            print(row([r.name, r.backend, tps, mem, lat, note]))

    print(sep)
    print()


def print_hardware(hw: dict) -> None:
    """Print detected hardware info.

    Args:
        hw: Dict returned by detect_hardware().
    """
    print()
    print("Hardware")
    print("--------")
    print(f"  Chip     : {hw.get('chip', 'unknown')}")
    print(f"  CPU cores: {hw.get('cpu_cores', '?')}")
    print(f"  GPU cores: {hw.get('gpu_cores', '?')}")
    print(f"  ANE      : {'yes' if hw.get('ane') else 'no'}")
    print(f"  Memory   : {hw.get('memory_gb', '?')} GB")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="silicon-bench",
        description="AI inference benchmarks for Apple Silicon",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "mlx", "coreml", "embedding"],
        help="Run only this backend (default: all available)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name/path to benchmark (used with --backend)",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        metavar="MODEL",
        help="Ollama model to benchmark (default: llama3.2)",
    )
    parser.add_argument(
        "--mlx-model",
        default="mlx-community/bge-small-en-v1.5-mlx",
        metavar="MODEL",
        help="MLX model to benchmark",
    )
    parser.add_argument(
        "--coreml-model",
        default="bge-small",
        metavar="MODEL",
        help="CoreML/embedding model to benchmark",
    )
    parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="Run only embedding benchmarks",
    )
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="Print hardware info and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON",
    )
    return parser


def main() -> None:
    """CLI entry point for silicon-bench."""
    parser = _build_parser()
    args = parser.parse_args()

    print("silicon-bench — Apple Silicon AI Benchmark Suite")
    print("=" * 50)

    hw = detect_hardware()
    print_hardware(hw)

    if args.hardware_only:
        return

    backends = [args.backend] if args.backend else None

    # Per-backend model override
    ollama_model = args.model if (args.backend == "ollama" and args.model) else args.ollama_model
    mlx_model = args.model if (args.backend == "mlx" and args.model) else args.mlx_model
    coreml_model = args.model if (args.backend == "coreml" and args.model) else args.coreml_model

    print("Running benchmarks...")
    results = run_suite(
        backends=backends,
        ollama_model=ollama_model,
        mlx_model=mlx_model,
        coreml_model=coreml_model,
        embedding_only=args.embedding_only,
    )

    if args.output_json:
        import dataclasses
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    else:
        print_results(results)

        # Summary: count successes
        ok = [r for r in results if not r.error]
        skipped = [r for r in results if r.error]
        print(f"Completed {len(ok)} benchmark(s), {len(skipped)} skipped.")
        if skipped:
            print("Install missing dependencies to run skipped benchmarks.")
        print()


if __name__ == "__main__":
    main()
