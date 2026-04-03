# silicon-bench

**How fast is YOUR Mac at AI inference?**

Unified benchmark suite for AI inference on Apple Silicon. Tests CoreML, MLX, PyTorch MPS, and Ollama. Auto-detects your chip and available backends — skip what you haven't installed, run what you have.

Supports M1 / M2 / M3 / M4 / M5. No configuration needed.

---

## Sample Results — M5 Pro (ARIS4U, 48 GB)

| Benchmark | Backend | Tokens/s | Memory MB | Latency ms | Model |
|---|---|---|---|---|---|
| coreml-embedding | coreml | 652.0 | 180.0 | 1.5 | BAAI/bge-small-en-v1.5 |
| mlx-inference | mlx | 114.0 | 420.0 | 85.0 | mlx-community/bge-small-en-v1.5-mlx |
| ollama-inference | ollama | 68.3 | 240.0 | 310.0 | llama3.2 |
| embedding-st | sentence_transformers | 89.0 | 195.0 | 11.0 | BAAI/bge-small-en-v1.5 |

> CoreML hits 652 embeddings/sec by routing through the Neural Engine (ANE). MLX is fast for
> generative inference. Ollama is the easiest to get started — just `ollama pull llama3.2`.

---

## Quick Start

```bash
# Install (base — no backend dependencies)
pip install silicon-bench

# Run all available benchmarks
silicon-bench

# Hardware info only
silicon-bench --hardware-only
```

## Install with backends

```bash
# Ollama inference (requires Ollama running: https://ollama.com)
pip install "silicon-bench[ollama]"
ollama pull llama3.2

# MLX (Apple Silicon only)
pip install "silicon-bench[mlx]"

# CoreML / sentence-transformers embeddings
pip install "silicon-bench[coreml]"

# Everything
pip install "silicon-bench[all]"
```

---

## CLI Reference

```
silicon-bench                            # run all available benchmarks
silicon-bench --backend ollama           # only Ollama
silicon-bench --backend ollama --model llama3.2
silicon-bench --backend mlx --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
silicon-bench --backend coreml
silicon-bench --embedding-only           # skip generative inference
silicon-bench --hardware-only            # chip + core + memory info
silicon-bench --json                     # machine-readable JSON output
```

### Full options

| Flag | Default | Description |
|---|---|---|
| `--backend` | all | `ollama`, `mlx`, `coreml`, or `embedding` |
| `--model` | — | Override model for selected backend |
| `--ollama-model` | `llama3.2` | Ollama model to benchmark |
| `--mlx-model` | `mlx-community/bge-small-en-v1.5-mlx` | MLX model |
| `--coreml-model` | `bge-small` | CoreML embedding model |
| `--embedding-only` | false | Skip generative benchmarks |
| `--hardware-only` | false | Print hardware info and exit |
| `--json` | false | Output results as JSON |

---

## What it measures

| Metric | Description |
|---|---|
| **tokens/sec** | Inference throughput (generated tokens per second) |
| **embeddings/sec** | Embedding throughput (vectors per second) |
| **latency ms** | First-token latency (time to first output) |
| **memory MB** | Peak RSS memory delta during benchmark |

---

## Backends

| Backend | What it tests | Requirement |
|---|---|---|
| `ollama` | LLM inference via local Ollama server | `ollama` running + model pulled |
| `mlx` | MLX generative inference on Apple Silicon | `pip install mlx mlx-lm` |
| `coreml` | Embedding via sentence-transformers (ANE/GPU) | `pip install coremltools sentence-transformers` |
| `embedding` | Auto-selects: Ollama embed or sentence-transformers | one of the above |

Missing backends are skipped with a note — not an error.

---

## Development

```bash
git clone https://github.com/amado-alvarez/silicon-bench
cd silicon-bench
pip install -e ".[dev]"
python -m pytest test_bench.py -v
```

---

## Share Your Results

Run `silicon-bench --json` and post your numbers in
[GitHub Discussions](https://github.com/amado-alvarez/silicon-bench/discussions).

Include: chip name, memory, and which backends you tested.

---

## License

MIT — Created by [Amado Alvarez Sueiras](https://github.com/amado-alvarez)
