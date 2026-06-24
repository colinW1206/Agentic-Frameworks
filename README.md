# Multi-Agentic Systems for Code Generation: A Comparative Analysis

This repository contains the implementation and evaluation framework for comparing different multi-agent Large Language Model (LLM) architectures for autonomous code generation. The project adapts three seminal agentic workflows—**AgentCoder**, **CodeCoR**, and **AgileCoder**—into a unified environment using the **CrewAI** framework.

The systems are evaluated against a zero-shot baseline using the **HumanEval** and **EvalPlus** (HumanEval+) benchmarks to measure algorithmic accuracy, test-driven self-correction capabilities, and edge-case robustness.

## Prerequisites

To run this project, you will need:
* **Docker Desktop** (Highly recommended for running the `evalplus` suite safely without OS-level dependency errors).
* **Python 3.10+** (If running natively).
* **uv** (Astral's ultra-fast Python package manager).
* An **OpenRouter API Key** (or equivalent OpenAI-compatible endpoint).

## Setup & Installation

1. **Clone the repository**

2. **Set up your environment variables:**
   Create a `.env` file in the root directory and add the following configurations:
   ```text
   OPENROUTER_API_KEY=your_key_here
    OPENAI_API_BASE=https://openrouter.ai/api/v1
    OPENAI_MODEL_NAME=openrouter/qwen/qwen3.5-flash-02-23 #Or any other model
    CREWAI_TRACING_ENABLED=false
   ```

3. **Install dependencies using `uv`:**
   ```bash
   uv sync
   ```

## Usage Instructions

The project is split into three distinct phases: **Generation**, **Sanitisation**, and **Evaluation**.

### Phase 1: Code Generation
You can generate code using any of the implemented frameworks (`baseline`, `agentcoder`, `codecor`, `agilecoder`).

**Running via Docker (Recommended for Windows users):**
This prevents cross-OS file pathing errors and standardises the execution environment.
```powershell
docker run --rm --env-file .env -v "${PWD}:/workspace" -w /workspace ghcr.io/astral-sh/uv:python3.12-bookworm uv run run.py codecor --mode full
```
*Note: Replace `codecor` with your desired framework. Use `--mode test` to run only the first problem instead of the full 164-problem benchmark.*

### Phase 2: Output Sanitisation
Because advanced reasoning models (like DeepSeek-R1) occasionally output internal thinking tags or append defensive `unittest` execution blocks, the outputs must be sanitised before evaluation to prevent benchmark crashes.

```bash
uv run python clean_samples.py
```
*This script will process the `outputs/` directories, strip `<think>` tags, remove `sys.exit()` calls, and ensure valid Python syntax.*

### Phase 3: Benchmark Evaluation (EvalPlus)
To grade the generated `samples.jsonl` files against the rigorous HumanEval and EvalPlus test suites, run the evaluation script.

**Running via Docker (Required for Windows):**
*EvalPlus utilises the Unix-only `resource` module to limit memory usage. Running via Docker bypasses this limitation.*
```powershell
docker run --rm --env-file .env -v "${PWD}:/workspace" -w /workspace ghcr.io/astral-sh/uv:python3.12-bookworm uv run run_evals.py
```

## Project Structure

```text
├── agentcoder/           # AgentCoder methodology implementation
├── agilecoder/           # AgileCoder methodology implementation
├── codecor/              # CodeCoR methodology implementation
├── outputs/              # Generated samples.jsonl, usage_metrics.jsonl and sample_eval_results.json
├── clean_samples.py      # Post-generation sanitisation script
├── run.py                # Main orchestrator for code generation
├── run_evals.py          # Automated evaluation loop for all frameworks
├── .env                  # API keys and environment variables
└── README.md             
```
