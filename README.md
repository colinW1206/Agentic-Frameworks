# Investigating Reliable Code Generation with Agentic AI

A comparative study of **multi-agent LLM architectures for autonomous code generation**. Three
seminal agentic workflows — **AgentCoder**, **CodeCoR**, and **AgileCoder** — are re-implemented
from their papers into a single unified [CrewAI](https://github.com/crewAIInc/crewAI) harness and
benchmarked head-to-head against a zero-shot baseline on **HumanEval** and **EvalPlus (HumanEval+)**.

> Final Year Project — BSc (Hons) Computer Science, University College Dublin (2026).
> Supervisor: Prof. Alessio Ferrari.

---

## TL;DR — Key Findings

- **Multi-agent orchestration lifts a cheap model to frontier-level.** Powered by the low-cost
  **Qwen 3.5 Flash**, **CodeCoR reached 0.945 Pass@1 on HumanEval+ — effectively matching the
  zero-shot score of a state-of-the-art frontier model (Gemini 3 Pro, 0.951)**.
- **Every multi-agent system beat the zero-shot baseline** (0.866 → up to 0.945 on HumanEval+),
  confirming that iterative test-and-repair loops can substitute for raw parameter scale on
  algorithmic problems.
- **The "Enterprise Alignment Penalty" (a finding original to this project):** AgileCoder's SDLC
  emulation produces *defensive, production-style* code (explicit `TypeError`/`ValueError` guards)
  that **conflicts with algorithmic fuzz-testing** — so it suffered the steepest HumanEval→HumanEval+
  drop (−7.9%). Methodology must match the deployment environment.
- **Reliability is expensive — and the money saved is an illusion.** The accuracy gains cost up to a
  **~28× increase in token consumption**, ~40 hours of cumulative runtime, and real operational
  fragility (hallucinated imports trapping repair agents in infinite loops). Strikingly, running
  **all three multi-agent frameworks on the cheap Qwen model cost ~$16 in total — about the same as
  a single zero-shot run of the frontier Gemini 3 Pro ($13.50)**. You can *buy* intelligence upfront
  at a high token rate, or *simulate* it iteratively with a cheaper model and agents: the dollar cost
  lands in the same place, but the agentic route pays a massive latency penalty.

---

## Results

### Functional correctness (Pass@1)

All agentic systems and the standalone baseline use the **identical** Qwen 3.5 Flash backbone, so
performance deltas are attributable to the *orchestration methodology* alone. Gemini 3 Pro is shown
zero-shot as a frontier ceiling.

| System Architecture                  | HumanEval (base) | HumanEval+ (EvalPlus) | Drop (HE→HE+) |
| ------------------------------------ | :--------------: | :-------------------: | :-----------: |
| Frontier ceiling (Gemini 3 Pro, 0-shot) |    0.994      |       **0.951**       |    −4.3%      |
| **CodeCoR** (self-reflective)        |      0.976       |       **0.945**       |  **−3.1%**    |
| **AgentCoder** (test-driven)         |      0.970       |         0.927         |    −4.3%      |
| **AgileCoder** (SDLC emulation)      |      0.957       |         0.878         |    −7.9%      |
| Baseline (Qwen 3.5 Flash, 0-shot)    |      0.896       |         0.866         |    −3.0%      |

*Evaluated on the full 164-problem HumanEval population (no sampling) — Pass@1 reflects absolute
performance.*

### Resource cost (the trade-off)

| System Architecture       | Avg. total tokens / problem | Token cost multiplier |
| ------------------------- | :-------------------------: | :-------------------: |
| Baseline (standalone)     |           ~4,675            |         1.0×          |
| AgentCoder                |           ~32,300           |         6.9×          |
| AgileCoder                |          ~126,500           |        27.0×          |
| CodeCoR                   |          ~130,000           |        27.8×          |

Token usage was captured by injecting a global `litellm` hook deep into the orchestration layer,
since CrewAI hides usage inside nested async micro-tasks. The distributions are heavily
right-skewed: a few hard problems trigger deep, expensive repair loops.

**The headline trade-off:** despite the ~28× token blow-up, running *all three* frameworks on the
cheap Qwen 3.5 Flash model cost **~$16 in total — essentially the same as one zero-shot pass of the
frontier Gemini 3 Pro ($13.50)**. The agentic approach doesn't save money; it trades wall-clock time
(~40h vs. near-instant) for the ability to reach frontier-level accuracy on a budget model.

---

## The three architectures

Each was chosen to represent a *distinct, dominant* paradigm in agentic code generation, then
faithfully reconstructed as an event-driven CrewAI `Flow`.

| Framework      | Paradigm                  | Core mechanic (as implemented) |
| -------------- | ------------------------- | ------------------------------ |
| **AgentCoder** | Test-Driven Development   | A **double-blind sprint**: a Programmer and a Test Designer agent generate code and an adversarial `unittest` suite *in parallel*, blind to each other. A deterministic sandbox runs them; failures route back into a bounded refine loop. ([flow](agentcoder/src/agentcoder/flow.py)) |
| **CodeCoR**    | Self-Reflective Pruning   | **Generate-and-prune at every stage**: pools of CoT prompts, test suites, and code snippets are produced in parallel, scored by evaluator agents (1/0 arrays), and pruned. Surviving code runs in a sandbox; failures feed a dedicated Repair Agent + Repair Evaluator loop. ([flow](codecor/src/codecor/flow.py)) |
| **AgileCoder** | Enterprise SDLC Emulation | A **scrum pipeline**: Product Manager → Scrum Master → Developer → Senior Developer → Tester run sequentially over sprints, carrying execution feedback between sprints until acceptance criteria pass. ([flow](agilecoder/src/agilecoder/flow.py)) |

Agent roles and task prompts are externalised to per-framework `config/agents.yaml` and
`config/tasks.yaml`, so each paper's personas are declarative and swappable.

---

## Engineering highlights

Running hundreds of LLM calls across nested agent loops for ~40 hours surfaced real systems
problems. The harness ([`run.py`](run.py)) was built to survive them:

- **Global token interception** — monkey-patches `litellm.completion`/`acompletion` (sync + async,
  streaming-aware) to aggregate exact prompt/completion tokens that CrewAI otherwise hides.
- **Resumable, idempotent generation** — outputs stream incrementally to `.jsonl`; a pre-run check
  cross-references completed `task_id`s so an interrupted 40-hour run resumes without repeating
  expensive calls.
- **Network resilience** — 5-attempt retry loop with a 30s backoff to ride out OpenRouter rate
  limits and dropped connections.
- **Dataset-integrity failsafe** — a catastrophic per-task failure writes a blank fallback rather
  than crashing, guaranteeing exactly 164 records so EvalPlus grades every framework on equal terms.
- **State extraction over routing artifacts** — CrewAI Flows return the *last method's* output,
  often a routing string (`"finish"`, `"repair"`). A cascading extractor salvages the real code from
  the Flow's internal state (`ranked_code_set`, `best_code`, …).
- **Output sanitisation** — OS-level UTF-8 enforcement, strict ASCII coercion, and stripping of
  reasoning-model artifacts (`<think>` tags, hallucinated `__main__`/`sys.exit()` blocks) that would
  otherwise crash the EvalPlus grader.
- **Sandboxed, execution-based evaluation** — generated code + tests run in isolated subprocesses
  with strict timeouts to catch infinite loops; final grading runs in Docker
  (`ganler/evalplus:latest`) to satisfy EvalPlus's Linux-only `resource` limits.

---

## Prerequisites

* **Docker Desktop** (recommended — runs the `evalplus` suite safely and bypasses Windows-only
  limitations).
* **Python 3.10+** (for native runs).
* **[uv](https://github.com/astral-sh/uv)** — fast Python package manager.
* An **OpenRouter API key** (or any OpenAI-compatible endpoint).

## Setup

1. **Clone the repository.**
2. **Create a `.env`** in the project root (see [`.env.example`](.env.example)):
   ```text
   OPENROUTER_API_KEY=your_key_here
   OPENAI_API_BASE=https://openrouter.ai/api/v1
   OPENAI_MODEL_NAME=openrouter/qwen/qwen3.5-flash-02-23   # or any OpenAI-compatible model
   CREWAI_TRACING_ENABLED=false
   ```
3. **Install dependencies:**
   ```bash
   uv sync
   ```

## Usage

The pipeline runs in three phases: **Generation → Sanitisation → Evaluation**.

### 1. Generate code

Pick any framework: `baseline`, `agentcoder`, `codecor`, `agilecoder`.

```powershell
# Recommended on Windows — standardises the execution environment via Docker
docker run --rm --env-file .env -v "${PWD}:/workspace" -w /workspace `
  ghcr.io/astral-sh/uv:python3.12-bookworm uv run run.py codecor --mode full
```

```bash
# Or natively
uv run run.py codecor --mode full
```

Use `--mode test` to run only the first problem instead of the full 164.

### 2. Sanitise outputs

Reasoning models occasionally emit `<think>` tags or defensive `unittest` blocks that crash the
grader. Clean the generated samples before evaluation:

```bash
uv run python clean_sample.py
```

### 3. Evaluate against EvalPlus

```powershell
# Docker required on Windows — EvalPlus uses the Unix-only `resource` module
docker run --rm --env-file .env -v "${PWD}:/workspace" -w /workspace `
  ghcr.io/astral-sh/uv:python3.12-bookworm uv run run_evals.py
```

`run_evals.py` grades every framework's `samples.jsonl` against both HumanEval and HumanEval+.

## Project structure

```text
├── agentcoder/           # AgentCoder — test-driven double-blind sprint (CrewAI Flow + YAML agents)
├── agilecoder/           # AgileCoder — agile SDLC scrum pipeline
├── codecor/              # CodeCoR  — self-reflective generate-and-prune + repair
├── outputs/              # Per-framework samples.jsonl, usage_metrics.jsonl, eval results
├── run.py                # Orchestrator: generation loop, token hook, retries, state extraction
├── run_evals.py          # Batch EvalPlus grading across all frameworks
├── clean_sample.py       # Post-generation output sanitiser
└── pyproject.toml
```

---

## Limitations & future work

- **Scope is single-file Python algorithms.** Findings generalise to self-contained algorithmic
  synthesis, not multi-file repository generation. AgileCoder's penalty is partly an artifact of
  forcing a repo-scale methodology into single-function puzzles under CrewAI's rigid Flow structure.
- **Operational fragility.** Hallucinated third-party imports crashed the sandbox; repair agents
  sometimes mistook environmental faults for logic bugs and looped on already-correct code.
- **Next steps:** evaluate SDLC-emulation frameworks on repository-level benchmarks (e.g.
  SWE-bench) where their personas can pay off; optimise repair-loop token cost (traceback
  truncation, smaller fine-tuned evaluator models).

## Research questions answered

- **RQ1 — Effectiveness:** Multi-agent orchestration definitively improves functional correctness;
  all frameworks beat the 0.866 baseline, CodeCoR reaching 0.945.
- **RQ2 — Methodology impact:** Self-reflective routing (CodeCoR) was most robust; double-blind TDD
  (AgentCoder) effectively pre-fuzzed code; SDLC emulation (AgileCoder) backfired on algorithmic
  fuzzing — the Enterprise Alignment Penalty.
- **RQ3 — Efficiency:** Severe trade-offs — up to 28× tokens, ~40h runtime, and hallucination-driven
  instability.

## References

Reconstructed from: AgentCoder ([arXiv:2312.13010](https://arxiv.org/abs/2312.13010)),
CodeCoR ([arXiv:2501.07811](https://arxiv.org/abs/2501.07811)),
AgileCoder ([IEEE/ACM FORGE 2025](https://arxiv.org/abs/2406.11912)).
Benchmarks: HumanEval ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374)),
EvalPlus ([NeurIPS 2023](https://arxiv.org/abs/2305.01210)).
