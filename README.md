# Local SLM App

Calling a frontier model's API is the easy path — but plenty of real deployments
can't take it. Regulated data that legally can't leave a device. Latency budgets
that a network round-trip blows through. Per-call API costs that don't pencil
out at scale. Edge or offline environments with no guaranteed connectivity. Most
engineers have never had to navigate these constraints hands-on, because "just
call the API" almost always works in a side project. This repo is a deliberate
exercise in the alternative: running a small (3-4B parameter) open model
entirely on local hardware via [Ollama](https://ollama.com), benchmarking what
that actually costs in speed, making its output reliable enough to build on
(structured JSON, validated, with a real retry path), and then comparing three
candidate models head-to-head the way an engineering team would before picking
one for production.

## Phase 3: model comparison report

Live results, same 45-prompt set, run sequentially on the hardware described
below (Mistral 7B used 2 trials/prompt instead of 5 — each of its generations
takes 12-25s on this hardware, mostly on CPU rather than GPU, so 5 full trials
would have taken close to an hour just for that one model; n is shown per row):

| Model | Tokens/sec | TTFT (s) | Memory | Factual acc. | Summarization | JSON valid rate | JSON field acc. |
|---|---|---|---|---|---|---|---|
| llama3.2:3b | 19.1 ± 3.6 (n=225) | 0.29 ± 0.09 | 2.5 GB | 100.0% | 0.85 | 100.0% | 97.8% |
| phi4-mini:3.8b | 14.6 ± 2.5 (n=225) | 0.39 ± 0.33 | 3.1 GB | 93.3% | 0.79 | 100.0% | 95.6% |
| mistral:7b | 5.9 ± 2.1 (n=90) | 1.09 ± 1.44 | 5.1 GB | 100.0% | 0.91 | 100.0% | 95.6% |

**Takeaway:** all three models are perfectly reliable at schema-constrained
JSON extraction on this prompt set (100% valid across the board — the
`format` grammar constraint does its job regardless of model size). Where they
actually differ is speed and open-ended quality: Llama 3.2 3B is more than 3x
faster than Mistral 7B on this hardware (19.1 vs 5.9 tok/s) with essentially
tied factual accuracy, while Mistral edges out both smaller models on
summarization quality (0.91 vs 0.85/0.79) and Phi-4-mini is both the slowest
of the two small models and the only one to miss a factual answer. For a
latency-sensitive local deployment on hardware like this, Llama 3.2 3B is the
clear pick; Mistral 7B only earns its cost if summarization quality is the
priority and multi-second latency is acceptable. Full data:
`results/phase3_model_comparison.json` / `.md`.

Quality scoring is heuristic, not human- or LLM-judged: factual accuracy is
exact/substring/fuzzy match (RapidFuzz, threshold 85) against a short reference
answer; summarization score is `0.7 × keyword_coverage + 0.3 × length_score`
against an ideal 15-45 word band; JSON metrics come from Pydantic schema
validation plus per-field comparison against expected values. These are
documented proxies for quality, not a substitute for human or LLM-judge
evaluation — see `src/local_slm/scoring/` for exact logic.

## Hardware used for all numbers in this README

| | |
|---|---|
| Machine | MacBook Air (M1, 2020) |
| CPU/GPU | Apple M1, 8-core (4P+4E), integrated GPU via Metal |
| RAM | 8 GB unified memory |
| OS | macOS 26.5.2 |
| Ollama | 0.32.3 |

All results are hardware-dependent by construction — that's the point of this
project. Reproduce them on your own machine with the commands in
[Reproducing the benchmarks](#reproducing-the-benchmarks-locally).

8GB of unified memory is the binding constraint here: Ollama loads one model
at a time and swaps on demand, so memory pressure is per-model rather than
cumulative, but Mistral 7B (5.1GB loaded, per `ollama ps`, vs. 2.5-3.1GB for
the other two) leaves noticeably less headroom and is visibly slower — over
3x slower than Llama 3.2 3B in the Phase 3 comparison below. That's a real,
measured trade-off, not hidden in the numbers.

## Phase 2: structured output — JSON schema, validation, and retry

Small local models are far more prone to producing malformed or
schema-violating JSON than frontier hosted models, so treating this as a
solved problem is a common portfolio-project gap. This repo enforces structure
in two independent layers:

1. **Ollama's native `format` parameter** — a JSON Schema (generated from a
   Pydantic model via `model_json_schema()`) is passed directly to Ollama,
   which constrains decoding via grammar so the output is syntactically valid
   JSON in the common case.
2. **Independent Pydantic validation** (`structured/validator.py`) — grammar
   constraints guarantee syntax, not semantics. A quantized model can still
   put a hallucinated string into a field or mistype a number, so every
   response is separately parsed and validated against the same Pydantic
   model. This is intentional belt-and-suspenders, not redundant work.

If validation fails, `structured/retry.py` builds a single feedback reprompt —
original prompt + the invalid output + the exact Pydantic error message — and
tries exactly once more. If the second attempt also fails, the pipeline
returns a `StructuredFailure` object and logs a warning; it never raises an
unhandled exception and never loops beyond two attempts. See
`tests/unit/test_retry_logic.py` for the enforced call-count guarantees.

### Temperature 0 vs 0.7 (live run, Llama 3.2 3B, 45 prompts × 8 repeats)

| Metric | temp = 0.0 | temp = 0.7 |
|---|---|---|
| Mean response diversity (unique responses / repeats) | 0.12 | 0.50 |
| JSON schema validation failure rate | 0.00% | 0.00% |

Mean agreement rate of temp-0.7 responses vs. the temp-0 reference answer:
**49.7%** — i.e. raising temperature to 0.7 changes the model's answer roughly
half the time, even on short factual questions with one objectively correct
answer. Diversity is far higher on open-ended summarization prompts (where
temp-0.7 responses are essentially always unique) than on short factual
answers. Encouragingly, on this prompt set the *format*-constrained JSON
extraction task never failed schema validation at either temperature — the
`format` parameter's grammar constraint held even under higher-temperature
sampling; full per-prompt breakdown in
`results/phase2_temperature_experiment_llama3.2_3b.md`.

## Phase 1: inference benchmarks (raw numbers)

Llama 3.2 3B, 45 standardized prompts × 5 trials each (n=225), measured
end-to-end from the client side (see caveats below). A third of these prompts
(the json_extraction subset) are generated under Ollama's `format` JSON-schema
constraint, matching how they're actually used elsewhere in this project —
see the throughput note below.

| Metric | Mean ± stdev | Min | Max |
|---|---|---|---|
| Time to first token | 0.355s ± 0.282s | 0.210s | 3.207s |
| Tokens/sec | 17.85 ± 4.71 | 5.83 | 24.76 |
| Total latency | 2.466s ± 2.064s | 0.250s | 8.311s |
| Cold load (first request only) | 2.903s | — | — |

**Measurement caveats, stated plainly:**
- TTFT and latency are measured client-side and include local NDJSON-parsing
  overhead on top of true model latency, not just server-side generation time.
- "Time to first token" specifically means first *non-empty* streamed chunk —
  Ollama sometimes emits an initial empty-content chunk, and counting that
  would understate real latency.
- Ollama's per-generation timing fields (`eval_count`, `eval_duration`,
  `load_duration`, `prompt_eval_count`) only appear on the final streamed
  chunk (`done: true`); reading them from any other chunk is a real bug this
  code guards against with a dedicated test.
- Cold-load time is reported separately (from the very first generation only)
  rather than folded into the tokens/sec mean, since a one-time model load
  would otherwise skew steady-state throughput. It varies run to run based on
  whether Ollama had already evicted the model since the last request.
- **Grammar-constrained decoding has a real throughput cost**: applying the
  `format` JSON-schema constraint to the json_extraction third of this prompt
  mix pulls the blended mean down from ~22 tok/s (no constraint on any prompt)
  to ~17.85 tok/s. That's a genuine trade-off, not measurement noise — see
  `src/local_slm/benchmarking/harness.py` for where the constraint is applied
  and why (it must match the production code path so this benchmark and
  Phase 3's quality scoring, which reuses these same responses, stay honest).

Raw per-trial data: `results/raw/phase1_benchmarks_llama3.2_3b_raw.json`.

## Project structure

```
src/local_slm/
├── ollama_client.py      # streaming NDJSON client: TTFT, tokens/sec, timing
├── prompts.py             # loads data/prompts.yaml (45 standardized prompts)
├── schemas.py              # core Pydantic models (GenerationResult, etc.)
├── structured/             # JSON schema enforcement + validation + retry
├── experiments/temperature.py   # temp 0 vs 0.7 determinism experiment
├── benchmarking/           # n-trial harness + `ollama ps` memory parsing
├── scoring/                # per-task-type quality scoring (Phase 3)
├── comparison/runner.py    # orchestrates the 3-model comparison
├── reporting/              # pure functions: results -> Markdown
├── api/                    # FastAPI service layer (thin wrapper, same core logic as the CLI)
└── cli.py                  # Typer CLI: benchmark / extract / temp-experiment / compare
data/prompts.yaml           # the 45 standardized prompts (single source of truth)
results/                    # committed, real run outputs — the reference numbers above
tests/unit/                 # network-free unit tests, all Ollama calls mocked
```

The benchmarking harness and comparison runner call `OllamaClient` directly
rather than routing through the FastAPI layer — that keeps uvicorn/Starlette
scheduling and JSON re-serialization overhead out of the TTFT/tokens-per-sec
numbers this project's credibility rests on. The FastAPI app
(`api/app.py`) exists to satisfy the "expose it as a service" use case and
imports the exact same `OllamaClient` and `extract_with_retry` the CLI uses —
no forked logic between the two paths.

## Setup

```bash
brew install ollama          # or see https://ollama.com/download
ollama pull llama3.2:3b
ollama pull phi4-mini:3.8b   # only needed for the Phase 3 comparison
ollama pull mistral:7b       # only needed for the Phase 3 comparison

make install                 # creates .venv, installs the package + dev deps
```

## Usage

```bash
# Stream a single generation live, with a running tokens/sec readout
local-slm stream "Explain photosynthesis in one sentence." --model llama3.2:3b

# Run the FastAPI service
uvicorn local_slm.api.app:app --reload
```

## Reproducing the benchmarks locally

Each command below is a `make` target too (see `Makefile`); direct CLI calls
shown for clarity. All of them are I/O-bound network calls to your local
Ollama instance — no external API calls, no cost.

```bash
# Phase 1: inference benchmark for one model (TTFT, tokens/sec, latency)
local-slm benchmark --model llama3.2:3b

# Phase 2: JSON-schema-validate-retry pipeline over the 15 extraction prompts
local-slm extract --model llama3.2:3b

# Phase 2: temperature 0 vs 0.7 determinism experiment (45 prompts x 8 repeats x 2 temps)
local-slm temp-experiment --model llama3.2:3b

# Phase 3: three-model comparison (speed, memory, quality) -- takes a while.
# On slower/larger models, override trial count per model to keep runtime sane:
local-slm compare --trials-override mistral:7b=2
```

Each command writes JSON + Markdown into `results/`, overwriting the
per-model files this README's numbers were generated from. The versions
currently checked into `results/` are this repo's reference numbers, produced
on the exact hardware described above.

## Testing and CI

```bash
make test   # pytest tests/unit -- no Ollama, no GPU, no pulled models required
make lint   # ruff check + ruff format --check
```

Every Ollama interaction in the test suite is mocked (via `httpx.MockTransport`
or duck-typed fake clients), so the full test suite runs identically on a
CI runner with no GPU and no local models. GitHub Actions (`.github/workflows/ci.yml`)
runs lint + unit tests on every push; it deliberately does **not** run the
actual benchmark/comparison suite, since those numbers are hardware-dependent
by design — see [Reproducing the benchmarks](#reproducing-the-benchmarks-locally)
above instead.

## What's not done (documented, not hidden)

- **GGUF Q4/Q5 quantization comparison** (Aishwarya's "extra mile" stretch
  goal) is left as future work rather than attempted now. Ollama's default
  pulls are already Q4_0-quantized, so this project's numbers already reflect
  a quantized baseline; a follow-up would pull explicit `q5_0`/`q8_0` tags for
  one model and compare quality/speed against the Q4_0 default.
- **Demo recording**: a terminal recording of `local-slm stream` (live
  tokens/sec readout) belongs at `docs/demo.gif` — this is a manual capture
  step, not something automatable from a coding session.

## Attribution

Project scope and phase structure follow Aishwarya Srinivasan's "Local SLM
App with Ollama" project as described in "5 AI Engineer Projects to Build in
2026." All code, prompts, benchmarks, and analysis in this repo are original.
