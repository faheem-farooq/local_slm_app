# Local SLM App

Runs small (3-7B) open models locally via [Ollama](https://ollama.com), with
benchmarking, schema-enforced JSON output, and a multi-model comparison.

## How it works

1. **Benchmarking** — `OllamaClient` streams generations from Ollama and
   measures time-to-first-token, tokens/sec, and total latency over N trials.
2. **Structured output** — JSON responses are constrained with Ollama's
   `format` (JSON Schema from a Pydantic model), then independently validated
   with Pydantic. If validation fails, one reprompt is sent with the error
   message attached; if that also fails, it returns a `StructuredFailure`
   instead of raising or retrying forever.
3. **Model comparison** — the same 45-prompt set (factual QA, summarization,
   JSON extraction) is run against multiple models, one at a time, scoring
   speed, memory, and per-task quality.

## Results (M1 MacBook Air, 8GB RAM, no discrete GPU)

| Model | Tokens/sec | TTFT (s) | Memory | Factual acc. | Summarization | JSON valid |
|---|---|---|---|---|---|---|
| llama3.2:3b | 19.1 ± 3.6 | 0.29 | 2.5 GB | 100% | 0.85 | 100% |
| phi4-mini:3.8b | 14.6 ± 2.5 | 0.39 | 3.1 GB | 93% | 0.79 | 100% |
| mistral:7b | 5.9 ± 2.1 | 1.09 | 5.1 GB | 100% | 0.91 | 100% |

Full breakdown in `results/`. Quality scores are heuristic (fuzzy/substring
match for factual QA, keyword coverage + length for summarization, schema
validation + field match for JSON) — see `src/local_slm/scoring/`.

**Temperature 0 vs 0.7** (Llama 3.2 3B, 45 prompts × 8 repeats): raising
temperature to 0.7 changes the answer ~50% of the time vs. the temp-0
response, and increases response diversity roughly 4x — but JSON schema
validation still passed 100% of the time at both temperatures.

## Setup

```bash
brew install ollama
ollama pull llama3.2:3b
ollama pull phi4-mini:3.8b   # only needed for `compare`
ollama pull mistral:7b       # only needed for `compare`

make install
```

## Usage

```bash
local-slm benchmark --model llama3.2:3b        # tokens/sec, TTFT, latency
local-slm extract --model llama3.2:3b          # JSON schema + retry pipeline
local-slm temp-experiment --model llama3.2:3b  # temp 0 vs 0.7
local-slm compare --trials-override mistral:7b=2   # multi-model comparison

local-slm stream "Explain photosynthesis in one sentence."  # live token stream
uvicorn local_slm.api.app:app --reload         # FastAPI service
```

Each command writes results as JSON + Markdown to `results/`.

## Project structure

```
src/local_slm/
├── ollama_client.py     # streaming client: TTFT, tokens/sec
├── prompts.py           # loads data/prompts.yaml
├── structured/          # JSON schema + validation + retry
├── experiments/         # temperature experiment
├── benchmarking/        # trial harness + memory parsing
├── scoring/             # quality scoring per task type
├── comparison/          # multi-model comparison runner
├── reporting/           # results -> Markdown
├── api/                 # FastAPI service
└── cli.py               # Typer CLI
data/prompts.yaml        # the 45 standardized prompts
results/                 # committed run outputs
tests/unit/               # mocked, network-free tests
```

## Testing

```bash
make test   # pytest, no Ollama/GPU/models required
make lint   # ruff check + format
```

CI runs lint + tests on every push. It does not run live benchmarks, since
those depend on local hardware — reproduce them with the commands above.
