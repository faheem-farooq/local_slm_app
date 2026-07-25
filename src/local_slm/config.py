"""Shared configuration: Ollama endpoint, model tags, paths, and experiment constants."""

from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"

# Ollama model tags used across the project. Mistral 7B is the largest of the
# three and the one most likely to feel memory pressure on 8GB unified-memory
# hardware -- see README for the measured impact, not hidden here.
MODEL_LLAMA32_3B = "llama3.2:3b"
MODEL_PHI4_MINI = "phi4-mini:3.8b"
MODEL_MISTRAL_7B = "mistral:7b"

COMPARISON_MODELS = [MODEL_LLAMA32_3B, MODEL_PHI4_MINI, MODEL_MISTRAL_7B]
DEFAULT_MODEL = MODEL_LLAMA32_3B

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = REPO_ROOT / "data" / "prompts.yaml"
RESULTS_DIR = REPO_ROOT / "results"

BENCHMARK_TRIALS = 5
TEMPERATURE_EXPERIMENT_REPEATS = 8
TEMP_LOW = 0.0
TEMP_HIGH = 0.7

FUZZY_MATCH_THRESHOLD = 85  # judgment call, documented in README/report, not rigorous ground truth
