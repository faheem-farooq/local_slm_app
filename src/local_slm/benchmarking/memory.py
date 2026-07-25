"""Parses `ollama ps` to report a loaded model's advertised memory footprint.

Caveat (documented, not hidden): this is Ollama's own reported model size, not a
precise per-process OS-level RSS measurement. Meaningful comparison across models
requires exactly one model loaded at a time, which matches Ollama's own default
swap-on-demand behavior -- the comparison runner never interleaves models.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from pydantic import BaseModel

_COLUMNS = ["NAME", "ID", "SIZE", "PROCESSOR", "UNTIL"]


class ModelMemoryInfo(BaseModel):
    model: str
    size_str: str
    processor: str


def parse_ollama_ps(output: str, model: str) -> ModelMemoryInfo | None:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return None

    header = lines[0]
    offsets = []
    for col in _COLUMNS:
        idx = header.find(col)
        if idx == -1:
            return None
        offsets.append(idx)
    offsets.append(len(header) + 1000)  # generous sentinel so the last column isn't truncated

    for row in lines[1:]:
        values = {col: row[offsets[i] : offsets[i + 1]].strip() for i, col in enumerate(_COLUMNS)}
        if values["NAME"] == model:
            return ModelMemoryInfo(
                model=model, size_str=values["SIZE"], processor=values["PROCESSOR"]
            )
    return None


def get_loaded_model_memory(
    model: str, run_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run
) -> ModelMemoryInfo | None:
    result = run_fn(["ollama", "ps"], capture_output=True, text=True, check=True)
    return parse_ollama_ps(result.stdout, model)
