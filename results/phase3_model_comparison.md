# Model comparison

| Model | Tokens/sec | TTFT (s) | Memory | Factual acc. | Summarization | JSON valid rate | JSON field acc. |
|---|---|---|---|---|---|---|---|
| llama3.2:3b | 19.1 ± 3.6 tok/s (n=225) | 0.29 ± 0.09 s (n=225) | 2.5 GB | 100.00% | 0.85 | 100.00% | 97.78% |
| phi4-mini:3.8b | 14.6 ± 2.5 tok/s (n=225) | 0.39 ± 0.33 s (n=225) | 3.1 GB | 93.33% | 0.79 | 100.00% | 95.56% |
| mistral:7b | 5.9 ± 2.1 tok/s (n=90) | 1.09 ± 1.44 s (n=90) | 5.1 GB | 100.00% | 0.91 | 100.00% | 95.56% |

Quality scoring is heuristic, not human- or LLM-judged: factual accuracy is exact/substring/fuzzy match against a short reference answer; summarization score is 0.7 x keyword coverage + 0.3 x an ideal-length band; JSON metrics come from Pydantic schema validation plus per-field comparison against expected values. See README for details and caveats.