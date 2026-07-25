# Temperature experiment: llama3.2:3b (n_repeats=8)

| Metric | temp=0.0 | temp=0.7 |
|---|---|---|
| Mean response diversity (unique/repeats) | 0.12 | 0.50 |
| JSON schema validation failure rate | 0.00% | 0.00% |

Mean agreement rate of temp=0.7 responses vs. the temp=0 reference answer: **49.72%**

## Per-prompt detail

| Prompt | Task | temp0 diversity | temp07 diversity | agreement vs temp0 |
|---|---|---|---|---|
| fact_01 | factual_qa | 0.12 | 0.12 | 100.00% |
| fact_02 | factual_qa | 0.12 | 0.12 | 100.00% |
| fact_03 | factual_qa | 0.12 | 0.25 | 75.00% |
| fact_04 | factual_qa | 0.12 | 0.25 | 87.50% |
| fact_05 | factual_qa | 0.12 | 0.25 | 50.00% |
| fact_06 | factual_qa | 0.12 | 0.25 | 87.50% |
| fact_07 | factual_qa | 0.12 | 0.12 | 100.00% |
| fact_08 | factual_qa | 0.12 | 0.25 | 75.00% |
| fact_09 | factual_qa | 0.12 | 0.50 | 37.50% |
| fact_10 | factual_qa | 0.12 | 0.12 | 100.00% |
| fact_11 | factual_qa | 0.12 | 0.38 | 37.50% |
| fact_12 | factual_qa | 0.12 | 0.25 | 75.00% |
| fact_13 | factual_qa | 0.12 | 0.25 | 62.50% |
| fact_14 | factual_qa | 0.12 | 0.38 | 37.50% |
| fact_15 | factual_qa | 0.12 | 0.25 | 25.00% |
| summ_01 | summarization | 0.12 | 1.00 | 0.00% |
| summ_02 | summarization | 0.12 | 1.00 | 0.00% |
| summ_03 | summarization | 0.12 | 1.00 | 0.00% |
| summ_04 | summarization | 0.12 | 1.00 | 0.00% |
| summ_05 | summarization | 0.12 | 1.00 | 0.00% |
| summ_06 | summarization | 0.12 | 1.00 | 0.00% |
| summ_07 | summarization | 0.12 | 1.00 | 0.00% |
| summ_08 | summarization | 0.12 | 1.00 | 0.00% |
| summ_09 | summarization | 0.12 | 1.00 | 0.00% |
| summ_10 | summarization | 0.12 | 1.00 | 0.00% |
| summ_11 | summarization | 0.12 | 1.00 | 0.00% |
| summ_12 | summarization | 0.12 | 1.00 | 0.00% |
| summ_13 | summarization | 0.12 | 1.00 | 0.00% |
| summ_14 | summarization | 0.12 | 1.00 | 0.00% |
| summ_15 | summarization | 0.12 | 1.00 | 12.50% |
| json_contact_01 | json_extraction | 0.12 | 0.25 | 75.00% |
| json_contact_02 | json_extraction | 0.12 | 0.25 | 62.50% |
| json_contact_03 | json_extraction | 0.12 | 0.38 | 62.50% |
| json_contact_04 | json_extraction | 0.12 | 0.25 | 62.50% |
| json_contact_05 | json_extraction | 0.12 | 0.38 | 50.00% |
| json_event_01 | json_extraction | 0.12 | 0.12 | 100.00% |
| json_event_02 | json_extraction | 0.12 | 0.25 | 87.50% |
| json_event_03 | json_extraction | 0.12 | 0.25 | 75.00% |
| json_event_04 | json_extraction | 0.12 | 0.12 | 100.00% |
| json_event_05 | json_extraction | 0.12 | 0.12 | 100.00% |
| json_product_01 | json_extraction | 0.12 | 0.25 | 75.00% |
| json_product_02 | json_extraction | 0.12 | 0.12 | 100.00% |
| json_product_03 | json_extraction | 0.12 | 0.38 | 62.50% |
| json_product_04 | json_extraction | 0.12 | 0.25 | 75.00% |
| json_product_05 | json_extraction | 0.12 | 0.25 | 87.50% |