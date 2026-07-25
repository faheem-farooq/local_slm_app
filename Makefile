.PHONY: install lint test bench extract temp-experiment compare serve

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest tests/unit -v

bench:
	local-slm benchmark --model llama3.2:3b

extract:
	local-slm extract --model llama3.2:3b

temp-experiment:
	local-slm temp-experiment --model llama3.2:3b

compare:
	local-slm compare

serve:
	uvicorn local_slm.api.app:app --reload
