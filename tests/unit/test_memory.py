from local_slm.benchmarking.memory import get_loaded_model_memory, parse_ollama_ps
from tests.conftest import FIXTURES_DIR


def test_parse_ollama_ps_finds_matching_model():
    output = (FIXTURES_DIR / "ollama_ps_sample.txt").read_text()
    info = parse_ollama_ps(output, "llama3.2:3b")
    assert info is not None
    assert info.size_str == "3.4 GB"
    assert info.processor == "100% GPU"


def test_parse_ollama_ps_returns_none_for_unknown_model():
    output = (FIXTURES_DIR / "ollama_ps_sample.txt").read_text()
    assert parse_ollama_ps(output, "mistral:7b") is None


def test_parse_ollama_ps_handles_empty_output():
    assert parse_ollama_ps("", "llama3.2:3b") is None


def test_get_loaded_model_memory_uses_injected_subprocess_run():
    class FakeCompletedProcess:
        stdout = (FIXTURES_DIR / "ollama_ps_sample.txt").read_text()

    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return FakeCompletedProcess()

    info = get_loaded_model_memory("llama3.2:3b", run_fn=fake_run)
    assert info.size_str == "3.4 GB"
    assert calls["cmd"] == ["ollama", "ps"]
