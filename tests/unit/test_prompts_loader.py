from local_slm.prompts import filter_by_task, load_prompts
from local_slm.schemas import TaskType
from local_slm.structured.schemas_catalog import SCHEMA_REGISTRY


def test_loads_real_prompt_file():
    prompts = load_prompts()
    assert len(prompts) == 45


def test_task_type_distribution_is_15_each():
    prompts = load_prompts()
    for task_type in TaskType:
        assert len(filter_by_task(prompts, task_type)) == 15


def test_factual_qa_prompts_have_expected_answer():
    prompts = filter_by_task(load_prompts(), TaskType.FACTUAL_QA)
    assert all(p.expected_answer for p in prompts)


def test_summarization_prompts_have_expected_keywords():
    prompts = filter_by_task(load_prompts(), TaskType.SUMMARIZATION)
    assert all(p.expected_keywords for p in prompts)


def test_json_extraction_prompts_reference_valid_schema_and_fields():
    prompts = filter_by_task(load_prompts(), TaskType.JSON_EXTRACTION)
    for p in prompts:
        assert p.schema_name in SCHEMA_REGISTRY
        assert p.expected_fields
        schema_cls = SCHEMA_REGISTRY[p.schema_name]
        # Expected fields must themselves validate against the target schema.
        schema_cls.model_validate(p.expected_fields)


def test_ids_are_unique():
    prompts = load_prompts()
    ids = [p.id for p in prompts]
    assert len(ids) == len(set(ids))
