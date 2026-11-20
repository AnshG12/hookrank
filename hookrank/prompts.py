"""Prompt templates in prompts/*.md with {{NAME}} placeholders. Prompt identity is the template's
content hash (recorded in run_manifest.json), so any edit changes cache keys automatically."""
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def render(name: str, **values: str) -> str:
    template = (PROMPT_DIR / name).read_text(encoding="utf-8")
    for key, value in values.items():
        template = template.replace("{{" + key.upper() + "}}", value)
    return template
