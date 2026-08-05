from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pages.diagram_utils import build_sequence_diagram_prompt, extract_job_series_from_prompt


def test_extract_job_series_from_prompt() -> None:
    prompt = "Please redraw the dependency graph for DKSD001, DKSD002, and DKSD003"

    assert extract_job_series_from_prompt(prompt) == ["DKSD001", "DKSD002", "DKSD003"]


def test_build_sequence_diagram_prompt_mentions_focus_jobs() -> None:
    prompt = build_sequence_diagram_prompt("context", focus_jobs=["DKSD001", "DKSD002"])

    assert "DKSD001" in prompt
    assert "DKSD002" in prompt
    assert "focus" in prompt.lower()
