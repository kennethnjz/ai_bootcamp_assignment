import re
from typing import Optional

JOB_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]{2,}\d{3,}[A-Za-z0-9_@-]*)(?![A-Za-z0-9_])")


def sanitize_mermaid_identifier(name: str) -> str:
    """Strip unsupported Mermaid/Graphviz identifier characters from job IDs."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", name)
    return cleaned or "p"


def extract_job_series_from_prompt(prompt: str) -> list[str]:
    """Extract job IDs from prompts that request a dependency-graph redraw."""
    if not isinstance(prompt, str):
        return []

    normalized = prompt.strip().lower()
    if not normalized:
        return []

    trigger_terms = [
        "redraw",
        "rebuild",
        "show",
        "diagram",
        "graph",
        "sequence",
        "dependency",
        "job series",
        "jobs",
    ]
    if not any(term in normalized for term in trigger_terms):
        return []

    ordered_jobs: list[str] = []
    seen: set[str] = set()
    for match in JOB_ID_PATTERN.finditer(prompt):
        clean_job = sanitize_mermaid_identifier(match.group(0))
        if not clean_job or clean_job in seen:
            continue
        seen.add(clean_job)
        ordered_jobs.append(clean_job)
    return ordered_jobs


def build_sequence_diagram_prompt(context: str, focus_jobs: Optional[list[str]] = None) -> str:
    """Create the prompt that asks the model for dependency JSON, not Mermaid syntax."""
    focus_instructions = ""
    if focus_jobs:
        focus_list = ", ".join(focus_jobs)
        focus_instructions = (
            "\nFocus requirement:\n"
            f"Prioritize the following job IDs when building the dependency graph: {focus_list}.\n"
            "If the retrieved context includes additional upstream or downstream jobs that are necessary to preserve the dependency path, include them as well."
        )

    return (
        "You are given the retrieved job documents. "
        "Return only a single JSON object with no markdown fences and no explanation.\n\n"
        "Output schema:\n"
        '{\n'
        '    "job_ID": "JOB_ID_1",\n'
        '    "edges": [\n'
        '        {"from": "JOB_ID_2", "to": "JOB_ID_3"},\n'
        '        {"from": "JOB_ID_2", "to": "JOB_ID_4"}\n'
        '    ]\n'
        '}\n\n'
        "Instructions:\n"
        "1. Include every explicitly named job ID from the retrieved context in the final dependency graph. "
        "This includes the main job and every dependent job that appears in the text.\n"
        "2. The 'job_ID' field must contain the main job ID that appears in the retrieved context.\n"
        "3. The 'edges' field must contain the full dependency relationship list. "
        "Each edge must use 'from' and 'to' keys, where 'from' is the job that runs before 'to'.\n"
        "4. Do not omit any job that is clearly named in the text, including jobs like DKSD012 or suffixed forms like DKSD002@.\n"
        "5. Only include job IDs that are explicitly written in the retrieved context as job names. "
        "Do not invent, infer, or add any other job ID that is not directly present in the text.\n"
        "6. Ignore any non-job entities, labels, or descriptive words.\n"
        "7. Determine the dependencies from the document evidence, not from alphabetical order.\n"
        "8. The 'edges' array must represent all explicit dependency links between jobs.\n"
        "9. Every value in the JSON object must be a bare job ID only, such as DKSD004 or DKSM018. "
        "Do not include arrows, participant lines, markdown fences, or Mermaid syntax.\n"
        "10. JSON must be valid and parseable with Python's json.loads().\n"
        "11. Do not add any prose outside the JSON object.\n"
        "12. The number of the estimated run time is in the number of minutes.\n"
        "13. The time format shown in the scheduling instructions is HH:MM PM/AM.\n"
        "14. When the day of the week is displayed as Mon - Fri, it refers to Monday through Friday. Interpret similar formats in this field as such.\n"
        "15. The days of the week in the scheduling instructions include Mon - Monday, Tue - Tuesday, Wed - Wednesday, Thu - Thursday, Fri - Friday.\n"
        "16. If there are any scheduling instructions that do not specify the time or day of the week, look for keywords such as 'Run after' with a specific job ID, and 'to run a specific job ID after'.\n"
        "17. If that is the case, calculate the starting run time by tracing the scheduling instructions of the job ID and add the estimated run time of that job to the starting run time.\n"
        "18. Do not invent, infer, or add any other job ID that is not directly present in the text.\n"
        "19. If the job's last run date is mentioned and is before the current date, indicate that the job is overdue and do not include it in your result.\n"
                
        f"{focus_instructions}\n"
        f"Context:\n{context}"
    )
