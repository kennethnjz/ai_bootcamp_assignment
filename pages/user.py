# Standard library import for JSON parsing of model output.
import json
# Standard library import for all pattern matching used in job extraction.
import re
# Plotting library used by st.pyplot histogram rendering.
import matplotlib.pyplot as plt
# Plotly is used for interactive histogram zooming.
import plotly.graph_objects as go
# Used to print full stack traces when chat calls fail.
import traceback

# Streamlit UI framework for page rendering and session state handling.
import streamlit as st
# OpenAI SDK client for chat completion calls.
from openai import OpenAI
# Chroma vector store class for remote embeddings retrieval.
from langchain_chroma import Chroma
# Authentication and user-scoped vector-store utilities from local module.
from auth import is_logged_in, current_user, logout
# Shared app-level cache helper used by both admin and user pages.
from vectorstore_cache import get_cached_vectorstore


# Regex used to detect job IDs (including optional suffixes like '@' or '-') in text.
JOB_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]{2,}\d{3,}[A-Za-z0-9_@-]*)(?![A-Za-z0-9_])")


def retrieve_context(vectorstore, query: str, k: int = 4) -> tuple[str, list]:
    """Retrieve top-k relevant chunks from the vector store and join them as a context string."""
    # Build a retriever from the vector store with the requested top-k value.
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # Run retrieval against the query text.
    docs = retriever.invoke(query)
    # Join all chunk contents with separators so the model can read distinct chunks.
    joined_context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    # Return both the merged context text and the original doc objects.
    return joined_context, docs


def build_rag_system_prompt(context: str) -> str:
    """Create a strict RAG system prompt that limits answers to retrieved context only."""
    # Return a system instruction that prevents answers outside the provided context.
    return (
        "Answer ONLY using the context below. "
        'If the answer is not in the context, say "I couldn\'t find that information in the uploaded document."\n\n'
        f"Context:\n{context}"
    )


def build_sequence_diagram_prompt(context: str) -> str:
    """Create the prompt that asks the model for dependency JSON, not Mermaid syntax."""
    # Return detailed JSON-only instructions so downstream parsing is deterministic.
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
        f"Context:\n{context}"
    )


def build_hourly_jobs_prompt(context: str) -> str:
    """Create prompt that asks for hourly job-count JSON in a fixed 24-hour schema."""
    return (
        "You are given job run details from retrieved operating-manual context. "
        "Return only one JSON object with no markdown fences and no extra text.\n\n"
        "Output schema (must include all hours 0-23):\n"
        '{\n'
        '  "hourly_job_counts": [\n'
        '    {"hour": 0, "job_count": 0},\n'
        '    {"hour": 1, "job_count": 0}\n'
        '  ]\n'
        '}\n\n'
        "Rules:\n"
        "1. Include exactly 24 entries, one per hour from 0 to 23.\n"
        "2. hour must be an integer in [0, 23].\n"
        "3. job_count must be a non-negative integer.\n"
        "4. Use only explicit evidence from context. If unknown for an hour, use 0.\n"
        "5. Return valid JSON parseable by json.loads().\n\n"
        f"Context:\n{context}"
    )


def parse_hourly_job_counts(hourly_json: str) -> tuple[list[int], dict]:
    """Parse hourly job-count JSON into a fixed-size 24-value list for plotting."""
    # Drop optional markdown fencing if the model wrapped JSON output.
    hourly_json = hourly_json.replace("```json", "").replace("```", "").strip()
    # Start with a zero-filled 24-hour vector.
    counts = [0] * 24

    try:
        payload = json.loads(hourly_json)
    except Exception:
        return counts, {"hourly_job_counts": [{"hour": h, "job_count": 0} for h in range(24)]}

    if not isinstance(payload, dict):
        return counts, {"hourly_job_counts": [{"hour": h, "job_count": 0} for h in range(24)]}

    # Preferred schema: list of objects under hourly_job_counts.
    rows = payload.get("hourly_job_counts")
    if isinstance(rows, list):
        for item in rows:
            if not isinstance(item, dict):
                continue
            hour = item.get("hour")
            job_count = item.get("job_count")
            try:
                hour_int = int(hour)
                count_int = max(0, int(job_count))
            except Exception:
                continue
            if 0 <= hour_int <= 23:
                counts[hour_int] = count_int
    else:
        # Fallback schema: mapping {"0": 3, "1": 0, ...}.
        mapping = payload.get("hourly") or payload.get("hourly_counts") or payload.get("by_hour")
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                try:
                    hour_int = int(key)
                    count_int = max(0, int(value))
                except Exception:
                    continue
                if 0 <= hour_int <= 23:
                    counts[hour_int] = count_int

    normalized_payload = {
        "hourly_job_counts": [
            {"hour": hour, "job_count": counts[hour]}
            for hour in range(24)
        ]
    }
    return counts, normalized_payload


def build_hourly_jobs_histogram(store) -> tuple[list[int], dict]:
    """Retrieve schedule context and ask model for hourly job-count JSON."""
    retrieval_prompt = (
        "List all job run time information, schedules, trigger times, and hourly frequencies "
        "from the available documents."
    )
    context, _ = retrieve_context(store, retrieval_prompt, k=25)
    system_prompt = build_hourly_jobs_prompt(context)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0,
        stream=False,
    )
    raw_json = response.choices[0].message.content.strip()
    return parse_hourly_job_counts(raw_json)


def render_hourly_histogram(hourly_counts: list[int], chart_title: str = "Hourly Job Distribution"):
    """Render a 24-hour histogram-style bar chart using st.pyplot."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(list(range(24)), hourly_counts)
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Number of Jobs")
    ax.set_title(chart_title)
    ax.set_xticks(list(range(24)))
    st.pyplot(fig)
    plt.close(fig)


def render_visualization_widget(sequence_diagram: str, graphviz_diagram: str, hourly_counts: list[int]):
    """Render one visualization at a time in a selector-driven widget."""
    selected_view = st.radio(
        "Choose a visualization",
        ["Sequence Diagram", "Job Dependency Graph", "Jobs Run by Hour"],
        horizontal=True,
        key="visualization_widget_selector",
    )

    if selected_view == "Sequence Diagram":
        st.mermaid_chart(sequence_diagram)
    elif selected_view == "Job Dependency Graph":
        st.graphviz_chart(graphviz_diagram)
    else:
        render_hourly_histogram(hourly_counts, chart_title="Jobs Run by Hour")


def render_hourly_histogram_plotly(hourly_counts: list[int]):
    """Render an interactive hourly histogram that supports zoom and pan."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(range(24)),
                y=hourly_counts,
                marker_color="#1f77b4",
                hovertemplate="Hour %{x}:00<br>Jobs %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Jobs Run by Hour",
        xaxis_title="Hour of Day (0-23)",
        yaxis_title="Number of Jobs",
        xaxis=dict(dtick=1),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def render_visualization_fullscreen(sequence_diagram: str, graphviz_diagram: str, hourly_counts: list[int]):
    """Render a full-page visualization viewer with no modal clipping."""
    st.subheader("Visualization Viewer")
    st.caption("Histogram supports mouse/trackpad zoom.")
    tabs = st.tabs(["Sequence Diagram", "Job Dependency Graph", "Jobs Run by Hour"])

    with tabs[0]:
        st.mermaid_chart(sequence_diagram)
    with tabs[1]:
        st.graphviz_chart(graphviz_diagram)
    with tabs[2]:
        render_hourly_histogram_plotly(hourly_counts)


def sanitize_mermaid_identifier(name: str) -> str:
    """Strip unsupported Mermaid/Graphviz identifier characters from job IDs."""
    # Remove any character except letters, numbers, and underscore.
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", name)
    # Return a safe fallback node ID if cleaning removed everything.
    return cleaned or "p"


def normalize_job_tokens(sequence_items: list[str]) -> list[str]:
    """Extract normalized unique job IDs from arbitrary sequence token strings."""
    # Keep output in stable first-seen order.
    ordered_jobs = []
    # Track seen jobs to avoid duplicates.
    seen = set()

    # Iterate through every raw sequence item.
    for item in sequence_items:
        # Skip non-string values defensively.
        if not isinstance(item, str):
            continue

        # Trim whitespace around the token.
        item = item.strip()
        # Skip empty strings.
        if not item:
            continue

        # Extract all job-like IDs found inside this token.
        for match in JOB_ID_PATTERN.finditer(item):
            # Sanitize the extracted ID for downstream diagram safety.
            clean_job = sanitize_mermaid_identifier(match.group(0))
            # Ignore empty or already-seen job IDs.
            if not clean_job or clean_job in seen:
                continue
            # Mark this ID as seen.
            seen.add(clean_job)
            # Preserve first-seen ordering in output.
            ordered_jobs.append(clean_job)

    # Return deduplicated normalized job list.
    return ordered_jobs


def parse_sequence_json(sequence_json: str) -> list[str]:
    """Compatibility helper that returns only ordered_jobs from parsed payload."""
    # Parse via shared payload parser and return just the ordered job list.
    return parse_sequence_payload(sequence_json).get("ordered_jobs", [])


def parse_sequence_payload(sequence_json: str) -> dict:
    """Parse JSON response and extract normalized jobs and dependency edges."""
    # Remove optional markdown fences if the model wrapped JSON in code blocks.
    sequence_json = sequence_json.replace("```json", "").replace("```", "").strip()
    try:
        # Parse JSON text into Python data.
        payload = json.loads(sequence_json)
    except Exception:
        # Return empty payload shape when parsing fails.
        return {"job_ID": None, "ordered_jobs": [], "edges": []}

    # Validate the top-level structure is an object.
    if not isinstance(payload, dict):
        # Return empty payload shape for non-object JSON values.
        return {"job_ID": None, "ordered_jobs": [], "edges": []}

    # Ordered list of normalized jobs collected from all payload fields.
    ordered_jobs: list[str] = []
    # Set used to keep ordered_jobs unique.
    seen = set()
    # Collected dependency edges as (source, target) tuples.
    edges: list[tuple[str, str]] = []

    # Accept alternate field names for main job ID for robustness.
    job_id = payload.get("job_ID") or payload.get("jobId") or payload.get("main_job")
    # Parse and normalize main job ID if present.
    if isinstance(job_id, str):
        for match in JOB_ID_PATTERN.finditer(job_id):
            # Normalize extracted main job token.
            clean_job = sanitize_mermaid_identifier(match.group(0))
            # Add unseen normalized job to ordered list.
            if clean_job and clean_job not in seen:
                seen.add(clean_job)
                ordered_jobs.append(clean_job)

    # Accept alternate keys for the edge list.
    raw_edges = payload.get("edges") or payload.get("dependencies") or payload.get("links") or []
    # Parse edges only when the field is a list.
    if isinstance(raw_edges, list):
        # Inspect each list item and keep only dict-shaped edges.
        for item in raw_edges:
            # Skip non-dictionary edge entries.
            if not isinstance(item, dict):
                continue
            # Accept alternate source field names.
            source = item.get("from") or item.get("source") or item.get("job_ID")
            # Accept alternate target field names.
            target = item.get("to") or item.get("target") or item.get("depends_on")
            # Skip edges that do not provide string endpoints.
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            # Sanitize source ID.
            source_job = sanitize_mermaid_identifier(source)
            # Sanitize target ID.
            target_job = sanitize_mermaid_identifier(target)
            # Skip invalid or self-loop edges.
            if not source_job or not target_job or source_job == target_job:
                continue
            # Record valid edge tuple.
            edges.append((source_job, target_job))
            # Include source node in ordered list if new.
            if source_job not in seen:
                seen.add(source_job)
                ordered_jobs.append(source_job)
            # Include target node in ordered list if new.
            if target_job not in seen:
                seen.add(target_job)
                ordered_jobs.append(target_job)

    # Accept alternate field names for linear sequence fallback.
    sequence = payload.get("sequence") or payload.get("jobs") or payload.get("order")
    # Parse list-form sequence entries.
    if isinstance(sequence, list):
        # Iterate through each sequence item.
        for item in sequence:
            # Extract each job-like token from this item.
            for match in JOB_ID_PATTERN.finditer(str(item)):
                # Normalize extracted token.
                clean_job = sanitize_mermaid_identifier(match.group(0))
                # Add unseen normalized jobs to ordered list.
                if clean_job and clean_job not in seen:
                    seen.add(clean_job)
                    ordered_jobs.append(clean_job)
    # Parse string-form sequence entries.
    elif isinstance(sequence, str):
        # Extract all job-like tokens from sequence string.
        for match in JOB_ID_PATTERN.finditer(sequence):
            # Normalize extracted token.
            clean_job = sanitize_mermaid_identifier(match.group(0))
            # Add unseen normalized jobs to ordered list.
            if clean_job and clean_job not in seen:
                seen.add(clean_job)
                ordered_jobs.append(clean_job)

    # Return normalized payload fields used by both Mermaid and Graphviz builders.
    return {
        "job_ID": job_id,
        "ordered_jobs": ordered_jobs,
        "edges": edges,
    }


def build_mermaid_from_job_order(ordered_jobs: list[str], edges: list[tuple[str, str]] | None = None) -> str:
    """Build a Mermaid sequenceDiagram from ordered jobs and optional explicit edges."""
    # Return a minimal valid diagram if no jobs were found.
    if not ordered_jobs:
        return "sequenceDiagram\nparticipant p\n"

    # Normalize and deduplicate all input jobs.
    safe_jobs = normalize_job_tokens(ordered_jobs)
    # Return a minimal valid diagram if normalization removed all jobs.
    if not safe_jobs:
        return "sequenceDiagram\nparticipant p\n"

    # Return a one-node diagram for single-job flows.
    if len(safe_jobs) == 1:
        return f"sequenceDiagram\nparticipant {safe_jobs[0]}\n"

    # Start Mermaid output with required header.
    lines = ["sequenceDiagram"]
    # Declare every job as a participant.
    lines.extend(f"participant {job}" for job in safe_jobs)

    # Prefer explicit dependency edges when available.
    if edges:
        lines.extend(
            f"{source}-->>{target}: run after {target}"
            for source, target in edges
        )
    else:
        # Fallback to a linear chain if no edges were provided.
        lines.extend(
            f"{safe_jobs[i]}-->>{safe_jobs[i + 1]}: run after {safe_jobs[i + 1]}"
            for i in range(len(safe_jobs) - 1)
        )
    # Return final Mermaid text with trailing newline.
    return "\n".join(lines) + "\n"


def build_graphviz_from_sequence_json(sequence_json: str, ordered_jobs: list[str] | None = None) -> str:
    """Build a Graphviz digraph from parsed sequence JSON and normalized job list."""
    # Parse payload from model JSON output.
    payload = parse_sequence_payload(sequence_json)
    # Normalize jobs from explicit ordered list or parsed payload fallback.
    jobs = normalize_job_tokens(ordered_jobs or payload.get("ordered_jobs", []))
    # Load parsed edges from payload.
    edges = payload.get("edges", [])

    # Return a minimal valid graph if no jobs were found.
    if not jobs:
        return 'digraph G {\n  rankdir=LR;\n  "p";\n}'

    # Start Graphviz output with left-to-right direction and box style.
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded"];',
    ]
    # Declare each job node.
    for job in jobs:
        lines.append(f'  "{job}";')
    # Add explicit dependency edges when available.
    for source, target in edges:
        lines.append(f'  "{source}" -> "{target}" [label="run after"];')
    # Fallback to linear edges when no explicit edges were parsed.
    if not edges:
        for i in range(len(jobs) - 1):
            lines.append(f'  "{jobs[i]}" -> "{jobs[i + 1]}" [label="run after"];')
    # Close graph block.
    lines.append("}")
    # Return final Graphviz text with trailing newline.
    return "\n".join(lines) + "\n"


def extract_job_ids_from_context(context: str) -> list[str]:
    """Extract unique job IDs from raw context in first-appearance order."""
    # Ordered list of cleaned job IDs.
    ordered_jobs = []
    # Set to enforce uniqueness.
    seen = set()

    # Scan all regex matches in the full context text.
    for match in JOB_ID_PATTERN.finditer(context):
        # Capture matched token and remove leading/trailing spaces.
        raw_job = match.group(0).strip()
        # Sanitize token for diagram-safe usage.
        clean_job = sanitize_mermaid_identifier(raw_job)
        # Skip empty or duplicate entries.
        if not clean_job or clean_job in seen:
            continue
        # Mark ID as seen.
        seen.add(clean_job)
        # Append first occurrence to ordered output.
        ordered_jobs.append(clean_job)

    # Return ordered unique job IDs.
    return ordered_jobs


def extract_job_dependency_order(context: str) -> list[str]:
    """Infer dependency-aware job order from context using explicit phrase patterns."""
    # Start from all job IDs in appearance order.
    all_jobs = extract_job_ids_from_context(context)
    # Store original appearance index for stable tie-breaking.
    appearance_index = {job: index for index, job in enumerate(all_jobs)}

    # Patterns that capture "job runs after dependency" style relationships.
    dependency_patterns = [
        re.compile(
            r"(?<![A-Za-z0-9_@-])([A-Za-z]{2,}\d{3,}[A-Za-z0-9_@-]*)(?![A-Za-z0-9_@-])(?:\s+[^\n]{0,80})?\b(?:runs after|depends on|depends upon|must run after)\b(?:\s+[^\n]{0,80})?(?<![A-Za-z0-9_@-])([A-Za-z]{2,}\d{3,}[A-Za-z0-9_@-]*)(?![A-Za-z0-9_@-])",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?<![A-Za-z0-9_@-])([A-Za-z]{2,}\d{3,}[A-Za-z0-9_@-]*)(?![A-Za-z0-9_@-])(?:\s+[^\n]{0,80})?\b(?:run after|after)\b(?:\s+[^\n]{0,80})?(?<![A-Za-z0-9_@-])([A-Za-z]{2,}\d{3,}[A-Za-z0-9_@-]*)(?![A-Za-z0-9_@-])",
            flags=re.IGNORECASE,
        ),
    ]

    # Map of job -> dependency jobs it requires.
    dependency_map: dict[str, set[str]] = {job: set() for job in all_jobs}
    # Reverse map of dependency -> jobs that depend on it.
    dependents_map: dict[str, set[str]] = {job: set() for job in all_jobs}
    # In-degree count for topological sorting.
    in_degree: dict[str, int] = {job: 0 for job in all_jobs}

    # Apply each pattern and record discovered edges.
    for pattern in dependency_patterns:
        # Iterate through all matches for this pattern in context.
        for match in pattern.finditer(context):
            # Left side job is the dependent job.
            job_name = sanitize_mermaid_identifier(match.group(1))
            # Right side job is the prerequisite dependency.
            dep_name = sanitize_mermaid_identifier(match.group(2))
            # Skip invalid or self-referential matches.
            if not job_name or not dep_name or job_name == dep_name:
                continue
            # Initialize unseen dependent node structures.
            if job_name not in dependency_map:
                dependency_map[job_name] = set()
                dependents_map[job_name] = set()
                in_degree[job_name] = 0
                appearance_index[job_name] = len(appearance_index)
            # Initialize unseen dependency node structures.
            if dep_name not in dependency_map:
                dependency_map[dep_name] = set()
                dependents_map[dep_name] = set()
                in_degree[dep_name] = 0
                appearance_index[dep_name] = len(appearance_index)

            # Add dependency edge only once.
            if dep_name not in dependency_map[job_name]:
                dependency_map[job_name].add(dep_name)
                dependents_map[dep_name].add(job_name)
                in_degree[job_name] += 1

    # Build initial queue of nodes with no unmet dependencies.
    queue = sorted([job for job, degree in in_degree.items() if degree == 0], key=lambda item: appearance_index[item])
    # Result list from topological ordering.
    ordered_jobs = []

    # Kahn's algorithm loop.
    while queue:
        # Pop earliest appearance node from current ready queue.
        current = queue.pop(0)
        # Emit node into ordered output.
        ordered_jobs.append(current)

        # Visit dependents and reduce their in-degree.
        for dependent in sorted(dependents_map[current], key=lambda item: appearance_index[item]):
            in_degree[dependent] -= 1
            # Enqueue dependent once all prerequisites are satisfied.
            if in_degree[dependent] == 0:
                queue.append(dependent)

        # Keep queue deterministic by original appearance order.
        queue.sort(key=lambda item: appearance_index[item])

    # If cycles or unmatched nodes exist, append missing jobs by appearance order.
    if len(ordered_jobs) < len(all_jobs):
        for job in all_jobs:
            if job not in ordered_jobs:
                ordered_jobs.append(job)

    # Return dependency order when available, otherwise fallback to appearance order.
    return ordered_jobs or all_jobs


def build_run_sequence_diagram(store) -> tuple[str, str]:
    """Retrieve context, ask model for dependency JSON, and build Mermaid + Graphviz diagrams."""
    # Retrieval query focused on explicit run-order phrasing.
    default_prompt = (
        "Extract the job run sequence from the available documents. "
        "Use explicit dependency phrases such as 'runs after', 'depends on', or 'must run after' "
        "when they appear in the text."
    )

    # Pull a broad context window to increase chance of finding all job links.
    context, _ = retrieve_context(store, default_prompt, k=25)
    # Build strict JSON output instructions for sequence extraction.
    system_prompt = build_sequence_diagram_prompt(context)

    # Call the model in non-streaming mode to get one JSON payload.
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0,
        stream=False,
    )
    # Read raw model response text.
    raw_json = response.choices[0].message.content.strip()
    # Parse normalized jobs and edges from model output.
    payload = parse_sequence_payload(raw_json)
    # Preferred ordered jobs from parsed payload.
    ordered_jobs = payload.get("ordered_jobs", [])
    # Preferred explicit edges from parsed payload.
    edges = payload.get("edges", [])

    # Fallback to deterministic regex-based dependency extraction when model list is empty.
    if not ordered_jobs:
        ordered_jobs = extract_job_dependency_order(context)
    # Final fallback to simple appearance-order extraction.
    if not ordered_jobs:
        ordered_jobs = extract_job_ids_from_context(context)

    # Build Mermaid sequence diagram string.
    mermaid_diagram = build_mermaid_from_job_order(ordered_jobs, edges)
    # Build Graphviz dependency graph string.
    graphviz_diagram = build_graphviz_from_sequence_json(raw_json, ordered_jobs)
    # Return both diagram representations.
    return mermaid_diagram, graphviz_diagram


# Default assistant behavior when no document context is available.
system_prompt_no_doc = "You are a helpful assistant."


def load_persisted_store_for(user_id: str) -> Chroma | None:
    """Load app-cached Chroma Cloud vector store by owner id if present."""
    # Resolve shared app-level cached handle for this owner id.
    return get_cached_vectorstore(user_id)


def load_persisted_user_store() -> Chroma | None:
    """Load the logged-in user's Chroma Cloud vector store if present."""
    # Load the store keyed to the currently logged-in user id.
    return load_persisted_store_for(current_user())


# Initialize default retrieval depth in session state.
if "k_value" not in st.session_state:
    st.session_state.k_value = 100
# Initialize default chat model in session state.
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"
# Initialize default temperature in session state.
if "temperature" not in st.session_state:
    st.session_state.temperature = 1.0

# Create OpenAI API client from Streamlit secret.
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# Read retrieval depth into local variable.
k_value = st.session_state.k_value
# Read active model into local variable.
model = st.session_state.model
# Read active temperature into local variable.
temperature = st.session_state.temperature


# Redirect to main page if no valid logged-in user session exists.
if not is_logged_in() or not current_user().lower().startswith("user"):
    st.switch_page("main.py")

# Render page title.
st.title("User Page")
# Show current logged-in username.
st.write(f"Welcome, **{current_user()}**!")
# Show role-level status message.
st.info("You have standard user access.")

# Initialize chat history container in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Resolve vector store for this request using shared app-level cache.
active_owner = current_user()
# First preference: the logged-in user's own persisted store.
store = load_persisted_user_store()
# Fallback: shared admin-indexed store under the literal "user" scope.
if store is None:
    store = load_persisted_store_for("user")
    if store is not None:
        active_owner = "user"

# Render sequence diagram and graph only when a vector store exists.
if store is not None:
    # Read metadata for UI summary count.
    store_meta = store.get()
    # Confirm vector store availability and indicate source scope.
    if active_owner.lower() == current_user().lower():
        st.success(f"Vector store is available for user '{current_user()}'.")
    else:
        st.success(f"Using shared vector store from '{active_owner}'.")
    # Show count of indexed document IDs from the store metadata.
    st.caption(f"Indexed documents in session store: {len(store_meta.get('ids', []))}")

    # Generate diagrams with a loading spinner while model and parsing run.
    with st.spinner("Preparing run sequence diagram..."):
        sequence_diagram, graphviz_diagram = build_run_sequence_diagram(store)

    # Build a lightweight key so this expensive prompt runs only when store shape changes.
    histogram_cache_key = (
        active_owner.lower(),
        len(store_meta.get("ids", [])),
    )

    # Recompute hourly distribution only when the source store has changed.
    if st.session_state.get("hourly_jobs_cache_key") != histogram_cache_key:
        with st.spinner("Preparing hourly jobs histogram..."):
            hourly_counts, hourly_payload = build_hourly_jobs_histogram(store)
        st.session_state.hourly_jobs_cache_key = histogram_cache_key
        st.session_state.hourly_jobs_counts = hourly_counts
        st.session_state.hourly_jobs_payload = hourly_payload

    # Read cached values for stable rendering across reruns.
    hourly_counts = st.session_state.get("hourly_jobs_counts", [0] * 24)
    hourly_payload = st.session_state.get(
        "hourly_jobs_payload",
        {"hourly_job_counts": [{"hour": hour, "job_count": 0} for hour in range(24)]},
    )

    # Show a button-like trigger that opens a full-page visualization viewer.
    if "show_visualization_viewer" not in st.session_state:
        st.session_state.show_visualization_viewer = False

    if st.button("Open Visualization Widget", key="open_visualization_widget_btn"):
        st.session_state.show_visualization_viewer = True

    if st.session_state.show_visualization_viewer:
        if st.button("Close Visualization Widget", key="close_visualization_widget_btn"):
            st.session_state.show_visualization_viewer = False
        else:
            render_visualization_fullscreen(sequence_diagram, graphviz_diagram, hourly_counts)
else:
    # Inform user that no store exists yet for retrieval-backed behavior.
    st.info("No vector store is available in this session yet.")

# Replay full prior chat history in UI.
for message in st.session_state.messages:
    # Render each message in role-appropriate chat bubble.
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Capture new user message from chat input box.
if prompt := st.chat_input("Type a message..."):
    # Persist user message in chat history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Render the just-submitted user message bubble.
    with st.chat_message("user"):
        st.write(prompt)

    # Start assistant response bubble.
    with st.chat_message("assistant"):
        try:
            # Use RAG context only when user-specific store is available.
            if store is not None:
                # Retrieve relevant context and source chunks for this user question.
                context, source_docs = retrieve_context(store, prompt, k_value)
                # Build strict context-bound system prompt.
                effective_system_prompt = build_rag_system_prompt(context)
            else:
                # No store means no sources are available.
                source_docs = []
                # Fall back to a generic system prompt.
                effective_system_prompt = system_prompt_no_doc

            # Build full message list with current system instruction at the front.
            api_messages = [{"role": "system", "content": effective_system_prompt}] + st.session_state.messages
            # Start streaming chat completion from model.
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                stream=True,
            )
            # Stream assistant reply directly into the UI and capture final text.
            reply = st.write_stream(stream)

            # Show retrieval source chunks when RAG context was used.
            if source_docs:
                with st.expander("🔍 View Sources"):
                    # Render each source chunk with an index label.
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.caption(doc.page_content)
        except Exception:
            # Print full traceback to logs for debugging.
            traceback.print_exc()
            # Set fallback reply text for UI continuity.
            reply = "⚠️ Sorry, something went wrong. Please try again."
            # Show visible error state to the user.
            st.error(reply)

    # Persist assistant response in chat history.
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Provide logout action button.
if st.button("Logout"):
    # Clear auth/session state through helper.
    logout()
    # Redirect user to main login page.
    st.switch_page("main.py")
