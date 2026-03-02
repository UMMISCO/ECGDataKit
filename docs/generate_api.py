#!/usr/bin/env python3
"""Generate per-function Hugo API reference pages from docstrings.

Usage:
    python docs/generate_api.py

Generates one markdown page per public function from ecgdatakit.processing
and ecgdatakit.plotting into docs/content/api/ref/.
"""

from __future__ import annotations

import html
import inspect
import re
import shutil
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the library is importable
# ---------------------------------------------------------------------------
LIB_SRC = Path(__file__).resolve().parent.parent / "lib" / "src"
sys.path.insert(0, str(LIB_SRC))

import ecgdatakit.plotting as plotting_mod  # noqa: E402
import ecgdatakit.processing as processing_mod  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "content" / "api" / "ref"


# ---------------------------------------------------------------------------
# Lightweight numpy-style docstring parser
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^(\w[\w\s]*)$")
_DASHES_RE = re.compile(r"^-{3,}$")


def _parse_numpy_docstring(docstring: str | None) -> dict:
    """Parse a numpy-style docstring into structured sections.

    Returns a dict with keys:
      summary        - first paragraph (str)
      description    - extended description after summary (str)
      parameters     - list of (name, type, description)
      returns        - list of (type, description)
      other_sections - dict of section_name -> text
    """
    result = {
        "summary": "",
        "description": "",
        "parameters": [],
        "returns": [],
        "other_sections": {},
    }
    if not docstring:
        return result

    lines = inspect.cleandoc(docstring).splitlines()

    # --- extract summary (first non-empty paragraph) ---
    summary_lines: list[str] = []
    idx = 0
    while idx < len(lines) and lines[idx].strip():
        summary_lines.append(lines[idx].strip())
        idx += 1
    result["summary"] = " ".join(summary_lines)

    # skip blank lines after summary
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    # --- split remaining into sections ---
    sections: dict[str, list[str]] = {}
    current_section = "_description"
    sections[current_section] = []

    while idx < len(lines):
        line = lines[idx]
        # Check if next line is a dashes underline (section header)
        if (
            idx + 1 < len(lines)
            and _DASHES_RE.match(lines[idx + 1].strip())
            and _SECTION_RE.match(line.strip())
        ):
            current_section = line.strip().lower()
            sections.setdefault(current_section, [])
            idx += 2  # skip header + dashes
            continue
        sections.setdefault(current_section, []).append(line)
        idx += 1

    result["description"] = "\n".join(sections.get("_description", [])).strip()

    # --- parse Parameters section ---
    if "parameters" in sections:
        result["parameters"] = _parse_param_section(sections["parameters"])

    # --- parse Returns section ---
    if "returns" in sections:
        result["returns"] = _parse_returns_section(sections["returns"])

    return result


def _parse_param_section(lines: list[str]) -> list[tuple[str, str, str]]:
    """Parse a Parameters section into (name, type, description) tuples."""
    params: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Parameter line: "name : type"
        if " : " in line:
            name, typ = line.split(" : ", 1)
            name = name.strip()
            typ = typ.strip()
            # Collect description lines (indented)
            desc_lines: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t") or not lines[i].strip()):
                stripped = lines[i].strip()
                if stripped:
                    desc_lines.append(stripped)
                elif desc_lines:  # blank line within description
                    desc_lines.append("")
                i += 1
            desc = " ".join(dl for dl in desc_lines if dl).strip()
            params.append((name, typ, desc))
        else:
            i += 1
    return params


def _parse_returns_section(lines: list[str]) -> list[tuple[str, str]]:
    """Parse a Returns section into (type, description) tuples."""
    returns: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Type line (not indented or lightly indented)
        typ = line
        desc_lines: list[str] = []
        i += 1
        while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t") or not lines[i].strip()):
            stripped = lines[i].strip()
            if stripped:
                desc_lines.append(stripped)
            i += 1
        desc = " ".join(desc_lines).strip()
        returns.append((typ, desc))
    return returns


# ---------------------------------------------------------------------------
# Signature formatting
# ---------------------------------------------------------------------------

def _format_signature(func) -> str:
    """Format a function signature as a clean string."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return f"{func.__name__}(...)"

    parts: list[str] = []
    seen_keyword_only = False

    for name, param in sig.parameters.items():
        if param.kind == param.VAR_POSITIONAL:
            parts.append(f"*{name}")
            seen_keyword_only = True  # keyword-only follows *args implicitly
            continue
        if param.kind == param.VAR_KEYWORD:
            parts.append(f"**{name}")
            continue
        if param.kind == param.KEYWORD_ONLY and not seen_keyword_only:
            parts.append("*")
            seen_keyword_only = True

        if param.default is inspect.Parameter.empty:
            parts.append(name)
        else:
            default = param.default
            if default is None:
                default_str = "None"
            elif isinstance(default, str):
                default_str = repr(default)
            elif isinstance(default, tuple):
                default_str = repr(default)
            elif isinstance(default, bool):
                default_str = repr(default)
            else:
                default_str = str(default)
            parts.append(f"{name}={default_str}")

    params_str = ", ".join(parts)
    return f"{func.__name__}({params_str})"


def _get_return_annotation(func) -> str:
    """Get the return type annotation as a string."""
    try:
        sig = inspect.signature(func)
        if sig.return_annotation is not inspect.Parameter.empty:
            ann = sig.return_annotation
            if hasattr(ann, "__name__"):
                return ann.__name__
            return str(ann).replace("typing.", "")
    except (ValueError, TypeError):
        pass
    return ""


# ---------------------------------------------------------------------------
# Determine source module
# ---------------------------------------------------------------------------

def _source_module(func) -> str:
    """Get the fully qualified module path of a function."""
    mod = getattr(func, "__module__", "")
    return mod


def _import_path(func, parent_module: str) -> str:
    """Get the user-facing import path like 'from ecgdatakit.processing import X'."""
    return f"from {parent_module} import {func.__name__}"


# ---------------------------------------------------------------------------
# Page generation
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """HTML-escape text for use in table cells."""
    return html.escape(text)


def _render_page(
    func,
    parent_module: str,
    weight: int,
    category: str,
) -> str:
    """Render a single function's Hugo markdown page."""
    name = func.__name__
    doc = _parse_numpy_docstring(func.__doc__)
    sig = _format_signature(func)
    source_mod = _source_module(func)
    import_line = _import_path(func, parent_module)

    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append(f'title: "{name}"')
    lines.append(f"weight: {weight}")
    lines.append("---")
    lines.append("")

    # Module path
    lines.append(f"`{source_mod}.{name}`")
    lines.append("")

    # Category badge
    lines.append(f"**Module:** {category}")
    lines.append("")

    # Signature
    lines.append("## Signature")
    lines.append("")
    lines.append("```python")
    lines.append(sig)
    lines.append("```")
    lines.append("")

    # Description
    summary = doc["summary"]
    description = doc["description"]
    if summary:
        lines.append("## Description")
        lines.append("")
        lines.append(summary)
        if description:
            lines.append("")
            lines.append(description)
        lines.append("")

    # Parameters
    if doc["parameters"]:
        lines.append("## Parameters")
        lines.append("")
        lines.append("<table>")
        lines.append("  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>")
        lines.append("  <tbody>")
        for pname, ptype, pdesc in doc["parameters"]:
            lines.append(
                f"    <tr>"
                f"<td><code>{_esc(pname)}</code></td>"
                f"<td><code>{_esc(ptype)}</code></td>"
                f"<td>{_esc(pdesc)}</td>"
                f"</tr>"
            )
        lines.append("  </tbody>")
        lines.append("</table>")
        lines.append("")

    # Returns
    if doc["returns"]:
        lines.append("## Returns")
        lines.append("")
        lines.append("<table>")
        lines.append("  <thead><tr><th>Type</th><th>Description</th></tr></thead>")
        lines.append("  <tbody>")
        for rtype, rdesc in doc["returns"]:
            lines.append(
                f"    <tr>"
                f"<td><code>{_esc(rtype)}</code></td>"
                f"<td>{_esc(rdesc)}</td>"
                f"</tr>"
            )
        lines.append("  </tbody>")
        lines.append("</table>")
        lines.append("")

    # Example
    lines.append("## Example")
    lines.append("")
    lines.append("```python")
    lines.append(import_line)
    lines.append("")
    # Generate a simple example call
    lines.append(f"result = {name}(...)")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

PROCESSING_CATEGORIES = {
    "ecgdatakit.processing.filters": "Filters",
    "ecgdatakit.processing.resample": "Resampling",
    "ecgdatakit.processing.normalize": "Normalization",
    "ecgdatakit.processing.peaks": "R-Peak Detection",
    "ecgdatakit.processing.hrv": "Heart Rate Variability",
    "ecgdatakit.processing.transforms": "Transforms & Segmentation",
    "ecgdatakit.processing.quality": "Signal Quality",
    "ecgdatakit.processing.leads": "Lead Derivation",
    "ecgdatakit.processing.clean": "ECG Cleaning",
}

PLOTTING_CATEGORIES = {
    "ecgdatakit.plotting.static": "Static Plots (matplotlib)",
    "ecgdatakit.plotting.interactive": "Interactive Plots (plotly)",
}


def _get_category(func) -> str:
    """Determine the category label for a function."""
    mod = _source_module(func)
    all_cats = {**PROCESSING_CATEGORIES, **PLOTTING_CATEGORIES}
    return all_cats.get(mod, mod)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Collect all functions
    functions: list[tuple[str, object, str]] = []  # (name, func, parent_module)

    for name in processing_mod.__all__:
        func = getattr(processing_mod, name)
        if callable(func):
            functions.append((name, func, "ecgdatakit.processing"))

    for name in plotting_mod.__all__:
        func = getattr(plotting_mod, name)
        if callable(func):
            functions.append((name, func, "ecgdatakit.plotting"))

    # Generate pages
    for weight, (name, func, parent_mod) in enumerate(functions, start=1):
        category = _get_category(func)
        content = _render_page(func, parent_mod, weight, category)
        out_path = OUTPUT_DIR / f"{name}.md"
        out_path.write_text(content)

    # Generate _index.md
    index_lines = [
        "---",
        'title: "Function Reference"',
        "weight: 40",
        "---",
        "",
        "Detailed reference for every public function in ECGDataKit.",
        "",
        "## Processing",
        "",
    ]

    # Group by category
    proc_by_cat: dict[str, list[str]] = {}
    plot_by_cat: dict[str, list[str]] = {}

    for name, func, parent_mod in functions:
        cat = _get_category(func)
        if parent_mod == "ecgdatakit.processing":
            proc_by_cat.setdefault(cat, []).append(name)
        else:
            plot_by_cat.setdefault(cat, []).append(name)

    for cat, names in proc_by_cat.items():
        index_lines.append(f"### {cat}")
        index_lines.append("")
        for n in names:
            index_lines.append(f"- [`{n}`](/api/ref/{n}/)")
        index_lines.append("")

    index_lines.append("## Plotting")
    index_lines.append("")

    for cat, names in plot_by_cat.items():
        index_lines.append(f"### {cat}")
        index_lines.append("")
        for n in names:
            index_lines.append(f"- [`{n}`](/api/ref/{n}/)")
        index_lines.append("")

    (OUTPUT_DIR / "_index.md").write_text("\n".join(index_lines))

    print(f"Generated {len(functions)} reference pages in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
