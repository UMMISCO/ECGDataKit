"""Sphinx configuration for ECGDataKit documentation."""

import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the library importable for autodoc
# ---------------------------------------------------------------------------
LIB_SRC = Path(__file__).resolve().parent.parent / "lib" / "src"
sys.path.insert(0, str(LIB_SRC))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "ECGDataKit"
copyright = "2026, Ahmad Fall — UMMISCO / IRD"
author = "Ahmad Fall"
release = "0.0.8"
version = "0.0.8"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_sitemap",
    "sphinxext.opengraph",
]

# ---------------------------------------------------------------------------
# MyST-Parser settings
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "tasklist",
]
myst_heading_anchors = 3
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ---------------------------------------------------------------------------
# Autodoc settings
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_mock_imports = [
    "scipy",
    "matplotlib",
    "plotly",
    "torch",
    "biosppy",
    "neurokit2",
]

# ---------------------------------------------------------------------------
# Napoleon settings (numpy-style docstrings)
# ---------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# ---------------------------------------------------------------------------
# Intersphinx mappings
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# ---------------------------------------------------------------------------
# Furo theme configuration
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "ECGDataKit"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]
pygments_dark_style = "monokai"
html_favicon = "_static/logo.svg"
html_theme_options = {
    "light_logo": "logo.svg",
    "dark_logo": "logo.svg",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/UMMISCO/ECGDataKit",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
    "source_repository": "https://github.com/UMMISCO/ECGDataKit",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#059669",
        "color-brand-content": "#059669",
        "color-admonition-background": "#ecfdf5",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6ee7b7",
        "color-brand-content": "#6ee7b7",
    },
}

# ---------------------------------------------------------------------------
# Copy button settings
# ---------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# ---------------------------------------------------------------------------
# Sitemap
# ---------------------------------------------------------------------------
html_baseurl = "https://ecgdatakit.ummisco.fr/"
sitemap_url_scheme = "{link}"

# ---------------------------------------------------------------------------
# Open Graph / social sharing
# ---------------------------------------------------------------------------
ogp_site_url = "https://ecgdatakit.ummisco.fr/"
ogp_site_name = "ECGDataKit"
ogp_description_length = 200
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary" />',
]

# ---------------------------------------------------------------------------
# Suppress specific warnings
# ---------------------------------------------------------------------------
suppress_warnings = ["myst.header"]


# ---------------------------------------------------------------------------
# Auto-generated Activity page (git commit timeline)
# ---------------------------------------------------------------------------
import re


def _esc(s):
    """HTML-escape a string."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _generate_activity_page(app):
    """Generate activity.md from git log at build time."""
    src_dir = Path(app.srcdir)
    out_path = src_dir / "activity.md"
    fallback = "# Activity\n\nNo activity data available.\n"
    repo_dir = src_dir.parent  # project root

    # ── Collect commit data ──────────────────────────────────────────────
    separator = "||"
    fmt = f"%h{separator}%ad{separator}%an{separator}%s{separator}%D"
    try:
        result = subprocess.run(
            [
                "git", "log", f"--format={fmt}", "--date=format:%Y-%m-%d",
                "--shortstat", "-200",
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            timeout=10,
        )
        if result.returncode != 0:
            out_path.write_text(fallback, encoding="utf-8")
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        out_path.write_text(fallback, encoding="utf-8")
        return

    # Get latest tag (release name)
    latest_release = ""
    try:
        tag_result = subprocess.run(
            ["git", "tag", "--sort=-creatordate"],
            capture_output=True, text=True, cwd=str(repo_dir), timeout=5,
        )
        tags = tag_result.stdout.strip().splitlines()
        if tags:
            latest_release = tags[0].strip()
    except Exception:
        pass

    # Parse commits
    raw_lines = result.stdout.strip().splitlines()
    all_commits = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        i += 1
        if not line:
            continue
        parts = line.split(separator, 4)
        if len(parts) < 4:
            continue
        short_hash, date_str, author, message = (p.strip() for p in parts[:4])
        refs = parts[4].strip() if len(parts) > 4 else ""
        tag = ""
        if refs:
            tag_match = re.search(r"tag:\s*([^,\s]+)", refs)
            if tag_match:
                tag = tag_match.group(1)
        message = message.lstrip("- ").strip()
        if not message:
            continue

        # Next non-empty line is --shortstat
        stat_line = ""
        while i < len(raw_lines):
            candidate = raw_lines[i].strip()
            i += 1
            if candidate:
                stat_line = candidate
                break

        changes = ""
        if stat_line:
            m_files = re.search(r"(\d+) files? changed", stat_line)
            m_ins = re.search(r"(\d+) insertions?", stat_line)
            m_del = re.search(r"(\d+) deletions?", stat_line)
            parts_c = []
            if m_files:
                parts_c.append(f"{m_files.group(1)} files")
            if m_ins:
                parts_c.append(f"+{m_ins.group(1)}")
            if m_del:
                parts_c.append(f"-{m_del.group(1)}")
            changes = " ".join(parts_c)

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        all_commits.append({
            "hash": short_hash, "date": date_str, "dt": dt,
            "author": author, "message": message, "changes": changes,
            "tag": tag,
        })

    if not all_commits:
        out_path.write_text(fallback, encoding="utf-8")
        return

    # ── Build contribution heatmap data (from first commit) ─────────────
    from datetime import timedelta

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Start from the 1st of the earliest commit's month, aligned to Sunday
    earliest = min(c["dt"] for c in all_commits)
    first_of_month = earliest.replace(day=1)
    days_since_sunday = (first_of_month.weekday() + 1) % 7  # Sun→0 … Sat→6
    start = first_of_month - timedelta(days=days_since_sunday)

    # Count commits per date
    date_counts = defaultdict(int)
    for c in all_commits:
        date_counts[c["date"]] += 1

    # Build weeks as columns: each week = list of 7 cells [Sun … Sat].
    # The last (current) week may be partial — pad with None so every
    # column has exactly 7 entries for a correct CSS grid.
    heatmap_weeks = []
    current = start
    week = []
    while current <= today:
        ds = current.strftime("%Y-%m-%d")
        count = date_counts.get(ds, 0)
        week.append({"date": ds, "count": count})
        if len(week) == 7:
            heatmap_weeks.append(week)
            week = []
        current += timedelta(days=1)
    if week:
        # Pad the last partial week so the grid stays aligned
        while len(week) < 7:
            week.append(None)
        heatmap_weeks.append(week)

    # Intensity levels: 0, 1-2, 3-5, 6-9, 10+
    def _level(n):
        if n == 0:
            return 0
        if n <= 2:
            return 1
        if n <= 5:
            return 2
        if n <= 9:
            return 3
        return 4

    # Month labels positioned by week index
    month_labels = []
    prev_month = None
    for wi, week_data in enumerate(heatmap_weeks):
        d = datetime.strptime(week_data[0]["date"], "%Y-%m-%d")
        m = d.month
        if m != prev_month:
            month_labels.append((wi, d.strftime("%b")))
            prev_month = m

    # ── Stats ────────────────────────────────────────────────────────────
    total = len(all_commits)
    last_date = all_commits[0]["date"]
    authors = sorted(set(c["author"] for c in all_commits))
    active_days = len(set(c["date"] for c in all_commits))

    # ── Build HTML ───────────────────────────────────────────────────────
    lines = [
        "# Activity",
        "",
        "````{raw} html",
    ]

    # Dashboard section
    lines.append('<div class="activity-dashboard">')

    # Stats cards
    lines.append('  <div class="activity-stats">')
    lines.append('    <div class="activity-stat-card">')
    lines.append(f'      <div class="activity-stat-value">{total}</div>')
    lines.append('      <div class="activity-stat-label">Commits</div>')
    lines.append("    </div>")
    lines.append('    <div class="activity-stat-card">')
    lines.append(f'      <div class="activity-stat-value">{active_days}</div>')
    lines.append('      <div class="activity-stat-label">Active days</div>')
    lines.append("    </div>")
    lines.append('    <div class="activity-stat-card">')
    lines.append(f'      <div class="activity-stat-value">{len(authors)}</div>')
    lines.append('      <div class="activity-stat-label">Contributors</div>')
    lines.append("    </div>")
    lines.append('    <div class="activity-stat-card">')
    lines.append(f'      <div class="activity-stat-value">{last_date}</div>')
    lines.append('      <div class="activity-stat-label">Last commit</div>')
    lines.append("    </div>")
    if latest_release:
        lines.append('    <div class="activity-stat-card">')
        lines.append(f'      <div class="activity-stat-value">{_esc(latest_release)}</div>')
        lines.append('      <div class="activity-stat-label">Latest release</div>')
        lines.append("    </div>")
    else:
        lines.append('    <div class="activity-stat-card">')
        lines.append(f'      <div class="activity-stat-value">v{release}</div>')
        lines.append('      <div class="activity-stat-label">Latest version</div>')
        lines.append("    </div>")
    lines.append("  </div>")

    # Contribution heatmap — rendered as SVG (like GitHub)
    cell = 13
    gap = 3
    step = cell + gap
    label_w = 32
    n_weeks = len(heatmap_weeks)
    svg_w = label_w + n_weeks * step
    month_h = 18
    svg_h = month_h + 7 * step

    # Color palette (matches GitHub)
    colors_light = ["#ebedf0", "#9be9a8", "#40c463", "#30a14e", "#216e39"]
    colors_dark = ["#161b22", "#0e4429", "#006d32", "#26a641", "#39d353"]

    lines.append('  <div class="activity-heatmap-wrap">')
    lines.append('    <div class="activity-heatmap-title">Contributions</div>')
    lines.append('    <div class="activity-heatmap-scroll">')

    # Light-mode SVG
    lines.append(f'    <svg class="activity-heatmap-svg light" width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">')

    # Month labels
    prev_month_col = -4
    for wi, label in month_labels:
        x = label_w + wi * step
        if wi - prev_month_col >= 4:
            lines.append(f'      <text x="{x}" y="12" class="activity-svg-month">{label}</text>')
            prev_month_col = wi

    # Day-of-week labels
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    for dow in (1, 3, 5):
        y = month_h + dow * step + cell - 2
        lines.append(f'      <text x="0" y="{y}" class="activity-svg-day">{day_names[dow]}</text>')

    # Cells
    for wi, week_data in enumerate(heatmap_weeks):
        x = label_w + wi * step
        for dow in range(7):
            c = week_data[dow]
            y = month_h + dow * step
            if c is None:
                continue
            lvl = _level(c["count"])
            fill = colors_light[lvl]
            tip = f'{c["date"]}: {c["count"]} commits' if c["count"] > 0 else f'{c["date"]}: No commits'
            lines.append(f'      <rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" fill="{fill}"><title>{tip}</title></rect>')

    lines.append("    </svg>")

    # Dark-mode SVG (identical structure, different fills)
    lines.append(f'    <svg class="activity-heatmap-svg dark" width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">')

    prev_month_col = -4
    for wi, label in month_labels:
        x = label_w + wi * step
        if wi - prev_month_col >= 4:
            lines.append(f'      <text x="{x}" y="12" class="activity-svg-month">{label}</text>')
            prev_month_col = wi

    for dow in (1, 3, 5):
        y = month_h + dow * step + cell - 2
        lines.append(f'      <text x="0" y="{y}" class="activity-svg-day">{day_names[dow]}</text>')

    for wi, week_data in enumerate(heatmap_weeks):
        x = label_w + wi * step
        for dow in range(7):
            c = week_data[dow]
            y = month_h + dow * step
            if c is None:
                continue
            lvl = _level(c["count"])
            fill = colors_dark[lvl]
            tip = f'{c["date"]}: {c["count"]} commits' if c["count"] > 0 else f'{c["date"]}: No commits'
            lines.append(f'      <rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" fill="{fill}"><title>{tip}</title></rect>')

    lines.append("    </svg>")

    # Legend
    lines.append('    <div class="activity-heatmap-legend">')
    lines.append('      <span>Less</span>')
    for lvl in range(5):
        lines.append(f'      <div class="activity-heatmap-legend-cell level-{lvl}"></div>')
    lines.append('      <span>More</span>')
    lines.append("    </div>")

    lines.append("    </div>")
    lines.append("  </div>")

    # Filter buttons
    lines.append('  <div class="activity-filters">')
    lines.append('    <span class="activity-filters-label">Show:</span>')
    lines.append('    <button class="activity-filter-btn active" data-days="0">All</button>')
    lines.append('    <button class="activity-filter-btn" data-days="365">Last year</button>')
    lines.append('    <button class="activity-filter-btn" data-days="90">Last 3 months</button>')
    lines.append('    <button class="activity-filter-btn" data-days="30">Last 30 days</button>')
    lines.append("  </div>")

    lines.append("</div>")  # end dashboard

    # ── Timeline section ─────────────────────────────────────────────────
    lines.append('<div class="activity-timeline">')

    # Group by year → day
    commits_by_year_day = defaultdict(lambda: defaultdict(list))
    for c in all_commits:
        commits_by_year_day[c["dt"].year][c["date"]].append(c)

    def _format_day(date_str):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%b %-d")

    for year in sorted(commits_by_year_day.keys(), reverse=True):
        days = commits_by_year_day[year]
        lines.append('  <div class="activity-year-group">')
        lines.append('    <div class="activity-year-header">')
        lines.append(f'      <span class="activity-year-badge">{year}</span>')
        lines.append("    </div>")

        for day in sorted(days.keys(), reverse=True):
            day_commits = days[day]
            day_label = _format_day(day)
            count = len(day_commits)
            count_label = f" &middot; {count} commits" if count > 1 else ""

            # Collect all release tags for this day
            day_tags = [c["tag"] for c in day_commits if c.get("tag")]

            lines.append(f'    <div class="activity-day-group" data-date="{day}">')

            # Release markers (above day header)
            tag_svg = (
                '<svg class="activity-release-icon" width="14" height="14" '
                'viewBox="0 0 16 16" fill="currentColor">'
                '<path d="M1 7.775V2.75C1 1.784 1.784 1 2.75 1h5.025c.464 0 '
                ".909.184 1.236.513l6.25 6.25a1.75 1.75 0 0 1 0 2.474l-5.026 "
                "5.026a1.75 1.75 0 0 1-2.474 0l-6.25-6.25A1.752 1.752 0 0 1 "
                '1 7.775ZM6 5a1 1 0 1 0 0 2 1 1 0 0 0 0-2Z"></path></svg>'
            )
            for day_tag in day_tags:
                tag_slug = day_tag.replace(".", "-")
                tag_href = f"releases.html#{tag_slug}"
                lines.append('      <div class="activity-release">')
                lines.append('        <div class="activity-dot activity-dot-release"></div>')
                lines.append(f'        <a class="activity-release-content" href="{tag_href}">')
                lines.append(f"          {tag_svg}")
                lines.append(
                    f'          <span class="activity-release-name">{_esc(day_tag)}</span>'
                )
                lines.append(
                    f'          <span class="activity-release-date">released {day_label}</span>'
                )
                lines.append("        </a>")
                lines.append("      </div>")

            lines.append('      <div class="activity-day-header">')
            lines.append('        <div class="activity-dot activity-dot-day"></div>')
            lines.append(
                f'        <span class="activity-day-label">{day_label}{count_label}</span>'
            )
            lines.append("      </div>")

            # Scrollable commit list
            lines.append('      <div class="activity-day-commits">')

            for commit in day_commits:
                msg = _esc(commit["message"])
                author = _esc(commit["author"])
                ch = commit["changes"]

                lines.append('        <div class="activity-entry">')
                lines.append('          <div class="activity-dot"></div>')
                lines.append('          <div class="activity-content">')
                lines.append(f'            <span class="activity-message">{msg}</span>')
                lines.append('            <div class="activity-meta">')
                commit_url = f"https://github.com/UMMISCO/ECGDataKit/commit/{commit['hash']}"
                lines.append(f'              <a class="activity-hash" href="{commit_url}" target="_blank" rel="noopener">{commit["hash"]}</a>')
                lines.append(f'              <span class="activity-author">{author}</span>')
                if ch:
                    lines.append(f'              <span class="activity-changes">{ch}</span>')
                lines.append("            </div>")
                lines.append("          </div>")
                lines.append("        </div>")

            lines.append("      </div>")  # end day-commits
            lines.append("    </div>")

        lines.append("  </div>")

    lines.append("</div>")

    # ── Filter JavaScript ────────────────────────────────────────────────
    lines.append("<script>")
    lines.append("""
(function() {
  var buttons = document.querySelectorAll('.activity-filter-btn');
  buttons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      buttons.forEach(function(b) { b.classList.remove('active'); });
      btn.classList.add('active');
      var days = parseInt(btn.getAttribute('data-days'), 10);
      var cutoff = '';
      if (days > 0) {
        var d = new Date();
        d.setDate(d.getDate() - days);
        cutoff = d.toISOString().slice(0, 10);
      }
      document.querySelectorAll('.activity-day-group').forEach(function(el) {
        var date = el.getAttribute('data-date');
        el.style.display = (!cutoff || date >= cutoff) ? '' : 'none';
      });
      // Hide year headers with no visible days
      document.querySelectorAll('.activity-year-group').forEach(function(yg) {
        var visible = yg.querySelectorAll('.activity-day-group:not([style*="display: none"])');
        yg.style.display = visible.length > 0 ? '' : 'none';
      });
    });
  });
})();
""".strip())
    lines.append("</script>")

    lines.append("````")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def setup(app):
    app.connect("builder-inited", _generate_activity_page)
