#!/usr/bin/env python3
"""Turn an executed Jupyter notebook into a clean, paginated PDF.

Usage:
    python scripts/export_pdf.py notebooks/california_housing_shallow_nn.ipynb [output.pdf]

Requirements: pip install playwright pygments markdown  &&  playwright install chromium
Optional: pandoc on the PATH (better tables and maths; falls back to python-markdown).
"""
import base64, html, json, re, shutil, subprocess, sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

CSS = """
@page { size: Letter; margin: 0.6in 0.6in 0.75in 0.6in; }
:root { --ink:#1E2B33; --muted:#5D6B73; --rule:#D8DEE3; --sign:#006747; --poppy:#E8772E; --code:#F5F7F8; }
html { font-size: 10pt; }
body { font-family: "Source Sans 3", "Segoe UI", "Helvetica Neue", Arial, "DejaVu Sans", sans-serif;
       color: var(--ink); line-height: 1.45; margin: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.md h1 { font-size: 21pt; line-height: 1.15; margin: 0 0 8pt; }
.md h2 { font-size: 14.5pt; margin: 18pt 0 6pt; padding-top: 7pt; border-top: 2.5px solid var(--sign); break-after: avoid; }
.md h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
.md p, .md li { orphans: 3; widows: 3; }
.md table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; break-inside: avoid; }
.md th, .md td { border-bottom: 1px solid var(--rule); padding: 3pt 7pt; text-align: left; vertical-align: top; }
.md th { border-bottom: 1.5px solid #8994A0; }
.md blockquote { margin: 8pt 0; padding: 6pt 10pt; border-left: 3px solid var(--poppy); background: #FFF6EE; }
.md code { font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 8.6pt; background: var(--code); padding: 0 2px; }
.md a { color: #1D4E89; text-decoration: none; }
.md math { font-size: 105%; }
.references p { padding-left: 0.4in; text-indent: -0.4in; margin: 0 0 5pt; font-size: 9.2pt; overflow-wrap: anywhere; }
.cell { margin: 5pt 0 8pt; }
.prompt { font: 7.5pt "DejaVu Sans Mono", monospace; color: var(--muted); margin: 0 0 1pt; }
.hl { background: var(--code); border: 1px solid #E3E8EC; border-radius: 3px; padding: 5pt 7pt; }
.hl pre, .outputs pre { margin: 0; font: 7.5pt/1.36 "DejaVu Sans Mono", Menlo, Consolas, monospace;
                        white-space: pre-wrap; overflow-wrap: anywhere; }
.outputs > * { margin-top: 4pt; }
.outputs pre { padding: 2pt 0 2pt 8pt; border-left: 2px solid var(--rule); }
.outputs pre.error { color: #A23B2A; border-left-color: #A23B2A; }
.figure { break-inside: avoid; text-align: left; }
.figure img { max-width: 100%; }
.html-out { break-inside: auto; }
table.dataframe { border-collapse: collapse; font-size: 7.6pt; margin: 2pt 0; }
table.dataframe th, table.dataframe td { padding: 2pt 6pt; border-bottom: 1px solid #E4E9EE; text-align: right; }
table.dataframe thead th { border-bottom: 1.5px solid #8994A0; vertical-align: bottom; }
table.dataframe tbody tr:nth-child(odd) { background: #FAFBFC; }
table.dataframe tr { break-inside: avoid; }
"""

FIT_TABLES = """
<script>
window.addEventListener('load', () => {
  const width = document.body.clientWidth;
  document.querySelectorAll('table').forEach(t => {
    const w = t.scrollWidth;
    if (w > width) { t.style.zoom = (width / w).toFixed(3); }
  });
});
</script>
"""


def text_of(x):
    return "".join(x) if isinstance(x, list) else (x or "")


def md_to_html(text):
    if shutil.which("pandoc"):
        r = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars+pipe_tables+raw_html-implicit_figures",
                            "-t", "html5", "--mathml"], input=text, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
    import markdown
    return markdown.markdown(text, extensions=["tables", "fenced_code", "md_in_html"])


def render_output(out):
    kind = out.get("output_type")
    if kind == "stream":
        return f'<pre class="stream">{html.escape(ANSI.sub("", text_of(out.get("text"))))}</pre>'
    if kind in ("display_data", "execute_result"):
        data = out.get("data", {})
        if "image/png" in data:
            png = text_of(data["image/png"]).replace("\n", "")
            return f'<div class="figure"><img src="data:image/png;base64,{png}"></div>'
        if "image/svg+xml" in data:
            return f'<div class="figure">{text_of(data["image/svg+xml"])}</div>'
        if "text/html" in data:
            return f'<div class="html-out">{text_of(data["text/html"])}</div>'
        if "text/markdown" in data:
            return f'<div class="md">{md_to_html(text_of(data["text/markdown"]))}</div>'
        if "text/plain" in data:
            return f'<pre class="plain">{html.escape(text_of(data["text/plain"]))}</pre>'
    if kind == "error":
        tb = ANSI.sub("", "\n".join(out.get("traceback", [])))
        return f'<pre class="error">{html.escape(tb)}</pre>'
    return ""


def notebook_html(nb):
    fmt = HtmlFormatter(style="friendly", cssclass="hl")
    parts = []
    for cell in nb["cells"]:
        src = text_of(cell.get("source"))
        if cell["cell_type"] == "markdown":
            parts.append(f'<section class="cell md">{md_to_html(src)}</section>')
        elif cell["cell_type"] == "code":
            n = cell.get("execution_count")
            outs = "".join(render_output(o) for o in cell.get("outputs", []))
            parts.append(f'<section class="cell code"><div class="prompt">In [{n if n is not None else " "}]:</div>'
                         f'{highlight(src, PythonLexer(), fmt)}<div class="outputs">{outs}</div></section>')
    style = CSS + fmt.get_style_defs(".hl") + " .hl { background: var(--code); }"
    title = html.escape(nb.get("metadata", {}).get("title", "Notebook"))
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{style}</style>'
            f'{FIT_TABLES}</head><body>{"".join(parts)}</body></html>')


def export(ipynb, pdf=None):
    ipynb = Path(ipynb)
    pdf = Path(pdf) if pdf else ipynb.with_suffix(".pdf")
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    html_path = pdf.with_suffix(".print.html")
    html_path.write_text(notebook_html(nb), encoding="utf-8")
    meta = nb.get("metadata", {})
    left = ", ".join(a.get("name", "") for a in meta.get("authors", []) if isinstance(a, dict))
    mid = meta.get("title", ipynb.stem)               # the footer comes from the notebook's own metadata
    footer = ('<div style="width:100%;font-size:7.5pt;color:#6B7780;font-family:\'DejaVu Sans\',Arial,sans-serif;'
              'padding:0 0.6in;display:flex;justify-content:space-between;">'
              f'<span>{html.escape(left)}</span><span>{html.escape(mid)}</span>'
              '<span>page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.wait_for_load_state("load")
        page.wait_for_timeout(300)
        page.pdf(path=str(pdf), format="Letter", print_background=True, display_header_footer=True,
                 header_template="<div></div>", footer_template=footer,
                 margin={"top": "0.6in", "bottom": "0.75in", "left": "0.6in", "right": "0.6in"})
        browser.close()
    html_path.unlink(missing_ok=True)
    print(f"wrote {pdf}")
    return pdf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
