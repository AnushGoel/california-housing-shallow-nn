"""How it works: system architecture, design decisions, live integrity status and sources."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import __version__
from .common import page_title, table

ARCHITECTURE = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.5;
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#9AA6AE", fontname="Helvetica", fontsize=11];
  edge [color="#5D6B73", fontname="Helvetica", fontsize=9];
  subgraph cluster_offline { label="Offline: research (Jupyter or Colab)"; color="#C9D1D9"; fontname="Helvetica";
    src [label="notebook sources\\n(plain text)"]; nb [label="notebook\\nexperiments + tests"]; src -> nb;
    art [label="artifacts/\\ngzip + SHA-256 manifest", fillcolor="#E8F3EC"]; nb -> art; }
  subgraph cluster_store { label="Storage"; color="#C9D1D9"; fontname="Helvetica";
    repo [label="GitHub repository"]; bucket [label="S3-compatible bucket\\n(optional)"]; }
  subgraph cluster_core { label="housing_app core (NumPy only)"; color="#C9D1D9"; fontname="Helvetica";
    store [label="storage layer\\nLocal | S3"]; contract [label="artifact contract\\nchecksums + schema"];
    service [label="PredictionService\\npredict, intervals,\\nShapley, drift", fillcolor="#E8F3EC"]; }
  subgraph cluster_faces { label="Interfaces"; color="#C9D1D9"; fontname="Helvetica";
    ui [label="Streamlit app\\n10 pages"]; api [label="REST API\\n(FastAPI)"]; cli [label="command line"];
    report [label="evidence report\\ndocs/RESULTS.md"]; }
  ci [label="GitHub Actions\\nlint, tests, size budget,\\nreport freshness", fillcolor="#FFF6EE"];
  art -> repo [label="git push"]; art -> bucket [label="publish", style=dashed];
  repo -> store; bucket -> store [style=dashed]; store -> contract -> service;
  service -> ui; service -> api; service -> cli; contract -> report;
  ui -> store [label="scenarios, scores", style=dashed]; ci -> repo [label="every push"];
}
"""

DESIGN = """
**Research and serving are separate.** The notebook trains with TensorFlow and exports plain NumPy weights. Everything
that serves the model, from the app to the API and the command line, goes through one `PredictionService` that needs
only NumPy, so the deployment is small, starts in seconds and cannot drift from what was trained.

**Artifacts are a checked contract.** The notebook writes gzip-compressed files plus a manifest of SHA-256 checksums.
The loader verifies every checksum and every required field and stops with a readable message if anything was altered,
truncated or is missing. A short fingerprint of the checksums is printed in the app, the report and the API.

**Uncertainty and explanation come with every prediction.** Split-conformal intervals turn validation residuals into a
distribution-free band around each price, and an exact Shapley breakdown explains it against a typical block group.

**Inputs are checked before they are trusted.** Batch scoring validates each row, flags rows that sit outside the
training distribution and measures drift for the batch as a whole.

**One storage interface, two backends.** The same code reads and writes a local folder or an S3-compatible bucket;
every saved scenario or score is its own append-only object, so simultaneous visitors never overwrite each other. If
the bucket is unreachable, the app falls back to local disk and says so.

**Light by design.** Compressed artifacts, stripped notebook outputs, small SVG figures and a size budget enforced in
continuous integration keep the repository small enough to clone in seconds.
"""

REFERENCES = """
Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association, 113*(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774). Curran Associates.

Mahalanobis, P. C. (1936). On the generalised distance in statistics. *Proceedings of the National Institute of Sciences of India, 2*(1), 49–55.

National Institute of Standards and Technology. (2015). *Secure hash standard (SHS)* (FIPS PUB 180-4). U.S. Department of Commerce. https://doi.org/10.6028/NIST.FIPS.180-4

Pace, R. K., & Barry, R. (1997). Sparse spatial autoregressions. *Statistics & Probability Letters, 33*(3), 291–297. https://doi.org/10.1016/S0167-7152(96)00140-X

Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W. Tucker (Eds.), *Contributions to the theory of games* (Vol. 2, pp. 307–317). Princeton University Press.

Siddiqi, N. (2006). *Credit risk scorecards: Developing and implementing intelligent credit scoring*. Wiley.

Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer. https://doi.org/10.1007/b106715

The full reference list for the study is in `docs/REFERENCES.md`.
"""


def render(ctx) -> None:
    b = ctx.bundle
    page_title("How it works", "Research happens offline in the notebook; this app only serves its results. The diagram "
               "shows how the pieces fit together, and the table below is a live check of the deployment.")
    st.graphviz_chart(ARCHITECTURE)
    st.markdown("### Design decisions")
    st.markdown(DESIGN)
    st.markdown("### Live status")
    versions = b.meta.get("versions", {})
    table(pd.DataFrame([
        {"component": "Results and model weights", "status": b.source},
        {"component": "Integrity", "status": f"fingerprint {b.fingerprint}; checksums "
                                            + ("verified against the manifest" if b.verified else "not listed in a manifest")},
        {"component": "NumPy against Keras",
         "status": "; ".join(f"{label}: largest difference {net.self_check():.1e}" for label, net in b.models.items())},
        {"component": "Saved scenarios and scores",
         "status": ctx.user_storage_label + (" (cloud)" if ctx.user_storage_is_cloud else " (this server's disk)")},
        {"component": "Training environment",
         "status": f"TensorFlow {versions.get('tensorflow', '?')}, Keras {versions.get('keras', '?')}, run {b.meta.get('generated', '?')}"},
        {"component": "App version", "status": __version__}]))
    if ctx.storage_warning:
        st.warning(ctx.storage_warning)
    st.markdown("### Methods and evidence")
    st.markdown("The study design, the testing protocol and the evidence behind every claim are written up in the "
                "repository's `docs/` folder: `METHODOLOGY.md`, `EVALUATION.md`, `ARCHITECTURE.md`, `MODEL_CARD.md`, "
                "`DATASHEET.md` and the generated `RESULTS.md`.")
    st.markdown("### Key references")
    st.markdown(REFERENCES)
