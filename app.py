"""California housing: a shallow neural network, explained. Streamlit entry point:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from housing_app import __version__
from housing_app.artifacts import ArtifactError, load_bundle, signature
from housing_app.config import load_settings
from housing_app.service import PredictionService
from housing_app.storage import LocalStorage, RecordStore, StorageError, build_storage
from housing_app.theme import register_plotly_template
from housing_app.views import AppContext, pages
from housing_app.views.common import hero, inject_css

st.set_page_config(page_title="California housing: a shallow neural network", page_icon="🛣️", layout="wide")
inject_css()
register_plotly_template()

SETUP_HELP = """
**How to fix this**

1. Build and run the notebook: `python scripts/build_notebook.py`, then run `notebooks/california_housing_shallow_nn.ipynb`
   from the `notebooks/` folder. Its export step writes the artifacts into `artifacts/`.
2. Commit them (`git add artifacts && git commit -m "Add trained artifacts"`) and push; Streamlit Community Cloud
   redeploys automatically.
3. Or keep them in a bucket: set `artifacts_backend = "s3"` in the app's secrets and run `python -m housing_app publish`.
"""


def read_secrets() -> dict:
    try:
        return {key: st.secrets[key] for key in st.secrets}
    except Exception:                      # no secrets file locally, or none configured on the cloud
        return {}


SETTINGS = load_settings(read_secrets())


@st.cache_data(ttl=60, show_spinner=False)
def artifact_signature(backend: str, location: str) -> str:
    """Changes whenever the artifacts change, which invalidates the cached bundle."""
    return signature(build_storage(SETTINGS, "artifacts"))


@st.cache_resource(show_spinner="Loading and verifying the trained models…")
def get_service(backend: str, location: str, sig: str) -> PredictionService:
    return PredictionService(load_bundle(build_storage(SETTINGS, "artifacts")))


@st.cache_resource(show_spinner=False)
def get_user_storage(backend: str, location: str):
    try:
        storage = build_storage(SETTINGS, "user")
        storage.list("scores/")                         # fail fast on bad credentials or a missing bucket
        return storage, None
    except Exception as err:
        return LocalStorage(SETTINGS.user_dir), f"Cloud storage is unavailable ({err}). Saves go to this server's disk for now."


def location(backend: str, local_dir) -> str:
    return str(local_dir) if backend == "local" else f"{SETTINGS.bucket}/{SETTINGS.prefix}"


art_location = location(SETTINGS.artifacts_backend, SETTINGS.artifacts_dir)
try:
    service = get_service(SETTINGS.artifacts_backend, art_location, artifact_signature(SETTINGS.artifacts_backend, art_location))
except (ArtifactError, StorageError) as err:
    hero("California housing: a shallow neural network", "This app serves the artifacts that the research notebook exports.")
    st.error(f"The results could not be loaded: {err}")
    st.markdown(SETUP_HELP)
    st.stop()

bundle = service.bundle
user_storage, storage_warning = get_user_storage(SETTINGS.user_backend, location(SETTINGS.user_backend, SETTINGS.user_dir))
ctx = AppContext(bundle=bundle, service=service, settings=SETTINGS, scenarios=RecordStore(user_storage, "scenarios"),
                 scores=RecordStore(user_storage, "scores"), user_storage_label=user_storage.label,
                 user_storage_is_cloud=user_storage.is_cloud, storage_warning=storage_warning)

PAGES = pages()
with st.sidebar:
    st.markdown('<div class="brand">California housing</div><div class="brand-sub">A shallow neural network, explained</div>',
                unsafe_allow_html=True)
    choice = st.radio("Go to", list(PAGES), key="nav", label_visibility="collapsed")
    st.caption(("Saves go to cloud storage" if user_storage.is_cloud else "Saves go to this server's disk") +
               f": {user_storage.label}")
    if storage_warning:
        st.warning(storage_warning)
    st.caption(f"Artifacts {bundle.fingerprint} ({'checksums verified' if bundle.verified else 'no manifest'}). "
               f"App version {__version__}.")
PAGES[choice](ctx)
