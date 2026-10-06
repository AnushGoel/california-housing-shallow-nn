"""Runtime settings. Environment variables (HOUSING_*) win over the [storage] table in Streamlit secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

APP_ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("local", "s3")


@dataclass(frozen=True)
class Settings:
    artifacts_backend: str = "local"        # where the notebook's results are read from
    user_backend: str = "local"             # where saved scenarios and game scores are written
    artifacts_dir: Path = APP_ROOT / "artifacts"
    user_dir: Path = APP_ROOT / "user_data"
    bucket: Optional[str] = None            # any S3-compatible bucket: AWS S3, Cloudflare R2, Backblaze B2, MinIO
    prefix: str = "california-housing-nn"
    endpoint_url: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = field(default=None, repr=False)
    secret_access_key: Optional[str] = field(default=None, repr=False)

    @property
    def cloud_configured(self) -> bool:
        return bool(self.bucket)


def load_settings(secrets: Optional[Mapping] = None, env: Optional[Mapping] = None) -> Settings:
    env = os.environ if env is None else env
    table = dict((secrets or {}).get("storage", {}) or {})

    def pick(key, default=None):
        value = env.get(f"HOUSING_{key.upper()}")
        if value in (None, ""):
            value = table.get(key)
        return default if value in (None, "") else value

    def backend(key):
        value = str(pick(key, "local")).lower()
        if value not in BACKENDS:
            raise ValueError(f"{key} must be one of {BACKENDS}, got {value!r}")
        return value

    artifacts_dir = env.get("ARTIFACTS_DIR") or pick("artifacts_dir")
    return Settings(
        artifacts_backend=backend("artifacts_backend"),
        user_backend=backend("user_backend"),
        artifacts_dir=Path(artifacts_dir) if artifacts_dir else APP_ROOT / "artifacts",
        user_dir=Path(pick("user_dir", APP_ROOT / "user_data")),
        bucket=pick("bucket"),
        prefix=str(pick("prefix", "california-housing-nn")).strip("/"),
        endpoint_url=pick("endpoint_url"),
        region=pick("region"),
        access_key_id=pick("aws_access_key_id"),
        secret_access_key=pick("aws_secret_access_key"),
    )
