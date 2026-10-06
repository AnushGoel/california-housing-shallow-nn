"""Storage layer: one small interface with two interchangeable backends.

LocalStorage keeps files in a folder; S3Storage talks to any S3-compatible bucket (AWS S3, Cloudflare R2,
Backblaze B2, MinIO). Pages never know which one they are using. RecordStore adds append-only JSON records
on top, so concurrent users of the deployed app never overwrite each other's saves.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .config import Settings

MISSING_CODES = {"NoSuchKey", "404", "NotFound"}


class StorageError(RuntimeError):
    """A backend could not complete a request (network, permissions, configuration)."""


class LocalStorage:
    is_cloud = False

    def __init__(self, root):
        self.root = Path(root)
        self.label = f"local folder {self.root}"

    def _path(self, key: str) -> Path:
        root = self.root.resolve()
        path = (root / key).resolve()
        if path != root and root not in path.parents:
            raise StorageError(f"key {key!r} points outside the storage folder")
        return path

    def read_bytes(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as err:
            raise KeyError(key) from err

    def write_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)                       # atomic rename: readers never see a half-written file

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def version(self, key: str) -> str:
        stat = self._path(key).stat()
        return f"{stat.st_mtime_ns}-{stat.st_size}"

    def list(self, prefix: str = "") -> list[str]:
        if not self.root.exists():
            return []
        keys = []
        for path in self.root.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp"):
                key = path.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)


class S3Storage:
    is_cloud = True

    def __init__(self, bucket: str, prefix: str = "", client=None, **client_kwargs):
        if client is None:
            try:
                import boto3
            except ImportError as err:
                raise StorageError("S3 storage needs boto3 (pip install boto3)") from err
            client = boto3.client("s3", **{k: v for k, v in client_kwargs.items() if v})
        self.client, self.bucket, self.prefix = client, bucket, prefix.strip("/")
        self.label = f"s3://{bucket}/{self.prefix}" if self.prefix else f"s3://{bucket}"

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    @staticmethod
    def _code(err) -> str:
        return str(getattr(err, "response", {}).get("Error", {}).get("Code", ""))

    def read_bytes(self, key: str) -> bytes:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except Exception as err:
            if self._code(err) in MISSING_CODES:
                raise KeyError(key) from err
            raise StorageError(f"could not read {key} from {self.label}: {err}") from err

    def write_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        try:
            self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, ContentType=content_type)
        except Exception as err:
            raise StorageError(f"could not write {key} to {self.label}: {err}") from err

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception as err:
            if self._code(err) in MISSING_CODES:
                return False
            raise StorageError(f"could not check {key} in {self.label}: {err}") from err

    def version(self, key: str) -> str:
        try:
            return str(self.client.head_object(Bucket=self.bucket, Key=self._key(key)).get("ETag", ""))
        except Exception as err:
            raise StorageError(f"could not read the version of {key}: {err}") from err

    def list(self, prefix: str = "") -> list[str]:
        keys, token = [], None
        strip = len(self.prefix) + 1 if self.prefix else 0
        try:
            while True:
                kwargs = {"Bucket": self.bucket, "Prefix": self._key(prefix)}
                if token:
                    kwargs["ContinuationToken"] = token
                page = self.client.list_objects_v2(**kwargs)
                keys += [obj["Key"][strip:] for obj in page.get("Contents", [])]
                if not page.get("IsTruncated"):
                    break
                token = page.get("NextContinuationToken")
        except Exception as err:
            raise StorageError(f"could not list {self.label}: {err}") from err
        return sorted(keys)


class RecordStore:
    """Append-only JSON records, one object per record, newest first when read back."""

    def __init__(self, storage, collection: str):
        self.storage, self.collection = storage, collection.strip("/")

    def add(self, record: dict) -> str:
        now = time.gmtime()
        record_id = f"{time.strftime('%Y%m%dT%H%M%SZ', now)}_{time.time_ns() % 10**9:09d}_{uuid.uuid4().hex[:6]}"
        payload = {"id": record_id, "saved_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", now), **record}
        self.storage.write_bytes(f"{self.collection}/{record_id}.json",
                                 json.dumps(payload, default=str).encode("utf-8"), "application/json")
        return record_id

    def recent(self, limit: int = 50) -> list[dict]:
        keys = [k for k in self.storage.list(self.collection + "/") if k.endswith(".json")]
        records = []
        for key in sorted(keys, reverse=True)[:limit]:
            try:
                records.append(json.loads(self.storage.read_bytes(key)))
            except (KeyError, ValueError, StorageError):
                continue                        # a damaged record must not break the page
        return records


def build_storage(settings: Settings, area: str):
    """Factory used by the app and scripts: area is 'artifacts' or 'user'."""
    if area not in ("artifacts", "user"):
        raise ValueError(f"unknown storage area {area!r}")
    kind = settings.artifacts_backend if area == "artifacts" else settings.user_backend
    if kind == "s3":
        if not settings.bucket:
            raise StorageError(f"{area} storage is set to s3 but no bucket is configured")
        return S3Storage(settings.bucket, f"{settings.prefix}/{area}", endpoint_url=settings.endpoint_url,
                         region_name=settings.region, aws_access_key_id=settings.access_key_id,
                         aws_secret_access_key=settings.secret_access_key)
    return LocalStorage(settings.artifacts_dir if area == "artifacts" else settings.user_dir)
