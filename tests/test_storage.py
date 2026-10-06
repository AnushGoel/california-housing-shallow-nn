import io

import pytest

from housing_app.config import load_settings
from housing_app.storage import LocalStorage, RecordStore, S3Storage, StorageError, build_storage


class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """In-memory stand-in for a boto3 S3 client, with paging, so no network or credentials are needed."""

    def __init__(self, page_size=2):
        self.objects, self.page_size = {}, page_size

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {"ETag": str(len(self.objects[(Bucket, Key)]))}

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = {"Contents": [{"Key": k} for k in keys[start:start + self.page_size]],
                "IsTruncated": start + self.page_size < len(keys)}
        if page["IsTruncated"]:
            page["NextContinuationToken"] = str(start + self.page_size)
        return page


def test_local_storage_round_trip(tmp_path):
    s = LocalStorage(tmp_path)
    s.write_bytes("a/b.json", b"{}")
    assert s.exists("a/b.json") and s.read_bytes("a/b.json") == b"{}"
    assert s.list("a/") == ["a/b.json"] and not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(KeyError):
        s.read_bytes("missing.json")


def test_local_storage_refuses_paths_outside_its_folder(tmp_path):
    with pytest.raises(StorageError):
        LocalStorage(tmp_path / "root").write_bytes("../escape.txt", b"x")


def test_s3_storage_round_trip_with_prefix_and_paging():
    client = FakeS3(page_size=2)
    s = S3Storage("bucket", "app/user", client=client)
    for i in range(5):
        s.write_bytes(f"scores/{i}.json", b"1")
    assert ("bucket", "app/user/scores/0.json") in client.objects
    assert s.list("scores/") == [f"scores/{i}.json" for i in range(5)]
    assert s.exists("scores/3.json") and not s.exists("scores/9.json")
    with pytest.raises(KeyError):
        s.read_bytes("scores/9.json")


def test_record_store_is_append_only_and_newest_first(tmp_path):
    for storage in (LocalStorage(tmp_path), S3Storage("b", "p", client=FakeS3())):
        store = RecordStore(storage, "scenarios")
        ids = [store.add({"n": i}) for i in range(3)]
        assert len(set(ids)) == 3
        assert [r["n"] for r in store.recent()] == [2, 1, 0]
        storage.write_bytes("scenarios/zzz_corrupt.json", b"not json")
        assert len(store.recent()) == 3                     # damaged records are skipped, not fatal


def test_settings_environment_overrides_secrets(tmp_path):
    secrets = {"storage": {"user_backend": "s3", "bucket": "from-secrets", "prefix": "/p/"}}
    s = load_settings(secrets, env={"HOUSING_BUCKET": "from-env", "ARTIFACTS_DIR": str(tmp_path)})
    assert s.user_backend == "s3" and s.bucket == "from-env" and s.prefix == "p" and s.artifacts_dir == tmp_path
    assert "secret" not in repr(load_settings({"storage": {"aws_secret_access_key": "secret"}}, env={}))


def test_s3_backend_without_bucket_fails_clearly():
    with pytest.raises(StorageError, match="no bucket"):
        build_storage(load_settings({"storage": {"user_backend": "s3"}}, env={}), "user")
    with pytest.raises(ValueError):
        load_settings({"storage": {"user_backend": "dropbox"}}, env={})
