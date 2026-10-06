from housing_app import report


def test_report_is_generated_flags_fixtures_and_passes_its_own_check(bundle, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.START}\nPLACEHOLDER-TEXT\n{report.END}\noutro\n")
    touched = report.build(bundle, docs, readme)
    text = (docs / "RESULTS.md").read_text()
    assert bundle.fingerprint in text and "synthetic test fixtures" in text
    for heading in ("## Headline", "## Findings", "## Conclusion", "## Limitations", "## References"):
        assert heading in text
    assert "README.md" in touched and "intro" in readme.read_text() and "PLACEHOLDER-TEXT" not in readme.read_text()
    assert report.check(bundle, docs)
    assert len(touched["figures"]) >= 3 and all(p.stat().st_size < 120_000 for p in touched["figures"])
