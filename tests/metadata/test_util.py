from pyklopp.metadata import load_schema


def test_load_schema_latest_success():
    schema = load_schema()
    assert len(schema) > 1
