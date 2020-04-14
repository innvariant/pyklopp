import pytest
import json

from pyklopp.metadata import *


def test_load_schema_latest_success():
    schema = load_schema()
    assert len(schema) > 1


def test_construct():
    Metadata()


def test_construct_with_specific_version():
    Metadata(version="0.1.0")


def test_attribute_unravelling():
    # Arrange
    original_dict = {'foo': 10, 'sub': {'attr': 8}}
    m = Metadata(schema="{}")

    # Act
    m.load(json.dumps(original_dict))

    # Assert
    assert m.foo == original_dict['foo']
    assert m.sub.attr == original_dict['sub']['attr']
    assert m['foo'] == original_dict['foo']
    assert m['sub']['attr'] == original_dict['sub']['attr']
    with pytest.raises(MetadataError):
        print(m['unknown_key'])
    with pytest.raises(MetadataError):
        print(m.unknown_attr)


def test_metadata_load_fail_on_any_dict():
    m = Metadata()

    with pytest.raises(MetadataError):
        m.load("""{
  "foo": 1
}""")


def test_metadata_load_fail_on_empty_metadata():
    m = Metadata()

    with pytest.raises(MetadataError):
        m.load("{}")


def test_metadata_load_fail_on_list():
    m = Metadata()

    with pytest.raises(MetadataError):
        m.load("""["a", "b", "c"]""")
