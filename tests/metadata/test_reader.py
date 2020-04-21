import pytest
import json

from pyklopp.metadata import MetadataReader, MetadataError


def test_construct():
    MetadataReader()

def test_construct_with_specific_version():
    MetadataReader(version="0.1.0")


def test_attribute_unravelling():
    # Arrange
    original_dict = {'foo': 10, 'sub': {'attr': 8}}
    m = MetadataReader(schema="{}")

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
    m = MetadataReader()

    with pytest.raises(MetadataError):
        m.load("""{
  "foo": 1
}""")


def test_metadata_load_fail_on_empty_metadata():
    m = MetadataReader()

    with pytest.raises(MetadataError):
        m.load("{}")


def test_metadata_load_fail_on_list():
    m = MetadataReader()

    with pytest.raises(MetadataError):
        m.load("""["a", "b", "c"]""")
