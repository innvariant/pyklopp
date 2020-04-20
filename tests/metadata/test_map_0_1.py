import pytest
import json

from pyklopp.metadata import MetadataMapV0V1, Metadata


def test_construct():
    MetadataMapV0V1({}, {})


def test_fill_empty_unstrict():
    empty_metadata = Metadata()
    result = {}

    map = MetadataMapV0V1({}, result)
    map.remap_all(empty_metadata)

    print(result)


def _test_construct_with_specific_version():
    reader = MetadataReader()
    reader.load(json.dumps({
        'schema_version': '0.1.0',
        'system': {
            'foo'
        }
    }))
    mapper = MetadataMapV0V1(reader)

    metadata = Metadata()
    mapper(metadata)



def _test_foo():
    m = MetadataMapV0V1(version="0.1.0")

    m.system_global_unique_id = "abc"
    assert m.system_global_unique_id == "abc"

