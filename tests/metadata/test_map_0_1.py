import pytest
import json

from pyklopp.metadata import MetadataMapV0V1, Metadata, validate_schema


def test_construct():
    MetadataMapV0V1({}, {})


def test_fill_empty_unstrict():
    empty_metadata = Metadata.build_fill_default()
    empty_metadata.schema_version = '0.3.1'
    result = {}

    map = MetadataMapV0V1({}, result)
    map.remap_all(empty_metadata)

    validate_schema(result, version='0.1.0')

