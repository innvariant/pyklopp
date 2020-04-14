import pytest
import json

from pyklopp.metadata import *


def test_construct():
    MetadataV0V1()


def test_construct_with_specific_version():
    MetadataV0V1(version="0.1.0")


def test_foo():
    m = MetadataV0V1(version="0.1.0")

    m.system_global_unique_id = "abc"
    assert m.system_global_unique_id == "abc"

