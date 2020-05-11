from pyklopp.metadata import PropertyObject, ScopedMetadataMap


@PropertyObject.with_annotated_properties
class TestData(PropertyObject):
    scope1_prop1: str
    scope1_prop2: str
    scope2_name: str
    scope2_sub_test: int
    scope2_sub_test2: int
    other_key: str


def test_map_strict():
    data = TestData()
    assert data.properties() is not None
    assert len(data.properties()) > 0

    reader = {
        "scope1": {"prop1": "xyz", "prop2": "abc"},
        "scope2": {"name": "hello", "sub_test": 5, "sub_test2": 10},
        "other_key": "some value",
    }
    scoped_map = ScopedMetadataMap(reader, {})
    scoped_map._get_scope_map = lambda: {"scope1": {}, "scope2": {}}
    scoped_map.map_all(data, strict=True)

    assert data.scope1_prop1 == reader["scope1"]["prop1"]
    assert data.scope1_prop2 == reader["scope1"]["prop2"]
    assert data.scope2_sub_test == reader["scope2"]["sub_test"]


def test_remapping():
    data = TestData()
    data.scope1_prop1 = "foo"
    data.scope1_prop2 = "bar"
    data.scope2_sub_test = 10

    assert data.properties() is not None
    assert len(data.properties()) > 0

    writer = {}
    scoped_map = ScopedMetadataMap({}, writer)
    scoped_map._get_scope_map = lambda: {"scope1": "scope1", "scope2": "scope2"}
    scoped_map.remap_all(data)

    assert writer["scope1"]["prop1"] == data.scope1_prop1
    assert writer["scope1"]["prop2"] == data.scope1_prop2
    assert writer["scope2"]["sub_test"] == data.scope2_sub_test
    assert writer["scope2"]["sub_test2"] == data.scope2_sub_test2


def test_other_map_strict():
    data = TestData()
    assert data.properties() is not None
    assert len(data.properties()) > 0

    reader = {
        "renamed_prop": "xyz",
        "scope1_prop2": "abc",
        "scope2": {"name": "hello", "sub": {"test": 5, "test2": 10}},
        "other_key": "some value",
    }
    scoped_map = ScopedMetadataMap(reader, {})
    scoped_map._get_scope_map = lambda: {
        "scope1_prop1": "renamed_prop",
        "scope2": {"sub": {}},
    }
    scoped_map.map_all(data, strict=True)

    assert data.scope1_prop1 == reader["renamed_prop"]
    assert data.scope1_prop2 == reader["scope1_prop2"]
    assert data.scope2_sub_test == reader["scope2"]["sub"]["test"]
    assert data.scope2_sub_test2 == reader["scope2"]["sub"]["test2"]


def test_other_remapping():
    data = TestData()
    data.scope1_prop1 = "foo"
    data.scope1_prop2 = "bar"
    data.scope2_sub_test = 10
    data.scope2_sub_test2 = "xyz"
    data.scope2_name = 30.4

    assert data.properties() is not None
    assert len(data.properties()) > 0

    writer = {}
    scoped_map = ScopedMetadataMap({}, writer)
    scoped_map._get_scope_map = lambda: {
        "scope1_prop1": "renamed_prop",
        "scope2": {"sub": {}},
    }
    scoped_map.remap_all(data)

    assert writer["renamed_prop"] == data.scope1_prop1
    assert writer["scope1_prop2"] == data.scope1_prop2
    assert writer["scope2"]["name"] == data.scope2_name
    assert writer["scope2"]["sub"]["test"] == data.scope2_sub_test
    assert writer["scope2"]["sub"]["test2"] == data.scope2_sub_test2
