from pyklopp.metadata import Metadata


def test_metadata_construct():
    Metadata()


def test_metadata_obtain_property():
    m = Metadata()

    assert m.properties() is not None
    assert len(m.properties()) > 0


def test_metadata_retrieve_and_set_property():
    m = Metadata()
    key = "system_global_unique_id"
    new_value = "xyz"

    assert getattr(m, key) is None
    setattr(m, key, new_value)
    assert getattr(m, key) == new_value
