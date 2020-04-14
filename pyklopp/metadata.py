import os
import json
import jsonschema
from importlib_resources import files

latest_version = '0.1.0'


def load_schema(version=None):
    if version is None:
        version = latest_version
    return json.loads(files('pyklopp').joinpath('schema/metadata-' + version + '.json').read_text())


def validate_schema(metadata, schema=None, version=None):
    if schema is None:
        version = version if version is not None else metadata['schema_version'] if 'schema_version' in metadata else None
        schema = load_schema(version)

    try:
        jsonschema.validate(instance=metadata, schema=schema)
    except jsonschema.ValidationError as e:
        raise MetadataError('Could not load metadata file. Did not match the specified schema', e)

    return schema


class MetadataError(Exception):
   """Base class for other exceptions"""
   pass


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Metadata(object):
    _props = {}
    _path_json_file = None
    _schema = None
    _schema_version = None

    def __init__(self, version: str=None, path_json_file=None, schema: str=None):
        self._path_json_file = path_json_file
        self._schema_version = version
        self._schema = json.loads(schema) if schema is not None else None

    @property
    def path(self):
        return self._path_json_file

    @path.setter
    def path(self, path):
        if self._path_json_file is not None:
            raise ValueError('Use one metadata object per metadata file and do not overwrite objects with new paths.')
        self._path_json_file = path

    @property
    def schema_version(self):
        return self._schema_version

    def read(self, validate=True):
        if self.path is None:
            raise ValueError('You need to specify the metadata path')

        with open(self.path, 'r') as read_handle:
            file_props = json.load(read_handle)

        if validate:
            self._schema = validate_schema(file_props)

        self._props = file_props

    def load(self, metadata: str, validate=True):
        props = json.loads(metadata)

        if validate:
            schema = self._schema
            schema_version = self._schema_version

            self._schema = validate_schema(props, schema=schema, version=schema_version)

        self._props = props

    def save(self, validate=True):
        if validate:
            validate_schema(self._props, schema=self._schema, version=self._schema_version)

        with open(self._path_json_file, 'w+') as write_handle:
            json.dump(self._props, write_handle)

    def __getitem__(self, item):
        if item in self._props:
            return self._props[item]
        raise MetadataError('Unknown metadata keyword <{name}>'.format(name=item))

    def __setitem__(self, key, value):
        self._props[key] = value

    def __getattr__(self, item):
        if item in self._props:
            value = self._props[item]
            return AttrDict(value) if type(value) is dict else value
        raise MetadataError('Unknown metadata keyword <{name}>'.format(name=item))


class MetadataV0V1(Metadata):
    def __init__(self, version: str=None, path_json_file=None, schema: str=None):
        super().__init__(version, path_json_file, schema)

        self._props = {
            'schema_version': {},
            'system': {},
            'time': {},
            'arguments': {},
            'params': {},
        }

    @property
    def system_global_unique_id(self):
        return self['system']['global_unique_id']

    @system_global_unique_id.setter
    def system_global_unique_id(self, value):
        self['system']['global_unique_id'] = value

