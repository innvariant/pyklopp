import os
import json
from typing import List

import time

import uuid

import jsonschema
import semantic_version as semver
from importlib_resources import files

from pyklopp import __version__

latest_version = '0.1.0'


def load_schema(version=None):
    if version is None:
        version = latest_version
    return json.loads(files('pyklopp').joinpath('schema/metadata-' + version + '.json').read_text())


def validate_schema(metadata: dict, schema=None, version=None):
    if schema is None:
        version = version if version is not None else metadata['schema_version'] if 'schema_version' in metadata else None
        schema = load_schema(version)

    try:
        jsonschema.validate(instance=metadata, schema=schema)
    except jsonschema.ValidationError as e:
        raise MetadataError('The given metadata does not match the expected schema.', e)

    return schema


class MetadataError(Exception):
   """Base class for other exceptions"""
   pass


class MappingError(MetadataError):
    def __init__(self, other: Exception, data: dict = None):
        super(MappingError, self).__init__(other)
        self._data = data

    def __str__(self):
        additional = ',\ndata=<%s>' % self._data if self._data is not None else ''
        return super(MappingError, self).__str__() + additional


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class MetadataReader(dict):
    _props = {}
    _path_json_file = None
    _schema = None
    _schema_version = None

    def __init__(self, version: str = None, path_json_file=None, schema: str = None, **kwargs):
        super().__init__(**kwargs)
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

    def read(self, path=None, validate=True):
        if path is not None:
            self.path = path
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


def with_annotated_properties(Cls):
    for attr_name in Cls.__annotations__:

        def getter(self, name=attr_name):
            class_prop_key = '_' + name
            return getattr(self, class_prop_key)

        def setter(self, value, name=attr_name):
            class_prop_key = '_' + name
            setattr(self, class_prop_key, value)

        prop = property(getter, setter)
        class_prop_key = '_' + attr_name
        #prop = property(lambda self: self._get_default(key, None)).setter(lambda self, value: self._set(key, value))
        setattr(Cls, attr_name, prop)
        setattr(Cls, class_prop_key, None)

    return Cls


class PropertyObject(object):
    def properties(self):
        return [name for name in dir(self) if not name.startswith('_') and not callable(getattr(self, name))]

    @classmethod
    def with_annotated_properties(clazz, Cls):
        for attr_name in Cls.__annotations__:

            def getter(self, name=attr_name):
                class_prop_key = '_' + name
                return getattr(self, class_prop_key)

            def setter(self, value, name=attr_name):
                class_prop_key = '_' + name
                setattr(self, class_prop_key, value)

            prop = property(getter, setter)
            class_prop_key = '_' + attr_name
            #prop = property(lambda self: self._get_default(key, None)).setter(lambda self, value: self._set(key, value))
            setattr(Cls, attr_name, prop)
            setattr(Cls, class_prop_key, None)

        return Cls

    @classmethod
    def build_fill_default(clazz):
        prop_obj = clazz()
        for attr_name in prop_obj.__annotations__:
            attr_type = prop_obj.__annotations__[attr_name]
            if attr_type == str:
                setattr(prop_obj, attr_name, '')
            elif attr_type == float:
                setattr(prop_obj, attr_name, 0.0)
            elif attr_type == int:
                setattr(prop_obj, attr_name, 0)
            elif attr_type == list:
                setattr(prop_obj, attr_name, [])
            else:
                name_default_getter = 'get_default'
                if hasattr(attr_type, name_default_getter):
                    type_def_value = getattr(attr_type, name_default_getter)
                    setattr(prop_obj, attr_name, type_def_value)
                else:
                    raise ValueError('Unknown type <{T}> and could not find a default getter <get> on it to specify a default value.'.format(T=attr_type, get=name_default_getter))
        return prop_obj


@PropertyObject.with_annotated_properties
class Metadata(PropertyObject):
    schema_version: str

    system_global_unique_id: str
    system_hostname: str
    system_pyklopp_version: str
    system_python_cwd: str
    system_python_seed_initial: int
    system_python_seed_local: int
    system_python_seed_random_lower_bound: int
    system_python_seed_random_upper_bound: int
    system_loaded_modules: list

    time_config_start: float
    time_config_end: float
    time_model_init_start: float
    time_model_init_end: float
    time_model_save_start: float
    time_model_save_end: float

    params_batch_size: int
    params_learning_rate: float
    params_device: str

    arguments_batch_size: int
    arguments_device: str
    arguments_dataset: str


class MetadataMap(object):
    def __init__(self, read_dict: dict, write_dict: dict):
        self._reader = read_dict
        self._writer = write_dict

    @property
    def specification(self) -> semver.SimpleSpec:
        raise NotImplementedError('Your mapping implementation needs to provide a requirement spec for which it agrees with schema versions.')

    def applies(self, version: semver.Version) -> bool:
        return self.specification.match(version)

    @property
    def reader(self):
        return self._reader

    @property
    def writer(self):
        return self._writer

    def __call__(self, data: PropertyObject):
        self.map_all(data)

    def map_all(self, data: PropertyObject, strict=False):
        for prop in data.properties():
            try:
                setattr(data, prop, self.map(prop, self.reader))
            except MappingError as e:
                if strict:
                    raise MappingError(e)

    def remap_all(self, data: PropertyObject):
        for prop in data.properties():
            self.remap(prop, getattr(data, prop), self.writer)

    def map(self, property_name: str, reader: dict) -> str:
        raise NotImplementedError

    def remap(self, property_name: str, value, write_to_dict: dict):
        raise NotImplementedError


class ScopedMetadataMap(MetadataMap):
    @property
    def scope_separator(self) -> str:
        return '_'

    def _get_scope_map(self):
        raise NotImplementedError('You need to define a mapper which translates a scoping from flat to dict')

    def map(self, metadata_key, reader: dict):
        scoped_parts = metadata_key.split(self.scope_separator)

        current_map = self._get_scope_map()
        prefix_idx = 0
        scopes = []
        unscoped_reader = reader
        while prefix_idx < len(scoped_parts):
            current_prefix = scoped_parts[prefix_idx]
            if current_prefix in current_map and current_prefix in unscoped_reader:
                unscoped_reader = unscoped_reader[current_prefix]
                current_map = current_map[current_prefix]
                scopes.append(current_prefix)
                prefix_idx += 1
            else:
                unscoped_key = self.scope_separator.join(scoped_parts[prefix_idx:])
                key = current_map[unscoped_key] if unscoped_key in current_map else unscoped_key
                if key not in unscoped_reader:
                    raise MappingError('Did not find <{name}> which was mapped to <{key}> with scope <{scope}>'.format(
                        name=metadata_key, key=key, scope=self.scope_separator.join(scopes)),
                        data=reader
                    )
                return unscoped_reader[key]

    def remap(self, metadata_key: str, value, write_to_dict: dict):
        scoped_parts = metadata_key.split(self.scope_separator)
        current_map = self._get_scope_map()
        unscoped = write_to_dict
        prefix_idx = 0
        while prefix_idx < len(scoped_parts):
            current_prefix = scoped_parts[prefix_idx]
            if current_prefix in current_map:
                current_map = current_map[current_prefix]
                if current_prefix not in unscoped:
                    unscoped[current_prefix] = {}
                unscoped = unscoped[current_prefix]
                prefix_idx += 1
            else:
                unscoped_key = self.scope_separator.join(scoped_parts[prefix_idx:])
                key = current_map[unscoped_key] if unscoped_key in current_map else unscoped_key
                unscoped[key] = value
                break


class MetadataMapV0V1(ScopedMetadataMap):
    _map_v0v1 = {
        'schema_version': 'schema_version',
        'system': 'system',
        'time': 'time',
        'arguments': 'arguments',
        'params': 'params'
    }

    @property
    def specification(self) -> semver.SimpleSpec:
        return semver.SimpleSpec('>=0.1.0,<0.2.0')

    def _get_scope_map(self):
        return self._map_v0v1


def init_metadata(**kwargs):
    m = Metadata.build_fill_default()
    m.schema_version = latest_version

    m.system_global_unique_id = str(uuid.uuid4())
    m.system_pyklopp_version = __version__
    m.system_python_cwd = os.getcwd()
    m.system_python_seed_initial = None
    m.system_python_seed_local = None
    m.system_python_seed_random_lower_bound = 0
    m.system_python_seed_random_upper_bound = 10000

    m.time_config_start = time.time()

    for name in kwargs:
        setattr(m, name, kwargs[name])

    return m


def get_mapping(version: semver.Version):
    raise NotImplementedError


def read_metadata(path_to_file: str):
    if not os.path.exists(path_to_file):
        raise FileNotFoundError('File not found: %s' % path_to_file)
    reader = MetadataReader(path_to_file)
    reader.read()
    version = reader.schema_version
