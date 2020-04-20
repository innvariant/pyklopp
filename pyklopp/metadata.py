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


@PropertyObject.with_annotated_properties
class Metadata(PropertyObject):
    schema_version: str
    system_global_unique_id: str
    system_pyklopp_version: str
    system_python_cwd: str
    system_python_seed_initial: int
    system_python_seed_local: int
    system_python_seed_random_lower_bound: int
    system_python_seed_random_upper_bound: int

    time_config_start: int
    time_config_end: int
    time_model_init_start: int
    time_model_init_end: int
    time_model_save_start: int
    time_model_save_end: int

    params_batch_size: int
    params_learning_rate: float
    params_device: str


class MetadataMap(object):
    def __init__(self, read_dict: dict, write_dict: dict):
        self._reader = read_dict
        self._writer = write_dict

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
            except KeyError as e:
                if strict:
                    raise KeyError(e)

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
        while prefix_idx < len(scoped_parts):
            current_prefix = scoped_parts[prefix_idx]
            if current_prefix in current_map:
                scopes.append(current_prefix)
                current_map = current_map[current_prefix]
                prefix_idx += 1
            else:
                break

        unscoped_key = self.scope_separator.join(scoped_parts[prefix_idx:])
        key = current_map[unscoped_key] if unscoped_key in current_map else unscoped_key

        unscoped_dict = reader
        for scope in scopes:
            unscoped_dict = unscoped_dict[scope]
        if key not in unscoped_dict:
            raise KeyError('Did not find <{name}> which was mapped to <{key}> with scope <{scope}>'.format(name=metadata_key, key=key, scope=self.scope_separator.join(scopes)))
        return unscoped_dict[key]

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
        'schema_version': {},
        'system': {},
        'time': {},
        'arguments': {},
        'params': {},
    }

    def _get_map(self):
        return self._map_v0v1

    def map(self, metadata_key, reader: dict):
        scoped_parts = metadata_key.split('_')

        current_map = self._map
        prefix_idx = 0
        scopes = []
        while prefix_idx < len(scoped_parts):
            current_prefix = scoped_parts[prefix_idx]
            if current_prefix in current_map:
                scopes.append(current_prefix)
                current_map = current_map[current_prefix]
                prefix_idx += 1
            else:
                break

        unscoped_key = '_'.join(scoped_parts[prefix_idx:])

        if unscoped_key in current_map:
            key = current_map[unscoped_key]
        else:
            key = unscoped_key

        unscoped_dict = reader
        for scope in scopes:
            unscoped_dict = unscoped_dict[scope]
        if key not in unscoped_dict:
            raise ValueError('Could not map <{name}> which was mapped to {key} with scope {scope}'.format(name=metadata_key, key=key, scope='_'.join(scopes)))
        return unscoped_dict[key]

    def remap(self, metadata_key: str, value, write_to_dict: dict):
        scoped_parts = metadata_key.split('_')
