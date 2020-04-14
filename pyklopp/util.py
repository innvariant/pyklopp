import json
import os
import socket
import sys
import time
import uuid
import warnings

import cleo
import torch

from pyklopp import __version__


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def subpackage_import(name: str):
    components = name.split('.')

    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod


def add_local_path_to_system(fn_info=None):
    # Add current absolute path to system path to load local modules
    # If initialized a module previously from a local module, then it must be available in path later again
    add_path = os.path.abspath('.')
    if add_path not in sys.path:
        sys.path.append(add_path)

        if fn_info is not None:
            fn_info('Added "%s" to path.' % add_path)


def remove_local_path_from_system():
    remove_path = os.path.abspath('.')
    sys.path = [path for path in sys.path if path != remove_path]


def load_modules(module_args: list) -> list:
    loaded_modules = []

    if module_args is None:
        module_args = []

    for module_option in module_args:
        module_option = str(module_option)
        possible_module_file_name = module_option + '.py' if not module_option.endswith('.py') else module_option
        if os.path.exists(possible_module_file_name):
            module_file_name = possible_module_file_name
            module_name = module_file_name.replace('.py', '')

            try:
                loaded_modules.append(__import__('.' + module_name, fromlist=['']))
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError('Could not import "%s". Have you added "." to your system path?' % module_name, e)

    return loaded_modules


def load_dataset_from_argument(dataset_arg : str, assembled_config : dict) -> torch.utils.data.Dataset:
    """

    :param dataset_arg:
    :param assembled_config:
    :return:
    """
    fn_get_dataset = None  # optional function to load the dataset based on the allocated configuration
    class_dataset = None  # optional class which will be instanatiated with the configuration sub key 'dataset_config'

    add_local_path_to_system()

    # For bash-completion, also allow the module name to end with .py and then simply remove it
    dataset_module_file_name = None
    dataset_possible_module_file_name = dataset_arg + '.py' if not dataset_arg.endswith('.py') else dataset_arg
    if os.path.exists(dataset_possible_module_file_name):
        # We've got an argument for which a local file exists, e.g. 'my_module' or 'my_module.py'
        dataset_module_file_name = dataset_possible_module_file_name
        dataset_module_name = dataset_module_file_name.replace('.py', '')

        try:
            module = __import__('.' + dataset_module_name, fromlist=[''])
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Could not import "%s"' % dataset_module_name)

        try:
            fn_get_dataset = module.get_dataset
        except AttributeError:
            raise ValueError('Could not find dataset() function in your module "%s". You probably need to define get_dataset(**kwargs)' % dataset_module_name)

    # Assume argument to be s.th. like 'torchvision.datasets.cifar.CIFAR10'
    if dataset_module_file_name is None:
        if not '.' in dataset_arg:
            raise ValueError('Dataset must be an attribute of a module. Expecting at least one dot.')
        try:
            class_dataset = subpackage_import(dataset_arg)
        except ModuleNotFoundError:
            raise ValueError('Could not import %s' % dataset_arg)
        except AttributeError as e:
            raise ValueError('You passed <{arg}> but we could not find the attribute within your package. It has thrown an attribute error'.format(arg=dataset_arg), e)

    if fn_get_dataset is None and class_dataset is None:
        raise ValueError('Neither a dataset class nor a get_dataset(**kwargs) is defined.')

    if fn_get_dataset:
        # The preferred way to load a dataset: via a getter function my_module.get_dataset(**config) -> torch.utils.data.Dataset
        dataset = fn_get_dataset(**assembled_config)

        assembled_config['dataset_getter'] = fn_get_dataset.__name__
    else:
        # Assemble a dataset-class-only configuration dict
        dataset_config = {}
        if 'dataset_config' in assembled_config:
            # The special key 'dataset_config' may contain several keys for the dataset
            dataset_config = assembled_config['dataset_config']
        # All keys prefixed with 'dataset_' are also added (without prefix) to the config:
        for key in assembled_config:
            if key.startswith('dataset_'):
                ds_config_key = key.replace('dataset_', '')
                dataset_config[ds_config_key] = assembled_config[key]

        # In case of class instanatiation, try to load the custom transformation function
        if 'get_dataset_transformation' in assembled_config:
            get_dataset_transformation = assembled_config['get_dataset_transformation']
            try:
                fn_get_custom_transformation = subpackage_import(get_dataset_transformation)
            except ModuleNotFoundError:
                raise ValueError('Could not import transformation %s' % get_dataset_transformation)
            except AttributeError as e:
                raise ValueError('Could not load transformation due to attribute error for %s: %s' % (get_dataset_transformation, e))

            dataset_config['transform'] = fn_get_custom_transformation()

        try:
            dataset = class_dataset(**dataset_config)
        except TypeError as e:
            raise ValueError('Could not initialize dataset class.'
                             'Probably arguments are missing.'
                             'Arguments prefixed with "dataset_" are passed directly to the dataset object.'
                             'Collected dataset config was < {config} >'.format(config=dataset_config), e)

        assembled_config['dataset_class'] = class_dataset.__name__

    return dataset


def save_paths_obtain_and_check(command: cleo.Command) -> (str, str):
    # Early check for save path
    save_path_base = None
    model_file_name = None
    if command.option('save'):
        save_path = str(command.option('save'))
        model_file_name = os.path.basename(save_path)
        # TODO check for model file name
        save_path_base = os.path.dirname(save_path)
        if len(save_path_base) < 1:
            raise ValueError('You did not specify a valid save path. Given was "%s"' % save_path)
        if os.path.exists(os.path.join(save_path_base, model_file_name)):
            raise ValueError('Path "%s" already exists' % save_path_base)

        if not os.path.exists(save_path_base):
            os.makedirs(save_path_base)

    return save_path_base, model_file_name


def build_default_config(command: cleo.Command, base_config: dict = None):
    if base_config is None:
        base_config = {}
    config = base_config.copy()

    config.update({
        'global_unique_id': str(uuid.uuid4()),
        'pyklopp_version': __version__,
        'loaded_modules': None,
        'gpus_exclude': [],
        'gpu_choice': None,  # if None, then random uniform of all available is chosen
        'python_seed_initial': None,
        'python_seed_random_lower_bound': 0,
        'python_seed_random_upper_bound': 10000,
        'python_cwd': os.getcwd(),
        'hostname': socket.gethostname(),
        'time_config_start': time.time(),
        'save_path_base': None,
        'model_persistence_name': None,
        'config_persistence_name': 'config.json',
        'argument_dataset': None,
    })

    all_options = command.option()
    all_arguments = command.argument()
    for prefix_name, keywords in zip(['option', 'argument'], [all_options, all_arguments]):
        for kwname in keywords:
            config[prefix_name+'_'+kwname] = keywords[kwname]

    return config


def build_default_training_config(base_config: dict = None):
    if base_config is None:
        base_config = {}
    config = base_config.copy()

    config.update({
        'model_root_path': None,
        'num_epochs': 10,
        'batch_size': 100,
        'learning_rate': 0.01,
        'get_dataset_transformation': 'pyklopp.defaults.get_transform',
        'get_optimizer': 'pyklopp.defaults.get_optimizer',
        'get_loss': 'pyklopp.defaults.get_loss',
        'get_dataset_test': None,
    })

    return config


def load_custom_config(config_option: str):
    if os.path.exists(config_option):
        with open(config_option, 'r') as handle:
            user_config = json.load(handle)
    else:
        try:
            user_config = json.loads(config_option)
        except TypeError as e:
            raise ValueError('Invalid JSON as config passed.', e)
        except json.decoder.JSONDecodeError as e:
            raise ValueError('Your configuration can not decoded properly. It might contain invalid JSON: < {text} >'.format(text=config_option), e)

    if not type(user_config) is dict:
        raise ValueError('User config must be a dictionary, but <{type}> was given: < {text} >'.format(text=user_config, type=type(user_config).__name__))

    reserved_prefixes = ['time', 'option', 'argument', 'config', 'python']
    if any(map(lambda k: any((k.startswith(p) for p in reserved_prefixes)), user_config)):
        warnings.warn('Custom config contains a keyword beginning with a reserved prefix.'
                      'Reserved prefixes are {prefixes}'.format(prefixes=reserved_prefixes))

    return user_config


