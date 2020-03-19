import os
import sys
import torch


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def subpackage_import(name: str):
    components = name.split('.')

    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod


def add_local_path(fn_info=None):
    # Add current absolute path to system path to load local modules
    # If initialized a module previously from a local module, then it must be available in path later again
    add_path = os.path.abspath('.')
    if add_path not in sys.path:
        sys.path.append(add_path)

        if fn_info is not None:
            fn_info('Added "%s" to path.' % add_path)


def load_modules(module_args : list):
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
            except ModuleNotFoundError:
                raise ModuleNotFoundError('Could not import "%s"' % module_name)

    return loaded_modules


def load_dataset_from_argument(dataset_arg : str, assembled_config : dict) -> torch.utils.data.Dataset:
    """

    :param dataset_arg:
    :param assembled_config:
    :return:
    """
    fn_get_dataset = None  # optional function to load the dataset based on the allocated configuration
    class_dataset = None  # optional class which will be instanatiated with the configuration sub key 'dataset_config'

    add_local_path()

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

    if fn_get_dataset is None and class_dataset is None:
        raise ValueError('Neither a dataset class nor a get_dataset(**kwargs) is defined.')

    if fn_get_dataset:
        # The preferred way to load a dataset: via a getter function my_module.get_dataset(**config) -> torch.utils.data.Dataset
        dataset = fn_get_dataset(**assembled_config)

        assembled_config['dataset_getter'] = fn_get_dataset.__name__
    else:
        dataset_config = {key.replace('dataset_', ''): assembled_config[key] for key in assembled_config if key.startswith('dataset_')}

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

        dataset = class_dataset(**dataset_config)

        assembled_config['dataset_class'] = class_dataset.__name__

    return dataset
