import random
import socket
import sys
import os
import json
import time
import uuid
import torch.nn
import numpy as np
import pyklopp.metadata as pkmd

from cleo import Command
from pyklopp import __version__, subpackage_import
from pyklopp.util import count_parameters, save_paths_obtain_and_check, load_custom_config, load_into_property_object
from pyklopp.loading import load_modules, add_local_path_to_system


class InitCommand(Command):
    """
    Initializes a model from a given module

    init
        {model : Name of the module with initialization method for the model.}
        {--m|modules=* : Optional modules to load.}
        {--c|config=* : JSON config or path to JSON config file.}
        {--s|save= : Path to save the model to}
        {--meta= : Path to save the meta information to}
    """

    def handle(self):
        # Early check for save path
        save_path_base, model_file_name = save_paths_obtain_and_check(self)

        # Add current absolute path to system path
        # This is required for local modules to load.
        add_local_path_to_system(self.info)

        """
        Optional (local) module to load.
        There several functionalities can be bundled at one place.
        """
        modules_option = self.option('modules')
        loaded_modules = load_modules(modules_option)
        self.info('Loaded modules: %s' % loaded_modules)


        """ ---------------------------
        Static initial configuration.
        This can be overwritten by user-defined configurations.
        Dynamic computed configurations can be configured by some parameters and are computed after the user-defined
        values have been set.
        """
        config = {
            'global_unique_id': str(uuid.uuid4()),
            'pyklopp_version': __version__,
            'loaded_modules': loaded_modules,
            'gpus_exclude': [],
            'python_seed_initial': None,
            'python_seed_random_lower_bound': 0,
            'python_seed_random_upper_bound': 10000,
            'python_cwd': os.getcwd(),
            'hostname': socket.gethostname(),
            'time_config_start': time.time(),
            'save_path_base': save_path_base,
            'model_persistence_name': model_file_name,  # If later set to None/empty, model will not be persisted
        }
        # TODO metadata will replace config in future releases
        metadata = pkmd.init_metadata()
        metadata.system_loaded_modules = loaded_modules

        # Load user-defined configuration
        if self.option('config'):
            for config_option in self.option('config'):
                config_option = str(config_option)
                user_config = load_custom_config(config_option)
                config.update(user_config)
                load_into_property_object(metadata, user_config, 'arguments_')

        """ ---------------------------
        Dynamic configuration computations.
        E.g. random numbers or seeds etc.
        """

        # If desired set an initial (global) random seed
        if config['python_seed_initial'] is not None:
            random.seed(int(config['python_seed_initial']))

        # Generate a (local) initial seed, which might depend on the initial global seed
        # This enables reproducibility
        a = config['python_seed_random_lower_bound']
        b = config['python_seed_random_upper_bound']
        python_seed_local = random.randint(a, b)
        config['python_seed_local'] = python_seed_local if 'python_seed_local' not in config else config['python_seed_local']
        random.seed(config['python_seed_local'])
        np.random.seed(config['python_seed_local'])
        torch.manual_seed(config['python_seed_local'])
        torch.cuda.manual_seed(config['python_seed_local'])

        config['time_config_end'] = time.time()


        # For ease of usage, get_model can be either 'my_module.py' containing a get_model(**config) function
        # or it can be a function such as my_module.get_model with signature (**config) where my_module can be
        # a loadable module in the local path or an already loaded module via --modules
        get_model = self.argument('model')
        get_model_components = get_model.split('.')
        fn_get_model = None  # This will be the model loading function with signature fn(**config)

        if get_model.endswith('.py') or os.path.exists(get_model_components[0] + '.py'):
            module_name = get_model_components[0]
            module_file_name = module_name + '.py'
            if not os.path.exists(module_file_name):
                raise ValueError('No such local file %s' % module_file_name)

            try:
                module = __import__(module_name, fromlist=[''])
            except ModuleNotFoundError:
                raise ModuleNotFoundError('Could not import module "' + module_name + '"')

            if get_model.endswith('.py') or len(get_model_components) < 2:
                # get_model was 'my_module.py' or 'my_module'
                try:
                    fn_get_model = module.get_model
                except AttributeError:
                    raise ValueError(
                        'Could not find get_model() function in your module "%s". You probably need to define get_model(**config)' % module_name)
            else:
                # get_model was s.th. like 'my_module.load_model'
                mod = module
                for load_name in get_model_components[1:]:
                    mod = getattr(mod, load_name)
                fn_get_model = mod
        else:
            try:
                fn_get_model = subpackage_import(get_model)
            except ModuleNotFoundError as e:
                raise ValueError('Could not find module <%s> via subpackage import.'.format(get_model), e)

        config['argument_model'] = get_model
        config['get_model'] = str(fn_get_model.__name__)


        self.info('Configuration:')
        self.info(json.dumps(config, indent=2, sort_keys=True))

        # Instantiate model
        config['time_model_init_start'] = time.time()
        model = fn_get_model(**config)
        config['time_model_init_end'] = time.time()
        assert isinstance(model, torch.nn.Module)

        config['model_pythonic_type'] = str(type(model))
        config['model_trainable_parameters'] = count_parameters(model)

        if save_path_base is not None:
            if config['model_persistence_name'] is not None and len(config['model_persistence_name']) > 0:
                model_file_name = config['model_persistence_name']
                model_file_path = os.path.join(save_path_base, model_file_name)
                self.info('Saving to "%s"' % model_file_path)
                config['time_model_save_start'] = time.time()
                torch.save(model, model_file_path)
                config['time_model_save_end'] = time.time()

            config_file_name = 'config.json'
            config_file_path = os.path.join(save_path_base, config_file_name)
            self.info('Writing configuration to "%s"' % config_file_path)
            with open(config_file_path, 'w') as handle:
                json.dump({'init': config}, handle, indent=2, sort_keys=True)

        self.info('Final configuration:')
        self.info(json.dumps(config, indent=2, sort_keys=True))
        self.info('Done.')

