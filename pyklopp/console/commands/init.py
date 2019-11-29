import random
import sys
import os
import json
import time
import uuid
import torch.nn

from cleo import Command
from pyklopp import __version__


class InitCommand(Command):
    """
    Initializes a model from a given module

    init
        {module : Name of the module with initialization method for the model.}
        {--c|config=* : JSON config or path to JSON config file.}
        {--s|save= : Path to save the model/config to}
    """

    def handle(self):
        # Early check for save path
        save_path_base = None
        model_file_name = None
        if self.option('save'):
            save_path = str(self.option('save'))
            model_file_name = os.path.basename(save_path)
            save_path_base = os.path.dirname(save_path)
            if os.path.exists(save_path_base):
                raise ValueError('Path "%s" already exists' % save_path_base)

            os.makedirs(save_path_base)

        module_name = self.argument('module')

        module_file_name = module_name + '.py'
        if not os.path.exists(module_file_name):
            raise ValueError('No such local file %s' % module_file_name)

        # Add current absolute path to system path
        add_path = os.path.abspath('.')
        sys.path.append(add_path)
        self.info('Added %s to path' % add_path)

        try:
            module = __import__('.' + module_name, fromlist=[''])
        except ModuleNotFoundError:
            raise ModuleNotFoundError('didnt work out for module name "' + module_name + '"')

        try:
            fn_init = module.init
        except AttributeError:
            raise ValueError('Could not find init() function in your module "%s". You probably need to define init(**kwargs)' % module_name)

        """ ---------------------------
        Static initial configuration.
        This can be overwritten by user-defined configurations.
        Dynamic computed configurations can be configured by some parameters and are computed after the user-defined
        values have been set.
        """
        config = {
            'global_unique_id': str(uuid.uuid4()),
            'pyklopp_version': __version__,
            'init_module_name': module_name,
            'gpus_exclude': [],
            'python_seed_initial': None,
            'python_seed_random_lower_bound': 0,
            'python_seed_random_upper_bound': 10000,
            'time_config_start': time.time(),
            'model_persistence_name': model_file_name,  # If later set to None/empty, model will not be persisted
        }

        # Load user-defined configuration
        if self.option('config'):
            for config_option in self.option('config'):
                config_option = str(config_option)
                if os.path.exists(config_option):
                    self.info('Loading configuration from "%s"' % config_option)
                    with open(config_option, 'r') as handle:
                        user_config = json.load(handle)
                else:
                    try:
                        user_config = json.loads(config_option)
                    except TypeError:
                        raise ValueError('Invalid JSON as config passed.')
                assert type(user_config) is dict, 'User config must be a dictionary.'

                config.update(user_config)

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

        config['time_config_end'] = time.time()

        self.info('Configuration:')
        self.info(str(config))

        # Instantiate model
        config['time_model_init_start'] = time.time()
        model = fn_init(**config)
        config['time_model_init_end'] = time.time()
        assert isinstance(model, torch.nn.Module)

        config['model_pythonic_type'] = str(type(model))

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
                json.dump({'init': config}, handle, indent=2)

        self.info('Done.')

