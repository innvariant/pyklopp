import json
import os
import random
import sys
import time
import uuid
import torch
import numpy as np
import ignite

from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from cleo import Command

from pyklopp import __version__


def subpackage_import(name):
    components = name.split('.')

    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod


class TrainCommand(Command):
    """
    Trains a model

    train
        {model : Path to the pytorch model file}
        {dataset : Data set module for training}
        {--m|modules=* : Optional module file to load.}
        {--c|config=* : Configuration JSON string or file path.}
        {--s|save= : Path (including name) to save the model to}
    """

    def handle(self):
        # Early check for save path
        save_path_base = None
        model_file_name = None
        if self.option('save'):
            save_path = str(self.option('save'))
            if os.path.exists(save_path):
                raise ValueError('Path to save model to "%s" already exists' % save_path)

            model_file_name = os.path.basename(save_path)
            save_path_base = os.path.dirname(save_path)

        # Model file path (a persisted .pth pytorch model)
        model_path = self.argument('model')
        if not os.path.exists(model_path):
            raise ValueError('Model not found in path "%s"' % model_path)

        """
        Optional (local) module to load.
        There several functionalities can be bundled at one place.
        """
        modules_option = self.option('modules')
        loaded_modules = []
        if modules_option:
            # Add current absolute path to system path
            add_path = os.path.abspath('.')
            sys.path.append(add_path)
            self.info('Added %s to path' % add_path)

            for module_option in modules_option:
                possible_module_file_name = module_option + '.py' if not module_option.endswith('.py') else module_option
                if os.path.exists(possible_module_file_name):
                    module_file_name = possible_module_file_name
                    module_name = module_file_name.replace('.py', '')

                    try:
                        loaded_modules.append(__import__('.' + module_name, fromlist=['']))
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError('Could not import "%s"' % module_name)
        self.info('Loaded modules: %s' % loaded_modules)

        """
        Load dataset module file
        """
        dataset = None  # will be the dataset class, should be of type 'torch.utils.data.Dataset'
        fn_get_dataset = None  # optional function to load the dataset based on the allocated configuration
        class_dataset = None  # optional class which will be instanatiated with the configuration sub key 'dataset_config'

        # Either 'my_dataset' / 'my_dataset.py' or s.th. like 'torchvision.datasets.cifar.CIFAR10' or 'torchvision.datasets.mnist.MNIST'
        dataset_argument = self.argument('dataset')

        # For bash-completion, also allow the module name to end with .py and then simply remove it
        dataset_module_file_name = None
        dataset_possible_module_file_name = dataset_argument + '.py' if not dataset_argument.endswith('.py') else dataset_argument
        if os.path.exists(dataset_possible_module_file_name):
            dataset_module_file_name = dataset_possible_module_file_name
            dataset_module_name = dataset_module_file_name.replace('.py', '')

            # Add current absolute path to system path
            add_path = os.path.abspath('.')
            sys.path.append(add_path)
            self.info('Added %s to path' % add_path)

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
            if not '.' in dataset_argument:
                raise ValueError('Dataset must be an attribute of a module. Expecting at least one dot.')
            try:
                class_dataset = subpackage_import(dataset_argument)
            except ModuleNotFoundError:
                raise ValueError('Could not import %s' % dataset_argument)

        """
        Assemble configuration with pre-defined config and user-defined json/file.
        """
        config = {
            'global_unique_id': str(uuid.uuid4()),
            'pyklopp_version': __version__,
            'dataset_argument': dataset_argument,
            'python_seed_initial': None,
            'python_seed_random_lower_bound': 0,
            'python_seed_random_upper_bound': 10000,
            'time_config_start': time.time(),
            'model_persistence_name': model_file_name,  # If later set to None/empty, model will not be persisted
            'save_path_base': save_path_base,
            'config_persistence_name': 'config.json',
            'config_key': 'training',
            'gpus_exclude': [],
            'gpu_choice': None,  # if None, then random uniform of all available is chosen
            'num_epochs': 10,
            'learning_rate': 0.01,
            'dataset_config': {},
            'get_dataset_transformation': 'pyklopp.defaults.get_transform',
            'get_optimizer': 'pyklopp.defaults.get_optimizer',
            'get_loss': 'pyklopp.defaults.get_loss',
            'get_dataset_test': None,
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


        # Dynamic configuration computations.
        # E.g. random numbers or seeds etc.

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

        # Re-Check file path for persistence before going into training
        if config['model_persistence_name'] is not None and len(config['model_persistence_name']) > 0:
            model_file_name = config['model_persistence_name']
            model_file_path = os.path.join(save_path_base, model_file_name)
            if os.path.exists(model_file_path):
                raise ValueError('Model file path "%s" already exists' % model_file_path)

        config['time_config_end'] = time.time()

        """
        Load configured dataset for training.
        """
        self.info('Loading dataset.')
        if dataset is None:
            if fn_get_dataset is None and class_dataset is None:
                raise ValueError('Neither a dataset class nor a get_dataset(**kwargs) is defined.')
            if fn_get_dataset:
                # Dataset configuration key is only used for class instantiation
                del(config['dataset_config'])
                del(config['get_dataset_transformation'])

                dataset = fn_get_dataset(**config)
            else:
                # In case of class instanatiation, try to load the custom transformation function
                if 'dataset_config' in config and 'get_dataset_transformation' in config:
                    get_dataset_transformation = config['get_dataset_transformation']
                    try:
                        fn_get_custom_transformation = subpackage_import(get_dataset_transformation)
                    except ModuleNotFoundError:
                        raise ValueError('Could not import transformation %s' % get_dataset_transformation)
                    except AttributeError as e:
                        raise ValueError('Could not load transformation due to attribute error for %s: %s' % (get_dataset_transformation, e))
                    config['dataset_config']['transform'] = fn_get_custom_transformation()

                dataset = class_dataset(**config['dataset_config'])

        config['dataset'] = str(dataset.__class__.__name__)
        n_training_samples = 30000
        n_val_samples = 5000
        train_sampler = torch.utils.data.SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
        validation_sampler = torch.utils.data.SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
        config['time_dataset_loading_start'] = time.time()
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=2
        )
        dataiter = iter(train_loader)
        initial_features, initial_labels = dataiter.next()
        config['time_dataset_loading_end'] = time.time()

        # Determine device to use (e.g. cpu, gpu:0, gpu:1, ..)
        if torch.cuda.is_available():
            cuda_no = config['gpu_choice']
            if cuda_no is None:
                cuda_no = np.random.choice(np.setdiff1d(np.arange(torch.cuda.device_count()), config['gpus_exclude']))
            elif cuda_no in config['gpus_exclude']:
                raise ValueError('Your configured GPU device number is in the exclusion list of your configuration.')
            device = torch.device('cuda:%s' % cuda_no)
        else:
            device = torch.device('cpu')
        config['device'] = str(device)

        # Load the model
        model = torch.load(model_path)
        model.to(device)

        fn_get_optimizer = subpackage_import(config['get_optimizer'])
        optimizer = fn_get_optimizer(model.parameters(), **config)
        config['optimizer'] = str(optimizer.__class__.__name__)

        fn_get_loss = subpackage_import(config['get_loss'])
        fn_loss = fn_get_loss(**config)
        config['loss'] = str(fn_loss.__class__.__name__)

        self.info('Configuration:')
        self.info(str(config))

        """
        The training loop.
        """
        trainer = create_supervised_trainer(model, optimizer, fn_loss, device=device)

        pbar = ProgressBar()
        pbar.attach(trainer, 'all')

        config['time_model_training_start'] = time.time()
        trainer.run(train_loader, max_epochs=config['num_epochs'])
        config['time_model_training_end'] = time.time()

        """
        Evaluation
        """
        if 'get_dataset_test' in config and config['get_dataset_test'] is not None:
            fn_get_dataset_test = subpackage_import(config['get_dataset_test'])
            dataset_test = fn_get_dataset_test(**config)
            config['dataset_test'] = str(dataset_test.__class__.__name__)

            dataset_test_length = len(dataset_test)
            test_sampler = torch.utils.data.SubsetRandomSampler(np.arange(dataset_test_length, dtype=np.int64))
            test_loader = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=config['batch_size'],
                sampler=test_sampler,
                num_workers=2
            )

            metric_accuracy = ignite.metrics.Accuracy()
            metric_precision = ignite.metrics.Precision(average=False)
            metric_recall = ignite.metrics.Recall(average=False)
            metric_f1 = (metric_precision * metric_recall * 2 / (metric_precision + metric_recall)).mean()
            evaluation_metrics = {
                'accuracy': metric_accuracy,
                'precision': ignite.metrics.Precision(average=True),
                'recall': ignite.metrics.Recall(average=True),
                'f1': metric_f1,
                'loss': ignite.metrics.Loss(fn_loss)
            }
            evaluator = create_supervised_evaluator(
                model,
                metrics=evaluation_metrics,
                device=device
            )

            config['time_model_evaluation_start'] = time.time()
            evaluation_state = evaluator.run(test_loader)
            config['time_model_evaluation_end'] = time.time()

            for metric_name in evaluation_metrics:
                config['evaluation_%s' % metric_name] = np.float(evaluation_state.metrics[metric_name])

        """
        Optional configuration & model persistence.
        """
        if save_path_base is not None:
            if config['model_persistence_name'] is not None and len(config['model_persistence_name']) > 0:
                model_file_name = config['model_persistence_name']
                model_file_path = os.path.join(save_path_base, model_file_name)
                self.info('Saving to "%s"' % model_file_path)
                config['time_model_save_start'] = time.time()
                torch.save(model, model_file_path)
                config['time_model_save_end'] = time.time()

            config_file_name = config['config_persistence_name']
            config_file_path = os.path.join(save_path_base, config_file_name)

            full_config = {}
            if os.path.exists(config_file_path):
                # Load possible existing config
                with open(config_file_path, 'r') as handle:
                    full_config = json.load(handle)

            config_key = config['config_key']
            if config_key not in full_config:
                full_config[config_key] = []

            full_config[config_key].append(config)

            self.info('Writing configuration to "%s"' % config_file_path)
            with open(config_file_path, 'w') as handle:
                json.dump(full_config, handle, indent=2)

        self.info('Done.')
