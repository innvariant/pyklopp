import json
import os
import random
import socket
import sys
import time
import uuid
import torch
import numpy as np
import ignite

from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_evaluator
from cleo import Command

from pyklopp import __version__, subpackage_import
from pyklopp.util import load_modules, load_dataset_from_argument, add_local_path


class EvalCommand(Command):
    """
    Evaluates a pre-trained model on a given test data set.

    eval
        {model : Path to the pytorch model file}
        {testset : Function to retrieve the test set based on the assembled configuration}
        {--m|modules=* : Optional module file to load.}
        {--c|config=* : Configuration JSON string or file path.}
        {--s|save= : Path (including optional name) to save the configuration to, e.g. sub/path/config.json}
    """

    def handle(self):
        """
            Early path checking. This helps detecting errors early before loading a model or a dataset.
        """
        # Early check for model path
        model_path = str(self.argument('model'))
        if not os.path.exists(model_path):
            raise ValueError('No model file found on "%s"' % model_path)

        # Early check for save path
        existing_config = None
        save_path_base = None
        config_file_name = None
        if self.option('save'):
            save_path = str(self.option('save'))
            if os.path.exists(save_path):
                with open(save_path, 'r') as handle:
                    existing_config = json.load(handle)

            config_file_name = os.path.basename(save_path)
            save_path_base = os.path.dirname(save_path)
            if not os.path.isdir(save_path_base):
                os.makedirs(save_path_base)

        # Load optional modules
        add_local_path(self.info)
        modules_option = self.option('modules')
        loaded_modules = load_modules(modules_option)

        """
            Assemble configuration
        """
        dataset_argument = str(self.argument('testset'))
        config = {
            'global_unique_id': str(uuid.uuid4()),
            'pyklopp_version': __version__,
            'loaded_modules': loaded_modules,
            'python_seed_initial': None,
            'python_seed_random_lower_bound': 0,
            'python_seed_random_upper_bound': 10000,
            'python_cwd': os.getcwd(),
            'hostname': socket.gethostname(),
            'time_config_start': time.time(),
            'model_path': model_path,
            'save_path_base': save_path_base,
            'config_persistence_name': 'config.json',
            'config_key': 'evaluation',
            'gpus_exclude': [],
            'gpu_choice': None,  # if None, then random uniform of all available is chosen
            'batch_size': 100,
            'argument_dataset': dataset_argument,
            'get_dataset_transformation': 'pyklopp.defaults.get_transform',
            'get_loss': 'pyklopp.defaults.get_loss'
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

        """
        Load configured dataset for evaluation.
        """
        self.info('Loading dataset.')
        dataset = load_dataset_from_argument(dataset_argument, config)

        config['dataset'] = str(dataset.__class__.__name__)
        n_samples = len(dataset)
        train_sampler = torch.utils.data.SubsetRandomSampler(np.arange(n_samples, dtype=np.int64))
        config['time_dataset_loading_start'] = time.time()
        dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=2
        )
        dataiter = iter(dataset_loader)
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
        try:
            model = torch.load(model_path, map_location=device)
        except ModuleNotFoundError as e:
            raise ValueError('Could not find module when loading model: %s' % e)
        model.to(device)
        config['model_pythonic_type'] = str(type(model))

        fn_get_loss = subpackage_import(config['get_loss'])
        fn_loss = fn_get_loss(**config)
        config['loss'] = str(fn_loss.__class__.__name__)

        self.info('Configuration:')
        self.info(json.dumps(config, indent=2, sort_keys=True))

        """
            Evaluation
        """
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
        evaluation_state = evaluator.run(dataset_loader)
        config['time_model_evaluation_end'] = time.time()

        for metric_name in evaluation_metrics:
            config['evaluation_%s' % metric_name] = np.float(evaluation_state.metrics[metric_name])

        self.info('Final configuration:')
        self.info(json.dumps(config, indent=2, sort_keys=True))
        self.info('Done.')