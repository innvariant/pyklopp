import json
import os
import sys

import time

import uuid

import torch

from cleo import Command

from pyklopp import __version__


class TrainCommand(Command):
    """
    Trains a model

    train
        {dataset : Data set module for training}
        {model : Path to the pytorch model file}
        {--c|config=* : Configuration JSON string or file path.}
        {--s|save : Path (including name) to save the model to}
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

        model_path = self.argument('model')

        if not os.path.exists(model_path):
            raise ValueError('Model not found in path "%s"' % model_path)


        """
        Load dataset module file
        """

        dataset_module_name = self.argument('dataset')

        dataset_module_file_name = dataset_module_name + '.py'
        if not os.path.exists(dataset_module_file_name):
            raise ValueError('No such local file %s' % dataset_module_file_name)

        # Add current absolute path to system path
        add_path = os.path.abspath('.')
        sys.path.append(add_path)
        self.info('Added %s to path' % add_path)

        try:
            module = __import__('.' + dataset_module_name, fromlist=[''])
        except ModuleNotFoundError:
            raise ModuleNotFoundError('didnt work out for module name "' + dataset_module_name + '"')

        try:
            fn_train_loader = module.train_loader
        except AttributeError:
            raise ValueError('Could not find train_loader() function in your module "%s". You probably need to define train_loader(**kwargs)' % dataset_module_name)




        config = {
            'global_unique_id': str(uuid.uuid4()),
            'pyklopp_version': __version__,
            'dataset_module_name': dataset_module_name,
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

        self.info('Configuration:')
        self.info(str(config))

        train_loader = fn_train_loader(**config)

        model = torch.load(model_path)

        cuda_no = 0
        if torch.cuda.is_available():
            cuda_no = 0
        device = torch.device('cuda:%s' % cuda_no if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        fn_loss = torch.nn.CrossEntropyLoss()

        model.to(device)

        for epoch in range(10):
            self.info('Epoch %s' % epoch)
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for i, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(features)
                loss = fn_loss(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss.backward()
                optimizer.step()

            self.info(total_loss / len(train_loader))
            self.info(100 * correct / total)

        """
        # Create a new progress bar (50 units)
        progress = self.progress_bar(50)
    
        # Start and displays the progress bar
        for _ in range(50):
            # ... do some work
    
            # Advance the progress bar 1 unit
            progress.advance()
    
            # You can also advance the progress bar by more than 1 unit
            # progress.advance(3)
    
        # Ensure that the progress bar is at 100%
        progress.finish()
        """