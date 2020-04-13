import os
import shutil

from cleo import CommandTester, Application

from pyklopp.console.commands.init import InitCommand
from pyklopp.console.commands.train import TrainCommand


def test_success_init_simple_model():
    # Arrange
    ## set up application with command
    application = Application()
    application.add(InitCommand())
    application.add(TrainCommand())

    # set up file path variables
    module_name = 'bar'
    module_file_path = module_name + '.py'
    save_path = 'bar-config/model.py'
    dataset_module = 'foo'
    dataset_module_file_path = dataset_module + '.py'

    # clean up possible existing files
    if os.path.exists(module_file_path):
        os.remove(module_file_path)
    if os.path.exists(dataset_module_file_path):
        os.remove(dataset_module_file_path)
    if os.path.exists(save_path):
        shutil.rmtree(os.path.dirname(save_path))

    # write model to file path from which we want to import from
    content = '''
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, width: int, height: int):
        super(MyModel, self).__init__()
        c = 10  # intermediate_channels
        self.conv = nn.Conv2d(3, c, 5)
        self.fc = nn.Linear((width-5+1)*(height-5+1)*c, 10)
    
    def forward(self, x):
         out = F.relu(self.conv(x))
         out = out.view(out.size(0), -1)
         return F.relu(self.fc(out))

def get_model(**args):
    return MyModel(width=32, height=32)

'''
    with open(module_file_path, 'a') as the_file:
        the_file.write(content)

    # write model to file path from which we want to import from
    content = '''
import torch
import numpy as np
from torch.utils import data

class MyDataset(data.Dataset):
    def __len__(self):
        return 200

    def __getitem__(self, index):
        return torch.rand((3, 32, 32)), np.random.randint(0, 10)

def get_dataset(**args):
    return MyDataset()

'''
    with open(dataset_module_file_path, 'a') as the_file:
        the_file.write(content)

    command_init = application.find('init')
    init_tester = CommandTester(command_init)
    init_tester.execute(module_name + ' --save=' + save_path)

    command_train = application.find('train')
    train_tester = CommandTester(command_train)
    train_tester.execute(save_path + ' ' + dataset_module + '.get_dataset')

    # Cleanup
    os.remove(module_file_path)
    os.remove(dataset_module_file_path)
    shutil.rmtree(os.path.dirname(save_path))
