import json
import os
import shutil

from cleo import Application
from cleo import CommandTester

from pyklopp.console.commands.pack import PackCommand


def test_success_pack():
    # Arrange
    # set up application with command
    application = Application()
    application.add(PackCommand())

    # set up file path variables
    module_name = "module_name_for_training"
    module_file_path = module_name + ".py"
    dataset_module = "tmp_training_dataset"
    dataset_module_file_path = dataset_module + ".py"

    # clean up possible existing files
    if os.path.exists(module_file_path):
        os.remove(module_file_path)
    if os.path.exists(dataset_module_file_path):
        os.remove(dataset_module_file_path)

    # write model to file path from which we want to import from
    content_model = """
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, width: int, height: int):
        super(MyModel, self).__init__()
        c = 6  # intermediate_channels
        self.conv = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=5)
        self.fc = nn.Linear((width-5+1)*(height-5+1)*c, 10)

    def forward(self, x):
         out = F.relu(self.conv(x))
         out = out.view(out.size(0), -1)
         return F.relu(self.fc(out))


def get_model(**args):
    return MyModel(width=32, height=32)

"""
    with open(module_file_path, "a") as model_handle:
        model_handle.write(content_model)

    # write model to file path from which we want to import from
    content_dataset = """
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

"""
    with open(dataset_module_file_path, "a") as dataset_handle:
        dataset_handle.write(content_dataset)

    command_pack = application.find("pack")
    pack_tester = CommandTester(command_pack)
    pack_tester.execute(
        "packtest"
        + " --module "
        + module_file_path
        + " --module "
        + dataset_module_file_path
    )

    # Cleanup
    os.remove(module_file_path)
    os.remove(dataset_module_file_path)
