# pyklopp [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Python 3.6](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) ![Tests](https://github.com/innvariant/pyklopp/workflows/Tests/badge.svg)
Tired of logging all hyperparameter configurations of your model prototyping to disk?

Pyklopp is a tool to initialize, train and evaluate pytorch models (currently for supervised problems).
It persists all relevant hyperparameters, timings and model configurations.
Your prototyping is reduced to defining your model, the dataset and your desired parameters.

**Important note:** we are undergoing an architectural change from writing config json files to writing meta data files given a jsonschema.
So to keep your experiments reproducible and program against a current design of pyklopp, reference the exact pyklopp version in your experiment.
E.g. for your *environment.yml*:
```yaml
dependencies:
- pip:
  - pyklopp==0.3.0
```

![Workflow sketch for developing a model and running it with pyklopp.](res/approach.png)


## Installation
You can install a version from PyPi with: ``pip install pyklopp``.

To install the latest development build, you can clone the repository and invoke ``poetry build`` (having poetry installed).
Then you can use the built package and install it with pip in your current environment by ``pip install dist/xxx.whl``.


# Defining model & dataset
Used imports:
```python
import pypaddle.sparse
import pypaddle.util
import torch.nn as nn
import torch.nn.functional as F

```
Specify your model in a plain python file, e.g.:
```python
# my_model.py

# Your model can be any pytorch module
# Make sure to not define it locally (e.g. within the get_model()-function)
class LeNet(nn.Module):
    def __init__(self, output_size):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# This is your model-instantiation function
# It receives an assembled configuration keyword argument list and should return an instance of your model
def get_model(**kwargs):
    output_size = int(kwargs['output_size'])

    return LeNet(output_size)
```

Invoke pyklopp to initialize it: ``pyklopp init my_model.get_model --save='test/model.pth' --config='{"output_size": 10}'``
Train it on *cifar10*:
- ``pyklopp train test/model.pth cifar10.py --save='test/trained.pth' --config='{"batch_size": 100}'``
- ``pyklopp train test/model.pth torchvision.datasets.cifar.CIFAR10 --save 'test/trained.pth' --config='{"dataset_root": "/media/data/set/cifar10"}'``


# Examples

```bash
# Initializing & Saving: mymodel.py
pyklopp init foo --save='mymodel1/model.pth'
pyklopp init foo --config='{"python_seed_initial": 100}' --save='mymodel2/model.pth'

# Training
pyklopp train path/to/mymodel.pth mnist
pyklopp train path/to/mymodel.pth mnist --config='{"batch_size": 100, "learning_rate": 0.01}'
```

```python
# foo.py - Your model initialization function

def init(**kwargs):
    input_size = kwargs['input_size']
    output_size = kwargs['output_size']

    return pypaddle.sparse.MaskedDeepFFN(input_size, output_size, [100, 100])
```

```python
# mnist.py - Your dataset loading functions

def train_loader(**kwargs):
    batch_size = kwargs['batch_size']

    mnist_train_loader, mnist_test_loader, _, selected_root = pypaddle.util.get_mnist_loaders(batch_size, '/media/data/set/mnist')
    return mnist_train_loader


def test_loader(**kwargs):
    batch_size = kwargs['batch_size']

    mnist_train_loader, mnist_test_loader, _, selected_root = pypaddle.util.get_mnist_loaders(batch_size, '/media/data/set/mnist')
    return mnist_test_loader
```


# Development
- Create wheel files in *dist/*: ``poetry build``
- Install wheel in current environment with pip: ``pip install path/to/pyklopp/dist/pyklopp-0.1.0-py3-none-any.whl``

## Running CI image locally
Install latest *gitlab-runner* (version 12.3 or up):
```bash
# For Debian/Ubuntu/Mint
curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash

# For RHEL/CentOS/Fedora
curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh | sudo bash

apt-get update
apt-get install gitlab-runner

$ gitlab-runner -v
Version:      12.3.0
```
Execute job *tests*: ``gitlab-runner exec docker test-python3.6``

## Running github action locally
Install *https://github.com/nektos/act*.
Run ``act``

## Running pre-commit checks locally
``poetry run pre-commit run --all-files``
