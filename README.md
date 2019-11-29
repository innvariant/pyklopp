# pyklopp
Utility to initialize and train pytorch models.
It persists all relevant hyperparameters, timings and model configurations.

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
import pypaddle.sparse

def init(**kwargs):
    input_size = kwargs['input_size']
    output_size = kwargs['output_size']

    return pypaddle.sparse.MaskedDeepFFN(input_size, output_size, [100, 100])
```

```python
# mnist.py - Your dataset loading functions
import pypaddle.util


def train_loader(**kwargs):
    batch_size = kwargs['batch_size']

    mnist_train_loader, mnist_test_loader, _, selected_root = pypaddle.util.get_mnist_loaders(batch_size, '/media/data/set/mnist')
    return mnist_train_loader


def test_loader(**kwargs):
    batch_size = kwargs['batch_size']

    mnist_train_loader, mnist_test_loader, _, selected_root = pypaddle.util.get_mnist_loaders(batch_size, '/media/data/set/mnist')
    return mnist_test_loader
```
