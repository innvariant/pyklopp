import torchvision
import torch

def get_transform(**kwargs):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])


def get_optimizer(parameters, learning_rate, **kwargs):
    return torch.optim.SGD(parameters, lr=learning_rate)



def get_loss(**kwargs):
    return torch.nn.CrossEntropyLoss()
