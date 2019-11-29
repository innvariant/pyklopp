import torch


class Initializer(object):
    def __call__(self, module: torch.nn.Module):
        raise NotImplementedError()

    def __str__(self):
        return "UnknownInitializer"


class Norm001Initializer(Initializer):
    def __call__(self, m : torch.nn.Module):
        if type(m) in [torch.nn.Linear, pypaddle.sparse.MaskedLinearLayer]:
            torch.nn.init.normal_(m.weight, 0, 0.1)
            m.bias.data.fill_(0.01)

    def __str__(self):
        return "Norm(0; 0.1)"
