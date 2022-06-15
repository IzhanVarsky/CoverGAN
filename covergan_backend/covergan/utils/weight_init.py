import torch


def apply_weight_function_to_model(m, f, kwargs: dict):
    if isinstance(m, torch.nn.Linear):
        f(m.weight, **kwargs)
    elif isinstance(m, torch.nn.RNN):
        for weight in m.parameters():
            f(weight, **kwargs)
    elif isinstance(m, torch.nn.Conv2d):
        f(m.weight)
    elif isinstance(m, torch.nn.BatchNorm2d):
        f(m.weight, **kwargs)
        torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.Sequential):
        for module in m:
            apply_weight_function_to_model(module, f, kwargs)


def weights_init_uniform(m, bound: float):
    apply_weight_function_to_model(m, torch.nn.init.uniform_, {"a": -bound, "b": bound})


def weights_init_normal(m):
    apply_weight_function_to_model(m, torch.nn.init.normal_, {})
