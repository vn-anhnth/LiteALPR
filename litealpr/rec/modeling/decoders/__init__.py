from importlib import import_module

from torch import nn

__all__ = ['build_decoder']

class_to_module = {
    'EfficientRCTCDecoder': '.efficient_rctc_decoder',
}

def build_decoder(config):
    module_name = config.pop('name')

    if module_name not in class_to_module:
        raise ValueError(f'Unsupported decoder: {module_name}')

    module_str = class_to_module[module_name]
    # Dynamically import the module and get the class
    module = import_module(module_str, package=__package__)
    module_class = getattr(module, module_name)

    return module_class(**config)
