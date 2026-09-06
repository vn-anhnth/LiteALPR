__all__ = ['build_encoder']

from importlib import import_module

name_to_module = {
    'SVTRv2': '.svtrv2',
    'SVTRv2LNConv': '.svtrv2_lnconv',
    'SVTRv2LNConvTwo33': '.svtrv2_lnconv_two33',
}


def build_encoder(config):

    module_name = config.pop('name')
    assert module_name in name_to_module, Exception(
        f'Encoder only supports: {list(name_to_module.keys())}')

    module_path = name_to_module[module_name]
    mod = import_module(module_path, package=__package__)
    module_class = getattr(mod, module_name)(**config)

    return module_class
