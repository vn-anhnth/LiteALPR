import copy
from importlib import import_module

__all__ = ['build_post_process']

# Define class name to module path mapping
module_mapping = {
    'CTCLabelDecode': '.ctc_postprocess',
}

def build_post_process(config, global_config=None):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)

    assert module_name in module_mapping, Exception(
        f'post process only support {list(module_mapping.keys())}')

    module_path = module_mapping[module_name]

    # Dynamically import modules
    module = import_module(module_path, package=__package__)
    module_class = getattr(module, module_name)

    return module_class(**config)
