import copy
import importlib
import io

import numpy as np
from PIL import Image


class KeepKeys:

    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        return [data[key] for key in self.keep_keys]


class DecodeImagePIL:

    def __init__(self, img_mode='RGB', **kwargs):
        self.img_mode = img_mode

    def __call__(self, data):
        assert isinstance(data['image'], bytes) and len(data['image']) > 0
        img = Image.open(io.BytesIO(data['image'])).convert('RGB')

        if self.img_mode == 'Gray':
            img = img.convert('L')
        elif self.img_mode == 'BGR':
            img = Image.fromarray(np.array(img)[:, :, ::-1])

        data['image'] = img
        return data


def transform(data, ops=None):
    """transform."""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


# Class name to module mapping
MODULE_MAPPING = {
    'CTCLabelEncode': '.ctc_label_encode',
    'PARSeqAugPIL': '.rec_aug',
}


def dynamic_import(class_name):
    module_path = MODULE_MAPPING.get(class_name)
    if not module_path:
        raise ValueError(f'Unsupported class: {class_name}')

    module = importlib.import_module(module_path, package=__package__)
    return getattr(module, class_name)


def create_operators(op_param_list, global_config=None):
    ops = []
    for op_info in op_param_list:
        op_name = next(iter(op_info.keys()))
        param = copy.deepcopy(op_info[op_name]) or {}

        if global_config:
            param.update(global_config)

        if op_name in globals():
            op_class = globals()[op_name]
        else:
            op_class = dynamic_import(op_name)

        ops.append(op_class(**param))
    return ops
