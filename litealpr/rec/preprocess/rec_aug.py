
class PARSeqAugPIL:

    def __init__(self, **kwargs):
        from .parseq_aug import rand_augment_transform
        self.transforms = rand_augment_transform()

    def __call__(self, data):
        img = data['image']
        img_aug = self.transforms(img)
        data['image'] = img_aug
        return data
