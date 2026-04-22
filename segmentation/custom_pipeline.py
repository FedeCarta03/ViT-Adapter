import numpy as np
from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadNumpyFromFile(object):
    """Carica un array .npy 2D e lo replica su 3 canali."""
    def __init__(self, to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        filename = results['img_info']['filename']
        img = np.load(filename)
        
        if self.to_float32:
            img = img.astype(np.float32)
            
        # Replica la slice su 3 canali per ViT
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
            
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        
        # Bypass della normalizzazione ImageNet
        results['img_norm_cfg'] = dict(
            mean=np.zeros(3, dtype=np.float32),
            std=np.ones(3, dtype=np.float32),
            to_rgb=False
        )
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(to_float32={self.to_float32})'