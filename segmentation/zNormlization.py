import numpy as np
from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class MRILocalZNormalize(object):
    """
    Applica la Z-Normalization dinamica per immagini MRI, 
    calcolando media e deviazione standard solo sul tessuto cerebrale.
    MODIFICA: Spinge i pixel di background a un valore estremo negativo
    per preservare il contrasto del cranio per i Vision Transformer.
    """
    def __init__(self, clip_values=True, bg_value=-5.0):
        self.clip_values = clip_values
        self.bg_value = bg_value # Il nuovo valore "abisso" per lo sfondo nero

    def __call__(self, results):
        img = results['img'].astype(np.float32)
        norm_img = np.zeros_like(img)
        
        # Iteriamo sui 3 canali
        for c in range(3):
            channel = img[:, :, c]
            
            # Maschera del cervello e dello sfondo
            brain_mask = channel > 0
            bg_mask = ~brain_mask # Tutto ciò che era nero (0)
            
            brain_voxels = channel[brain_mask]
            
            if len(brain_voxels) > 0:
                mean_val = np.mean(brain_voxels)
                std_val = np.std(brain_voxels)
                
                # Normalizziamo SOLO il cervello
                if std_val > 0:
                    channel[brain_mask] = (channel[brain_mask] - mean_val) / std_val
                else:
                    channel[brain_mask] = channel[brain_mask] - mean_val
            
            # --- IL TRUCCO È QUI ---
            # Assegniamo allo sfondo il valore -5.0 invece di lasciarlo a 0.0
            channel[bg_mask] = self.bg_value
            
            norm_img[:, :, c] = channel
            
        # Clipping intelligente
        if self.clip_values:
            # Dobbiamo clippare (tagliare a -3 e +3) SOLO i valori del cervello.
            # Se facessimo un clip globale, il nostro sfondo a -5.0 verrebbe 
            # tagliato a -3.0, perdendo parte dell'effetto "abisso".
            brain_mask_3d = img > 0
            norm_img[brain_mask_3d] = np.clip(norm_img[brain_mask_3d], -3.0, 3.0)
            
        results['img'] = norm_img
        
        # Diciamo al framework che l'immagine è già pronta
        results['img_norm_cfg'] = dict(
            mean=np.zeros(3, dtype=np.float32),
            std=np.ones(3, dtype=np.float32),
            to_rgb=False
        )
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(clip_values={self.clip_values}, bg_value={self.bg_value})'