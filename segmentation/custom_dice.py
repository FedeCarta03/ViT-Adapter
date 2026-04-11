from mmseg.models.builder import LOSSES
from mmseg.models.losses.dice_loss import DiceLoss

@LOSSES.register_module()
class CustomDiceLoss(DiceLoss):
    """
    Eredita tutto dalla DiceLoss originale ma aggiunge forzatamente
    la proprietà loss_name che mancava.
    """
    @property
    def loss_name(self):
        return 'loss_dice'