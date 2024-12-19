#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time : 2024/3/30 22:07
@desc:
"""

from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss,SoftBCEWithLogitsLoss

def edgeSCE_loss(pred, target, edge):

    BinaryCrossEntropy_fn = SoftBCEWithLogitsLoss(smooth_factor=None, reduction='None')

    edge_weight = 4.
    loss_sce = BinaryCrossEntropy_fn(pred, target)
    edge[edge == 0] = 1.
    edge[edge > 0] = edge_weight
    loss_sce *= edge
    loss_sce_, ind = loss_sce.contiguous().view(-1).sort()
    min_value = loss_sce_[int(0.5 * loss_sce.numel())]
    loss_sce = loss_sce[loss_sce >= min_value]
    loss_sce = loss_sce.mean()
    loss = loss_sce
    return loss

