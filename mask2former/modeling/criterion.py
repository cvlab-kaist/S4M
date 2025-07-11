# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py

"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
    generate_regular_grid_point_coords
)


from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
import os
import pdb
import sys
import matplotlib.pyplot as plt 
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule



def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def calculate_uncertainty_normalized(logits):
    """
    Normalize logits and calculate uncertainty.
    """
    assert logits.shape[1] == 1
    # Normalize logits to [0, 1]
    normalized_logits = (logits - logits.min()) / (logits.max() - logits.min() + 1e-6)
    # Compute uncertainty as 1 - normalized logits
    return -(torch.abs(normalized_logits - 0.5))

def calculate_certainty(logits):
    """
    We estimate certainty as the absolute value of the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images.
            The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains certainty scores with
            the most certain locations having the highest certainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return torch.abs(gt_class_logits)


@torch.jit.script
def calculate_multiplier(sizes):
    return (-4.*torch.log10(sizes + 1e-5) + 19.)/3.

def calculate_uncertainty_weighted(logits, ground_truth):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    gt = F.interpolate(ground_truth, size=logits.shape[-2:])

    mask_size = gt.sum(dim=(2, 3), keepdim=True)
    mask_mult = gt*calculate_multiplier(mask_size) # N, 1, H, W
    mask = -(torch.abs(gt_class_logits))
    idx = torch.where(mask_mult)
    mask[idx]  *= mask_mult[idx]

    return mask

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, deep_supervision=True):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.deep_supervision = deep_supervision

    def calc_similarity_map(self, feats, point_coords):
        """
        Compute similarity map between feature map and point coordinates.

        Args:
            feats (Tensor): Feature map of shape N, res*res, C
            point_coords (Tensor): Point coordinates of shape (N, P, 2).

        Returns:
            Tensor: The similarity map of shape (N, P, H*W).
        """
        res = int(feats.shape[1] ** 0.5)

        point_indices = (point_coords[:, :, 0] * res + point_coords[:, :, 1]).long()  # (N, P)
        extracted_features = torch.gather(feats, dim=1, index=point_indices.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))  # (N, P, C)
        
        extracted_features_norm = extracted_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, P, 1)
        extracted_features_normalized = extracted_features / extracted_features_norm  # (N, P, C)
        feats_norm = feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, res*res, 1)
        feats_normalized = feats / feats_norm  # (N, res*res, C)
        
        sim_map = torch.bmm(extracted_features_normalized, feats_normalized.permute(0, 2, 1))  # (N, P, res*res)
        
        return sim_map      # (N, P, res*res)
        
    def loss_dec_cossim_v1(self, outputs, targets, out_feats, target_feats, point_coords, indices, num_masks, num_points=128):
        """
        compute localizability distillation loss

        Args:
            outputs: Model outputs.
            targets: Ground truth target dictionaries.
            out_feats (Tensor): Student feature map of shape (N, C, H, W).
            target_feats (Tensor): Teacher feature map of shape (N, C, H, W).
            indices: Matching indices.
            num_masks (int): Number of object masks per instance.
            num_points (int): Number of points to sample from each feature map.

        Returns:
            dict: A dictionary containing the batch-wise RKD correlation loss.
        """
        tgt_sim_map = target_feats                      # N , num_points, h*w 
        res = int((tgt_sim_map.shape[-1] ** 0.5 ))
        
        out_feats = F.interpolate(out_feats, size=(res, res), mode="bilinear", align_corners=False)     # N, C, res, res
        out_feats = out_feats.permute(0, 2, 3, 1).reshape(out_feats.shape[0], -1, out_feats.shape[1])   # N, res*res, C
        out_sim_map = self.calc_similarity_map(out_feats, point_coords.squeeze())   # N, num_points, res*res

        loss = F.smooth_l1_loss(out_sim_map, tgt_sim_map, reduction="mean")
        losses = {"loss_sd": loss}
        return losses

    def loss_labels(self, outputs, targets, out_feats, target_feats, point_coords, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, out_feats, target_feats, point_coords, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        # print('mask shapes ', src_masks.shape, target_masks.shape)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        
        # print(point_logits.shape, point_labels.shape, num_masks)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, out_feats, target_feats, point_coords, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'sd' : self.loss_dec_cossim_v1 
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, out_feats, target_feats, point_coords, indices, num_masks)

    def forward(self, outputs, targets, out_feats, target_feats, point_coords):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # sample point_coords  # N x P x 2

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, out_feats, target_feats, point_coords, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if self.deep_supervision and "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "sd" :
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, out_feats, target_feats, point_coords, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
    def forward(self, outputs, targets, out_feats=None, target_feats=None, point_coords=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # sample point_coords  # N x P x 2

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, out_feats, target_feats, point_coords, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if self.deep_supervision and "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "sd" :
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, out_feats, target_feats, point_coords, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
