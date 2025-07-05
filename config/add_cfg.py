# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from detectron2.config import CfgNode as CN
import os 

def add_s4m_config(cfg):
    # ---------------------------------------------------------------------------- #
    # S4M options
    # ---------------------------------------------------------------------------- #
    _C = cfg
    _C.SSL = CN()
    _C.SSL.TRAIN_SSL = True
    _C.SSL.TEACHER_CKPT = ""
    _C.SSL.BURNIN_ITER = 15000
    _C.SSL.PERCENTAGE = 100
    _C.SSL.FREQ = 1 #3
    _C.SSL.EMA_DECAY = 0.9996
    _C.SSL.CKPT_TARGET = "TEACHER"
    _C.SSL.EVAL_WHO = "STUDENT"
    _C.SSL.WEIGHTS = ""
    _C.SSL.USE_SAM = False
    _C.SSL.SAM_CKPT_DIR = ""
    _C.SSL.SAM_TYPE = "large"      # base_plus, large, small, tiny
    _C.SSL.SAM_CFG = sam2_config(_C.SSL.SAM_TYPE)
    _C.SSL.INTERM_STAGE = False
    _C.SSL.INTERM_TRAINABLE = ""
    _C.SSL.SAM_RANDOM_AUG = False
    _C.SSL.SAM_CP_AUG = True
    _C.SSL.AUG_STATIC = True
    _C.SSL.SAM_PMPT = "npoints"
    _C.SSL.REFINE_THRESH = None
    _C.SSL.NUM_POINTS = 3
    _C.SSL.SD_WEIGHT = 1
    
    
def sam2_config(sam2_type):
    _cfgs = {
        "small": "sam2_hiera_s.yaml",
        "base_plus": "sam2_hiera_b+.yaml",
        "medium": "sam2_hiera_m.yaml",
        "large": "sam2_hiera_l.yaml",
    }
    return _cfgs[sam2_type]   
    