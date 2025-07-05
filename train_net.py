# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import warnings 

warnings.filterwarnings("ignore")
import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
# from config.config import get_cfg
from config.add_cfg import add_s4m_config
# from detectron2.data import MetadataCatalog, build_detection_train_loader
from data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    # DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from PIL import Image

from modules.defaults import DefaultTrainer

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    ImageDatasetMapper,
    MaskFormerImageDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)


# override function process of CityscapesInstanceEvaluator
class CityscapesInstanceEvaluator_(CityscapesInstanceEvaluator):
    """
    Evaluate instance segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """
    def __init__(self, dataset_name, output_dir):
        """
        Args:
            dataset_name (str): the name of the dataset.
            output_dir (str): the directory to save output results.
        """
        # parent class initialization
        super().__init__(dataset_name)
        # additional initialization
        self.outputdir = output_dir
        os.makedirs(self.outputdir, exist_ok=True)  
    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")

            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                num_instances = len(output)
                with open(pred_txt, "w") as fout:
                    for i in range(num_instances):
                        pred_class = output.pred_classes[i]
                        classes = self._metadata.thing_classes[pred_class]
                        class_id = name2label[classes].id
                        score = output.scores[i]
                        mask = output.pred_masks[i].numpy().astype("uint8")
                        png_filename = os.path.join(
                            self._temp_dir, basename + "_{}_{}.png".format(i, classes)
                        )
                        # png_filename_save = os.path.join(self.outputdir, os.path.basename(png_filename))
                        png_filename_save = os.path.join(self.outputdir, basename + "_{}_{}_{}.png".format(i, classes, score))

                        Image.fromarray(mask * 255).save(png_filename)
                        Image.fromarray(mask * 255).save(png_filename_save)
                        fout.write(
                            "{} {} {}\n".format(os.path.basename(png_filename), class_id, score)
                        )
            else:
                # Cityscapes requires a prediction file for every ground truth image.
                with open(pred_txt, "w") as fout:
                    pass

    

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, caching=False):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            if caching :
                return CityscapesInstanceEvaluator_(dataset_name, output_dir = output_folder)
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)

            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.INPUT = cfg.INPUT
            cfg_gan.DATASETS.TRAIN = ("cityscapes_fine_unlabel_train")
            
            cfg_gan.freeze()
            cfg.freeze()
            mapper_unl = MaskFormerImageDatasetMapper(cfg_gan, True)

            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.INPUT = cfg.INPUT
            cfg_gan.DATASETS.TRAIN = ("cityscapes_fine_unlabel_train")
            cfg_gan.freeze()
            cfg.freeze()
            mapper_unl = MaskFormerImageDatasetMapper(cfg_gan, True)
            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            
            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.INPUT = cfg.INPUT
            cfg_gan.DATASETS.TRAIN = ("cityscapes_fine_unlabel_train")

            if cfg.SSL.TRAIN_SSL:
                cfg_gan.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
                # cfg.DATALOADER.REPEAT_THRESHOLD = 10.0

            cfg_gan.freeze()
            cfg.freeze()
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            # unlabeled mapper : 
            mapper_unl = MaskFormerImageDatasetMapper(cfg_gan, True) #ImageDatasetMapper(cfg_gan, True)
            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.DATASETS.TRAIN = ("coco_2017_unlabel_train",)
            if cfg.SSL.TRAIN_SSL:
                cfg_gan.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
            cfg_gan.freeze()
            cfg.freeze()
            mapper_unl = ImageDatasetMapper(cfg_gan, True)
            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.DATASETS.TRAIN = ("coco_2017_unlabel_train",)
            cfg_gan.freeze()
            cfg.freeze()
            mapper_unl = ImageDatasetMapper(cfg_gan, True)
            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper), None

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_s4m_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Update dataloader for SSL case.
    if cfg.SSL.PERCENTAGE != 100:
        cfg.DATASETS.TRAIN = (cfg.DATASETS.TRAIN[0]+f"_{cfg.SSL.PERCENTAGE}",)
    if cfg.SSL.TRAIN_SSL:
        cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
        cfg.DATALOADER.REPEAT_THRESHOLD = 10.0
    
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

def parse_ckpt(ckpt_dir):
    if ckpt_dir.startswith('detectron2://'):
        return 'STUDENT'
    ckpt = torch.load(ckpt_dir, map_location='cpu')['model']
    student_keys = False
    teacher_keys = False
    
    # detect which keys are in the ckpt.
    for k in ckpt.keys():
        if k.startswith('modelStudent'):
            student_keys = True
        if k.startswith('modelTeacher'):
            teacher_keys = True
        if student_keys and teacher_keys:
            break

    if student_keys and teacher_keys:
        return 'BOTH'
    elif student_keys:
        return 'STUDENT'
    elif teacher_keys:
        return 'TEACHER'
    else:
        return 'NEITHER'
    

import glob
import cv2
import numpy as np
import tqdm
WINDOW_NAME = "mask2former demo"
import time 

def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=args.resume)

        if cfg.SSL.TRAIN_SSL:
            which = parse_ckpt(cfg.MODEL.WEIGHTS)
            if which == 'BOTH':
                checkp = trainer._trainer.ensemble_model
            elif which == 'STUDENT':
                checkp = trainer._trainer.ensemble_model.modelStudent
            elif which == 'TEACHER':
                checkp = trainer._trainer.ensemble_model.modelTeacher
            else:
                checkp = trainer._trainer.ensemble_model.modelStudent
        else:
            checkp = trainer._trainer.ensemble_model #model

        DetectionCheckpointer(checkp, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        # Choose which model to evaluate according to the evaluation target parameter.
        eval_tar = cfg.SSL.EVAL_WHO
        if eval_tar == 'STUDENT':
            model = trainer._trainer.ensemble_model.modelStudent
        else:
            model = trainer._trainer.ensemble_model.modelTeacher

        res = Trainer.test(cfg, model, output_dir=args.test_save_dir, caching=True)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    # args = default_argument_parser().parse_args()
    parser = default_argument_parser()
    parser.add_argument("--visualize", action="store_true", help="Visualize the model")
    parser.add_argument(
        "--input", type = str
    )    
    # parser.add_argument(
    #     "--input",
    #     nargs="+",
    #     help="A list of space separated input images; "
    #     "or a single glob pattern such as 'directory/*.jpg'",
    # )    
    parser.add_argument("--output", type=str, default = "./visualizations", help="Output image directory")
    parser.add_argument("--test_save_dir", type=str, default = "./test_results", help="Output image directory")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
