# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import logging
import numpy as np
import time
import weakref
from typing import List, Mapping, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage

from modules.ensemble import EnsembleTSModel

from collections import OrderedDict

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer"]

from detectron2.engine.train_loop import HookBase
import torch.nn.functional as F
from .augmentation import arp_augmentation

import sys
import pdb

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


def calc_cossim_map(feat, coords):
    """
    Args:
        feat: feature map of shape ( num_points, h*w, C )
        coords: query coords ( num_points, 1, 2 )
    
    Return:
        similarity map of shape num_points, h*w
    """
    
    h = w = int(( feat.shape[1] ) ** 0.5)
    
    coords = coords.long()
    pixel_indices = coords[:, 0] * w + coords[:, 1]                        
    pixel_indices = torch.tensor(pixel_indices.reshape(-1, 1))           
    batch_indices = torch.arange(pixel_indices.shape[0]).unsqueeze(1)      
    selected_feature = feat[batch_indices.int(), pixel_indices.int()].squeeze()                  
    
    selected_feature_normalized = torch.nn.functional.normalize(selected_feature, p=2, dim=-1).unsqueeze(1) 
    feat_normalized = torch.nn.functional.normalize(feat, p=2, dim=-1)  # (P, h*w, C)
    similarity_matrix = torch.bmm(selected_feature_normalized, feat_normalized.permute(0, 2, 1)).squeeze(1)  

    return similarity_matrix
            
class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_backward(self):
        for h in self._hooks:
            h.after_backward()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        data_loader_unl,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        async_write_metrics=False,
    ):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            gather_metric_period: an int. Every gather_metric_period iterations
                the metrics are gathered from all the ranks to rank 0 and logged.
            zero_grad_before_forward: whether to zero the gradients before the forward.
            async_write_metrics: bool. If True, then write metrics asynchronously to improve
                training speed
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self.data_loader_unl = data_loader_unl

        self.ensemble_model = EnsembleTSModel(self.model, None)
        # to access the data loader iterator, call `self._data_loader_iter`
        self._data_loader_iter_obj = None
        self._data_loader_unl_iter_obj = None
        self.optimizer = optimizer
        self.gather_metric_period = gather_metric_period
        self.zero_grad_before_forward = zero_grad_before_forward
        self.async_write_metrics = async_write_metrics
        # create a thread pool that can execute non critical logic in run_step asynchronically
        # use only 1 worker so tasks will be executred in order of submitting.
        self.concurrent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        # if self.model.module.do_ssl and self.model.module.iter % self.model.module.ssl_freq == 0 :
        #     data_unl = next(self._data_loader_unl_iter)
        # else: 
        #     data_unl = None
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """
        ouput = self.model(data, return_preds=True)
        import pdb; pdb.set_trace()
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj

    @property
    def _data_loader_unl_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_unl_iter_obj is None:
            self._data_loader_unl_iter_obj = iter(self.data_loader_unl)
        return self._data_loader_unl_iter_obj

    def reset_data_loader(self, data_loader_builder):
        """
        Delete and replace the current data loader with a new one, which will be created
        by calling `data_loader_builder` (without argument).
        """
        del self.data_loader
        data_loader = data_loader_builder()
        self.data_loader = data_loader
        self._data_loader_iter_obj = None

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: Optional[int] = None,
    ) -> None:
        logger = logging.getLogger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                SimpleTrainer.write_metrics(loss_dict, data_time, iter, prefix)
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar(
                "{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter
            )
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def after_train(self):
        super().after_train()
        self.concurrent_executor.shutdown(wait=True)


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        data_loader_unl,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        grad_scaler=None,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
        async_write_metrics=False,
        refiner = False
    ):
        """
        Args:
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward,
                async_write_metrics: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
            precision: torch.dtype as the target precision to cast to in computations
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(
            model, data_loader, data_loader_unl, optimizer, gather_metric_period, zero_grad_before_forward
        )

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler
        self.refiner = refiner 
        self.feat_res = 64
        self.queries = 128
        
    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.refiner is not None:        
            with torch.no_grad():
                for b in range(len(data)):
                    img_l = data[b]["image"].permute(1,2,0).cpu().numpy()       
                    res_h, res_w = data[b]["resized_image_hw"][0], data[b]["resized_image_hw"][1]
                    try :
                        merged_gt_masks = data[b]["instances"].gt_masks.max(dim=0)[0]     # coco : 1024 , 1024 torch.size
                    except : 
                        merged_gt_masks = torch.zeros((img_l.shape[0], img_l.shape[1]), dtype=torch.uint8)
                    
                    random_coords_hw, pl_coords_wh = self.extract_point_gt_rand(merged_gt_masks, num_samples=self.queries, res_h=res_h, res_w=res_w)
                    pl_labels = torch.ones(pl_coords_wh.shape[:-1])
                    self.refiner.set_image(img_l)
                    dec_feat = self.refiner.predict(point_coords = pl_coords_wh ,
                                                    point_labels = pl_labels,
                                                    multimask_output=False, 
                                                    batched_in=True, decfeat_only=True)[-1]

                    dec_sim = calc_cossim_map(dec_feat, random_coords_hw.squeeze())
                    data[b]["sam_dec_feat"] = dec_sim   
                    data[b]["coords_hw"] = random_coords_hw.long()  

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        with autocast(dtype=self.precision):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
    
    def extract_point_gt_rand(self, merged_gt_mask, num_samples=128, res_h=1024, res_w=1024, fres=64):
        nonzero_indices = torch.nonzero(merged_gt_mask)  
        num_gt_samples = num_samples // 2  
        num_rand_samples = num_samples - num_gt_samples  

        if len(nonzero_indices) >= num_gt_samples:
            random_indices = torch.randperm(len(nonzero_indices))[:num_gt_samples]
            sampled_gt_pixels = nonzero_indices[random_indices]  
        else:
            sampled_gt_pixels = nonzero_indices  
            num_rand_samples += num_gt_samples - len(nonzero_indices) 
        random_h = torch.randint(0, res_h, (num_rand_samples,))
        random_w = torch.randint(0, res_w, (num_rand_samples,))
        random_pixels = torch.stack([random_h, random_w], dim=1)  # (num_rand_samples, 2)
        
        random_pixels_hw = torch.cat([sampled_gt_pixels, random_pixels], dim=0)
        random_pixels_wh = random_pixels_hw[:, [1, 0]]
        
        random_pixels_hw = random_pixels_hw.unsqueeze(1).float()  # (num_samples, 1, 2)
        random_pixels_hw[:, :, 0] *= (self.feat_res / res_h)
        random_pixels_hw[:, :, 1] *= (self.feat_res / res_w)
        random_pixels_wh = random_pixels_wh.unsqueeze(1).float()  # (num_samples, 1, 2)

        return random_pixels_hw, random_pixels_wh
        

class SimpleTrainerSSL(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(
        self,
        model,
        model_teacher,
        data_loader,
        data_loader_unl,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        async_write_metrics=False,
        refiner = None
    ):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            gather_metric_period: an int. Every gather_metric_period iterations
                the metrics are gathered from all the ranks to rank 0 and logged.
            zero_grad_before_forward: whether to zero the gradients before the forward.
            async_write_metrics: bool. If True, then write metrics asynchronously to improve
                training speed
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()
        model_teacher.eval()
        for param in model_teacher.parameters():
            param.detach_()
        model_teacher.requires_grad_(False)

        self.model = model
        self.model_teacher = model_teacher

        self.ensemble_model = EnsembleTSModel(self.model_teacher, self.model)

        self.data_loader = data_loader
        self.data_loader_unl = data_loader_unl
        # to access the data loader iterator, call `self._data_loader_iter`
        self._data_loader_iter_obj = None
        self._data_loader_unl_iter_obj = None
        self.optimizer = optimizer
        self.gather_metric_period = gather_metric_period
        self.zero_grad_before_forward = zero_grad_before_forward
        self.async_write_metrics = async_write_metrics
        # create a thread pool that can execute non critical logic in run_step asynchronically
        # use only 1 worker so tasks will be executred in order of submitting.
        self.concurrent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.refiner = refiner

    def update_ema_module(self, module, ema_module, ema_decay):
        # Update parameters.
        module_params = OrderedDict(module.named_parameters())
        ema_module_params = OrderedDict(ema_module.named_parameters())
        
        assert module_params.keys() == ema_module_params.keys()

        for name, param in module_params.items():
            ema_module_params[name].sub_((1. - ema_decay) * (ema_module_params[name] - param))

        # Update buffers.
        module_buffers = OrderedDict(module.named_buffers())
        ema_module_buffers = OrderedDict(ema_module.named_buffers())

        assert module_buffers.keys() == ema_module_buffers.keys()

        for name, buffer in module_buffers.items():
            if buffer.dtype == torch.float32:
                ema_module_buffers[name].sub_((1. - ema_decay) * (ema_module_buffers[name] - buffer))
            else:
                ema_module_buffers[name] = buffer.clone()

    def update_teacher_model(self, ema_decay):
        self.update_ema_module(self.model, self.model_teacher, ema_decay=ema_decay)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        if self.model.module.do_ssl and self.model.module.iter % self.model.module.ssl_freq == 0 :
            data_unl = next(self._data_loader_unl_iter)
        else: 
            data_unl = None
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        Update teacher model weights.
        """
        if self.iter == self.model_teacher.module.burn_in:
            self.update_teacher_model(ema_decay=0.)
        elif self.iter > self.model_teacher.module.burn_in:
            self.update_teacher_model(ema_decay=self.model_teacher.module.ema_decay)

        """
        If you want to do something with the losses, you can wrap the model.
        """

        if self.refiner is not None:
            with torch.no_grad():
                teacher_preds = self.model_teacher(data_unl, return_preds=True)
                teacher_pl = self.model_teacher.module.prepare_ssl_outputs(teacher_preds)
                # teacher_pl = self.model_teacher.module.prepare_ssl_outputs(teacher_preds)
                for b in range(len(teacher_pl)):
                    unl_data = data_unl[b]
                    pl_data = teacher_pl[b]
                    refined_ = []
                    img_ul = unl_data["image_aug"].permute(1,2,0).cpu().numpy()
                    self.refiner.set_image(img_ul)
                    pl_resized = F.interpolate(pl_data["masks"].float().unsqueeze(1), size=(256,256), mode='nearest')  # q , 1, h , w
                    
                    for idx, pl_ in enumerate(pl_resized):
                        masks_, iou_preds_, low_res_masks_  = self.refiner.predict(mask_input = pl_, multimask_output=True)
                        refined_.append(masks_[iou_preds_.argmax(0)])
                        
                    if len(refined_) >= 1: 
                        refined = F.interpolate(torch.stack(refined_).unsqueeze(1), size=(128,256), mode='nearest')
                        teacher_pl[b]["masks"] = refined.squeeze(1)
        else : 
            with torch.no_grad():
                teacher_preds = self.model_teacher(data_unl, return_preds=True)
            teacher_pl = self.model_teacher.module.prepare_ssl_outputs(teacher_preds)

        loss_dict = self.model(data, branch='supervised')
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        
        data_ssl = {'data': data_unl, 'pseudo_label': teacher_pl}
 
        loss_dict_unl = self.model(data_ssl, branch = 'semi-supervised')
        if isinstance(loss_dict_unl, torch.Tensor):
            losses_unl = loss_dict_unl
            loss_dict_unl['unsup_loss'] = loss_dict_unl
        else:
            losses_unl = sum(loss_dict_unl.values())
        loss_dict.update(loss_dict_unl)

        losses_all = losses + losses_unl
        
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses_all.backward()

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj

    @property
    def _data_loader_unl_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_unl_iter_obj is None:
            self._data_loader_unl_iter_obj = iter(self.data_loader_unl)
        return self._data_loader_unl_iter_obj

    def reset_data_loader(self, data_loader_builder):
        """
        Delete and replace the current data loader with a new one, which will be created
        by calling `data_loader_builder` (without argument).
        """
        del self.data_loader
        data_loader = data_loader_builder()
        self.data_loader = data_loader
        self._data_loader_iter_obj = None

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: Optional[int] = None,
    ) -> None:
        logger = logging.getLogger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                SimpleTrainer.write_metrics(loss_dict, data_time, iter, prefix)
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar(
                "{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter
            )
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def after_train(self):
        super().after_train()
        self.concurrent_executor.shutdown(wait=True)

class AMPTrainerSSL(SimpleTrainerSSL):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(
        self,
        model,
        model_teacher,
        data_loader,
        data_loader_unl,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        grad_scaler=None,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
        async_write_metrics=False,
        refiner = None,
        aug = True,
        aug_static=True, 
        num_points = 3,
    ):
        """
        Args:
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward,
                async_write_metrics: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
            precision: torch.dtype as the target precision to cast to in computations
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(
            model, model_teacher, data_loader, data_loader_unl, optimizer, gather_metric_period, zero_grad_before_forward
        )

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler
        self.refiner = refiner
        self.device = refiner.device if refiner else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = num_points
        self.aug = aug
        self.aug_static = aug_static
        
        
    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        if self.model.module.do_ssl and self.model.module.iter % self.model.module.ssl_freq == 0 :
            data_unl = next(self._data_loader_unl_iter)
        else: 
            data_unl = None
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        """
        Update teacher model weights.
        """
        if self.iter == self.model_teacher.module.burn_in:
            self.update_teacher_model(ema_decay=0.)
        elif self.iter > self.model_teacher.module.burn_in:
            self.update_teacher_model(ema_decay=self.model_teacher.module.ema_decay)

        self.im_hw = data_unl[0]['image'].shape[-2:]

        if self.refiner is not None:
            with torch.no_grad():
                teacher_preds = self.model_teacher(data_unl, return_preds=True)
                teacher_pl = self.model_teacher.module.prepare_ssl_outputs(teacher_preds, return_mask_logits=True)    
                for b in range(len(teacher_pl)): 
                    teacher_pl[b]["teacher_masks"] = teacher_pl[b]["masks"].detach().clone()
                    teacher_pl[b]["teacher_mask_logits"] = teacher_pl[b]["mask_logits"].detach().clone()
                    unl_data = data_unl[b]
                    pl_data = teacher_pl[b]
                    if pl_data["masks"].shape[0] == 0:
                        continue
                    
                    mask_logit = pl_data["mask_logits"]
                    probs = F.softmax(mask_logit.flatten(1), dim=1).cpu().numpy()
                    sampled_indices = [ torch.from_numpy(np.random.choice(len(prob), size=self.num_points, p=prob, replace=False)) for prob in probs]
                    sampled_indices = torch.stack(sampled_indices)
                    
                    img_ul = unl_data["image"].permute(1,2,0).cpu().numpy()
                    self.refiner.set_image(img_ul)
                    ratio = img_ul.shape[0] / pl_data["masks"].shape[-2]    
                    pl_h, pl_w = pl_data["masks"].shape[-2:]
                    pl_prompt = torch.stack([sampled_indices % pl_w, sampled_indices // pl_w], dim=2).float() * ratio 
                    pl_labels = torch.ones(pl_prompt.shape[:-1])
                    masks_, iou_preds_, _  = self.refiner.predict(point_coords = pl_prompt, point_labels = pl_labels, multimask_output=True, batched_in=True)
                    highest_iou_ = iou_preds_.argmax(1)
                    highest_iou = highest_iou_.view(masks_.shape[0], 1, 1, 1).expand(masks_.shape[0], 1, masks_.shape[-2], masks_.shape[-1])
                    refined_ = torch.gather(masks_, 1, highest_iou)  
                    
                    original_size_mask = refined_.clone()  
                    teacher_pl[b]["original_size_mask"] = original_size_mask.squeeze(1)
                    
                    if len(refined_) >= 1: 
                        refined = F.interpolate(refined_, size=(pl_h, pl_w), mode='nearest')
                        teacher_pl[b]["masks"] = refined.squeeze(1)
                    
        else :
            with torch.no_grad():
                teacher_preds = self.model_teacher(data_unl, return_preds=True)
            teacher_pl = self.model_teacher.module.prepare_ssl_outputs(teacher_preds, return_mask_logits=True)
            for i in range(len(teacher_pl)):
                resized_mask_logit = F.interpolate(teacher_pl[i]['mask_logits'].unsqueeze(1), size=self.im_hw, mode="bilinear").squeeze(1)
                teacher_pl[i]['original_size_mask'] = (resized_mask_logit > 0).float()
            
        if self.aug:
            data_unl, teacher_pl = arp_augmentation(data_unl, teacher_pl, self.device, dynamic=(not self.aug_static))

        data_ssl = {'data': data_unl, 'pseudo_label': teacher_pl}
        
        del teacher_preds
    
        "Get pseudo-labels from teacher model."
        with autocast(dtype=self.precision):
            loss_dict = self.model(data, branch='supervised')
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
            
        
            loss_dict_unl = self.model(data_ssl, branch = 'semi-supervised')
            if isinstance(loss_dict_unl, torch.Tensor):
                losses_unl = loss_dict_unl
                loss_dict_unl['unsup_loss'] = loss_dict_unl
            else:
                losses_unl = sum(loss_dict_unl.values())
            loss_dict.update(loss_dict_unl)
            losses_all = losses + losses_unl
        
        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.grad_scaler.scale(losses_all).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    def preprocess_(self, x):
        x = x.to(self.refiner.device)
        x = (x.float() - self.model_teacher.module.pixel_mean) / self.model_teacher.module.pixel_std
        x = x / 255.0
        return x
