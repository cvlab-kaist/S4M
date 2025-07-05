import random
import torch
import os 
import sys
import pdb
import torch.nn.functional as F
import numpy as np

import random
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


def src_tgt_indices(batch_size):
    idx = list(range(batch_size))
    random.shuffle(idx)

    n_src = batch_size // 4
    n_tgt = batch_size // 4

    if n_src == 0 or n_tgt == 0:
        if batch_size >= 2:
            n_src = 1
            n_tgt = 1

    src_idxes = idx[:n_src]
    tgt_idxes = idx[n_src:n_src + n_tgt]
    none_idxes = idx[n_src + n_tgt:]

    return src_idxes, tgt_idxes, none_idxes

def arp_augmentation(data_unl, teacher_pl, device, strong_aug_fn, mask_hw):
        '''
            Args:
                data_unl : list of dict
                                keys: ['file_name', 'image_id', 'width', 'height', 'image', 'image_aug']
                                    image, image_aug: torch.tensor (3, 512, 1024)
                teacher_pl : list of dict
                                keys: ['labels', 'masks']
                                    labels: torch.tensor (Q)
                                    masks: torch.tensor (Q, 128, 256)
        '''
        h, w = data_unl[0]['image'].shape[-2:]
        
        batch_size = len(data_unl)
        assert batch_size > 2, "batch size should be greater than 2 for augmentation."
        
        src_idxes, tgt_idxes, none_idxes = src_tgt_indices(batch_size)
        assert len(src_idxes) == len(tgt_idxes), "source and target indexes should have the same length."
        
        for i in range(len(src_idxes)):
            src_idx, tgt_idx = src_idxes[i], tgt_idxes[i]
            
            if teacher_pl[tgt_idx]['masks'].shape[0] == 0:
                teacher_pl[tgt_idx]['original_size_mask'] = torch.zeros((0, h, w), dtype=torch.float32, device=device)
                
            if teacher_pl[src_idx]['masks'].shape[0] == 0:
                teacher_pl[src_idx]['original_size_mask'] = torch.zeros((0, h, w), dtype=torch.float32, device=device)
                
            tgt_masks = teacher_pl[tgt_idx]['original_size_mask']
            src_masks = teacher_pl[src_idx]['original_size_mask']
                
            src_overlap = torch.zeros((h, w), dtype=torch.bool, device=device)
            tgt_overlap = torch.zeros((h, w), dtype=torch.bool, device=device)
            num_original_tgt_masks = tgt_masks.shape[0]
            num_original_src_masks = src_masks.shape[0]
            valid_src_masks, src_instances_, src_labels_ = [], [], []
            valid_tgt_masks, tgt_instances_, tgt_labels_ = [], [], []
            for j in range(src_masks.shape[0]):
                src_mask_ = src_masks[j] > 0.5
                    
                # empty/small masks
                if src_mask_.sum() < 3000:
                    continue
                
                src_instance_ = src_mask_ * data_unl[src_idx]['image_wo_weak'].cuda()
                
                valid_src_masks.append(src_mask_)
                src_instances_.append(src_instance_)
                src_labels_.append(teacher_pl[src_idx]['labels'][j])
                
                src_overlap = src_overlap | (src_mask_.bool())
                
            for j in range(tgt_masks.shape[0]):
                tgt_mask_ = tgt_masks[j] > 0.5
                
                # empty/small masks
                if tgt_mask_.sum() < 3000:
                    continue
                
                tgt_instance_ = tgt_mask_ * data_unl[tgt_idx]['image_wo_weak'].cuda()
                
                valid_tgt_masks.append(tgt_mask_)
                tgt_instances_.append(tgt_instance_)
                tgt_labels_.append(teacher_pl[tgt_idx]['labels'][j])
                
                tgt_overlap = tgt_overlap | (tgt_mask_.bool())
                
                
            if valid_tgt_masks != []:                
                # extend labels and masks
                teacher_pl[src_idx]['labels'] = torch.cat([teacher_pl[src_idx]['labels'], torch.stack(tgt_labels_).to(device)], dim=0)

                if src_masks.shape[0] == 0:
                    src_masks = torch.stack(valid_tgt_masks)
                else: 
                    src_masks[:, tgt_overlap] = 0
                    src_masks = torch.cat([src_masks.to(device), torch.stack(valid_tgt_masks)], dim=0)

                # remove occluded (overlapped) parts from masks 
                for j in range(len(src_masks)-2, -1, -1):
                    src_masks[j] = torch.where(~torch.any(src_masks[j+1:].bool(), dim=0), src_masks[j], torch.zeros_like(src_masks[j]))
            
                # get indexes of empty masks and remove empty masks and corresponding labels
                empty_mask_indices = [i for i, mask in enumerate(valid_src_masks) if torch.sum(mask) == 0]
                src_masks = torch.stack([mask for i, mask in enumerate(src_masks) if i not in empty_mask_indices])
                teacher_pl[src_idx]['labels'] = torch.stack([label for i, label in enumerate(teacher_pl[src_idx]['labels']) if i not in empty_mask_indices])

                num_original_src_masks -= len(empty_mask_indices)
                
                # update masks 
                teacher_pl[src_idx]['masks'] = torch.stack([
                    F.interpolate(item.unsqueeze(0).unsqueeze(0).float(),
                                mode='nearest',
                                size=(128, 256)
                                ).squeeze(0).squeeze(0) 
                        for item in src_masks
                    ])
                
                # add instances
                for j in range(len(tgt_instances_)):
                    dv = data_unl[src_idx]['image_aug'].device
                    data_unl[src_idx]['image_wo_weak'] = torch.where(valid_tgt_masks[j].bool().to(dv), tgt_instances_[j].to(dv), data_unl[src_idx]['image_wo_weak'])
                
                
            if valid_src_masks != []:
                # teacher_pl, data_unl = paste_valid_instances(teacher_pl, data_unl, tgt_idx, valid_src_masks, src_labels_, None, src_instances_, src_overlap, tgt_masks, device, mask_hw)
                teacher_pl[tgt_idx]['labels'] = torch.cat([teacher_pl[tgt_idx]['labels'], torch.stack(src_labels_).to(device)], dim=0)
                
                if tgt_masks.shape[0] == 0:
                    tgt_masks = torch.stack(valid_src_masks)
                else:
                    tgt_masks[:, src_overlap] = 0 
                    tgt_masks = torch.cat([tgt_masks.to(device), torch.stack(valid_src_masks)], dim=0)
                            
                # remove occluded (overlapped) parts from masks 
                for j in range(len(tgt_masks)-2, -1, -1):
                    tgt_masks[j] = torch.where(~torch.any(tgt_masks[j+1:].bool(), dim=0), tgt_masks[j], torch.zeros_like(tgt_masks[j]))
                
                # get indexes of empty masks and remove empty masks and corresponding labels
                empty_mask_indices = [i for i, mask in enumerate(tgt_masks) if torch.sum(mask) == 0]
                tgt_masks = torch.stack([mask for i, mask in enumerate(tgt_masks) if i not in empty_mask_indices])
                teacher_pl[tgt_idx]['labels'] = torch.stack([label for i, label in enumerate(teacher_pl[tgt_idx]['labels']) if i not in empty_mask_indices])

                num_original_tgt_masks -= len(empty_mask_indices)
                
                # update masks 
                teacher_pl[tgt_idx]['masks'] = torch.stack([
                    F.interpolate(item.unsqueeze(0).unsqueeze(0).float(),
                                mode='nearest',
                                size=(128, 256)
                                ).squeeze(0).squeeze(0) 
                        for item in tgt_masks
                    ])
                
                # add instances
                for j in range(len(src_instances_)):
                    dv = data_unl[tgt_idx]['image_aug'].device
                    data_unl[tgt_idx]['image_wo_weak'] = torch.where(valid_src_masks[j].bool().to(dv), src_instances_[j].to(dv), data_unl[tgt_idx]['image_wo_weak'])
                
                
            # apply strong augmentation
            data_unl[src_idx] = strong_aug_fn(data_unl[src_idx])
            data_unl[tgt_idx] = strong_aug_fn(data_unl[tgt_idx])
        
        for i in none_idxes:
            data_unl[i] = strong_aug_fn(data_unl[i])
        
        return data_unl, teacher_pl
    
    
def arp_augmentation_dynamic(data_unl, teacher_pl, device, strong_aug_fn, mask_hw):
        '''
            Args:
                data_unl : list of dict
                                keys: ['file_name', 'image_id', 'width', 'height', 'image', 'image_aug']
                                    image, image_aug: torch.tensor (3, 512, 1024)
                teacher_pl : list of dict
                                keys: ['labels', 'masks']
                                    labels: torch.tensor (Q)
                                    masks: torch.tensor (Q, 128, 256)
        '''

        batch_size = len(data_unl)
        assert batch_size > 2, "batch size should be greater than 2 for augmentation."
        
        h, w = data_unl[0]['image'].shape[-2:]
        
        # Apply copy and paste aug to half of the batch
        idx = [i for i in range(batch_size)]
        random.shuffle(idx)
        
        quart_b = batch_size//4
        src_idxes = idx[:quart_b]
        tgt_idxes = idx[quart_b:2 * quart_b]
        none_idxes = idx[2 * quart_b:]
        
        assert len(src_idxes) == len(tgt_idxes), "source and target indexes should have the same length."
        
        for i in range(quart_b):
            src_idx, tgt_idx = src_idxes[i], tgt_idxes[i]
            
            if 'original_size_mask' not in teacher_pl[tgt_idx].keys():
                teacher_pl[tgt_idx]['original_size_mask'] = torch.zeros((0, h, w), dtype=torch.float32, device=device)
                
            if 'original_size_mask' not in teacher_pl[src_idx].keys():
                teacher_pl[src_idx]['original_size_mask'] = torch.zeros((0, h, w), dtype=torch.float32, device=device)
            
            tgt_masks = teacher_pl[tgt_idx]['original_size_mask']
            src_masks = teacher_pl[src_idx]['original_size_mask']
        
            src_overlap = torch.zeros((h, w), dtype=torch.bool, device=device)
            tgt_overlap = torch.zeros((h, w), dtype=torch.bool, device=device)
            valid_src_masks, src_instances_, src_labels_ = extract_instances(teacher_pl, data_unl, src_idx, src_masks, h, w, device)
            valid_tgt_masks, tgt_instances_, tgt_labels_ = extract_instances(teacher_pl, data_unl, tgt_idx, tgt_masks, h, w, device)
            
        if valid_tgt_masks != []:
            teacher_pl, data_unl = paste_valid_instances(teacher_pl, data_unl, src_idx, valid_tgt_masks, tgt_labels_, None, tgt_instances_, tgt_overlap, src_masks, device, mask_hw)
        
        if valid_src_masks != []:
            teacher_pl, data_unl = paste_valid_instances(teacher_pl, data_unl, tgt_idx, valid_src_masks, src_labels_, None, src_instances_, src_overlap, tgt_masks, device, mask_hw)
            
        for i in range(len(data_unl)):
            data_unl[i] = strong_aug_fn(data_unl[i])
        
        return data_unl, teacher_pl
    

def extract_instances(teacher_pl, data_unl, idx, masks, h, w, device):
    overlap = torch.zeros((h, w), dtype=torch.bool, device=device)
    valid_masks, instances_, labels_ = [], [], []
    for j in range(masks.shape[0]):
        mask_ = masks[j] > 0.5
        
        # empty/small masks
        if mask_.sum() < 3000:
            continue
        
        instance_ = mask_ * data_unl[idx]['image_wo_weak'].cuda()
        
        # random resize the mask and instance
        scale_factor = np.random.uniform(0.8, 1.2)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized_mask_ = F.interpolate(masks[j].unsqueeze(0).unsqueeze(0), mode='nearest', size=(new_h, new_w)).squeeze(0).squeeze(0)
        resized_instance_ = F.interpolate(instance_.unsqueeze(0).float(), mode='nearest', size=(new_h, new_w)).squeeze(0)
        
        # crop and pad to original size
        x_indices = torch.where(torch.any(resized_mask_, dim=0))[0]
        y_indices = torch.where(torch.any(resized_mask_, dim=1))[0]
        y_min, y_max = y_indices[0].item(), y_indices[-1].item()
        x_min, x_max = x_indices[0].item(), x_indices[-1].item()
        
        cropped_mask = resized_mask_[y_min:y_max+1, x_min:x_max+1]
        cropped_instance = resized_instance_[:, y_min:y_max+1, x_min:x_max+1]
        
        if (y_max - y_min + 1) - h > 0:
            crop_y = np.random.randint(0, (y_max - y_min + 1) - h)
        else:
            crop_y = 0
        if (x_max - x_min + 1) - w > 0:
            crop_x = np.random.randint(0, (x_max - x_min + 1) - w)
        else:
            crop_x = 0
        padded_mask = cropped_mask[crop_y:crop_y + h, crop_x:crop_x + w]
        padded_instance = cropped_instance[:, crop_y:crop_y + h, crop_x:crop_x + w]
        
        if padded_mask.shape[0] < h:
            pad_y = h - padded_mask.shape[0]
            pad_top = np.random.randint(0, pad_y)
            pad_bottom = pad_y - pad_top
            padded_mask = F.pad(padded_mask, (0, 0, pad_top, pad_bottom), mode='constant', value=0)
            padded_instance = F.pad(padded_instance, (0, 0, pad_top, pad_bottom), mode='constant', value=0)
        
        if padded_mask.shape[1] < w:
            pad_x = w - padded_mask.shape[1]
            pad_left = np.random.randint(0, pad_x)
            pad_right = pad_x - pad_left
            padded_mask = F.pad(padded_mask, (pad_left, pad_right, 0, 0), mode='constant', value=0)
            padded_instance = F.pad(padded_instance, (pad_left, pad_right, 0, 0), mode='constant', value=0)                   

        assert padded_mask.shape == (h, w)
        
        valid_masks.append(padded_mask)
        instances_.append(padded_instance)
        labels_.append(teacher_pl[idx]['labels'][j])
    
        # occluded parts
        overlap = overlap | (padded_mask.bool())

    return valid_masks, instances_, labels_
    
def paste_valid_instances(teacher_pl, data_unl, src_idx, valid_tgt_masks, tgt_labels_, tgt_logits_, tgt_instances_, tgt_overlap, src_masks, device, mask_hw):
                    
        # extend labels and masks
        teacher_pl[src_idx]['labels'] = torch.cat([teacher_pl[src_idx]['labels'], torch.stack(tgt_labels_).to(device)], dim=0)
        if src_masks.shape[0] == 0:
            src_masks = torch.stack(valid_tgt_masks)
        else: 
            src_masks[:, tgt_overlap] = 0
            src_masks = torch.cat([src_masks.to(device), torch.stack(valid_tgt_masks)], dim=0)

        # extend mask logits 
        if tgt_logits_ != None:
            if teacher_pl[src_idx]['mask_logits'].shape[0] == 0:
                teacher_pl[src_idx]['mask_logits'] = torch.stack(tgt_logits_).to(device).clone()
            else:
                teacher_pl[src_idx]['mask_logits'] = torch.cat([teacher_pl[src_idx]['mask_logits'], torch.stack(tgt_logits_).to(device)], dim=0)
        
        # remove occluded (overlapped) parts from masks 
        for j in range(len(src_masks)-2, -1, -1):
            src_masks[j] = torch.where(~torch.any(src_masks[j+1:].bool(), dim=0), src_masks[j], torch.zeros_like(src_masks[j]))
        
        if tgt_logits_ != None:
            for j in range(len(teacher_pl[src_idx]['mask_logits'])):
                teacher_pl[src_idx]['mask_logits'][j] = torch.where(~torch.any(F.interpolate(src_masks[j+1:].unsqueeze(1).float(), size=mask_hw, mode='nearest').squeeze(1).bool(), dim=0), 
                                                                    teacher_pl[src_idx]['mask_logits'][j], 
                                                                    torch.ones_like(teacher_pl[src_idx]['mask_logits'][j]) * torch.min(teacher_pl[src_idx]['mask_logits'][j])
                                                                    ) 

        # get indexes of empty masks and remove empty masks and corresponding labels
        empty_mask_indices = [i for i, mask in enumerate(src_masks) if torch.sum(mask) == 0]
        src_masks = torch.stack([mask for i, mask in enumerate(src_masks) if i not in empty_mask_indices])
        teacher_pl[src_idx]['labels'] = torch.stack([label for i, label in enumerate(teacher_pl[src_idx]['labels']) if i not in empty_mask_indices])

        if tgt_logits_ != None:
            # remove empty mask logits
            teacher_pl[src_idx]['mask_logits'] = torch.stack([mask_logits for i, mask_logits in enumerate(teacher_pl[src_idx]['mask_logits']) if i not in empty_mask_indices])
        
        # update masks 
        teacher_pl[src_idx]['masks'] = torch.stack([
            F.interpolate(item.unsqueeze(0).unsqueeze(0).float(),
                        mode='nearest',
                        size=mask_hw
                        ).squeeze(0).squeeze(0) 
                for item in src_masks
            ])
        
        # add instances
        for j in range(len(tgt_instances_)):
            dv = data_unl[src_idx]['image_aug'].device
            data_unl[src_idx]['image_wo_weak'] = torch.where(valid_tgt_masks[j].bool().to(dv), tgt_instances_[j].to(dv), data_unl[src_idx]['image_wo_weak'])
        
        return teacher_pl, data_unl