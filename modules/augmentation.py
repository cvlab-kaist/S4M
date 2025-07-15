import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as transforms

class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class StrongAugmentation:
    def __init__(self, sigma=(0.1, 2.0), image_key="image_wo_weak", out_key="image_aug"):
        self.image_key = image_key
        self.out_key = out_key
        self.transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(sigma)], p=0.5),
        ])

    def __call__(self, data_i):
        img_tensor = data_i[self.image_key]  # (C, H, W)
        img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        img_pil = Image.fromarray(img_np, mode="RGB")
        img_aug = self.transform(img_pil)

        img_aug_tensor = torch.as_tensor(np.ascontiguousarray(img_aug))
        if img_aug_tensor.shape[0] != 3:
            img_aug_tensor = img_aug_tensor.permute(2, 0, 1)

        data_i[self.out_key] = img_aug_tensor
        
        return data_i
    
def src_tgt_indices(batch_size):
    idx = list(range(batch_size))
    random.shuffle(idx)

    n_src = max(1, batch_size // 4)
    src_idices = idx[:n_src]
    tgt_idices = idx[n_src:2 * n_src]

    assert len(src_idices) == len(tgt_idices), "source and target indices should have the same length."
    return src_idices, tgt_idices

def extract_instances(teacher_pl, data_unl, idx, h, w, dynamic, device):
    masks = teacher_pl[idx]['original_size_mask']
    
    overlap = torch.zeros((h, w), dtype=torch.bool, device=device)
    valid_masks, instances_, labels_ = [], [], []
    for j in range(masks.shape[0]):
        mask_ = masks[j] > 0.5
        
        # empty/small masks
        if mask_.sum() < 3000:
            continue
        
        instance_ = mask_ * data_unl[idx]['image_wo_weak'].cuda()

        if dynamic:        
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
            mask_ = padded_mask
            instance_ = padded_instance

        valid_masks.append(mask_)
        instances_.append(instance_)
        labels_.append(teacher_pl[idx]['labels'][j])

        # occluded parts
        overlap = overlap | (mask_.bool())

    return valid_masks, instances_, labels_, overlap
    
def paste_valid_instances(teacher_pl, data_unl, src_idx, valid_tgt_masks, tgt_instances_, tgt_labels_, tgt_overlap, device):
    src_masks = teacher_pl[src_idx]['original_size_mask']
    mask_hw = teacher_pl[0]['masks'].shape[-2:]

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
    empty_mask_indices = [i for i, mask in enumerate(src_masks) if torch.sum(mask) == 0]
    src_masks = torch.stack([mask for i, mask in enumerate(src_masks) if i not in empty_mask_indices])
    teacher_pl[src_idx]['labels'] = torch.stack([label for i, label in enumerate(teacher_pl[src_idx]['labels']) if i not in empty_mask_indices])

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

def arp_augmentation(data_unl, teacher_pl, device, dynamic=False):
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

    src_idices, tgt_idices = src_tgt_indices(batch_size)
    h, w = data_unl[0]['image'].shape[-2:]

    for i in range(len(src_idices)):
        src_idx, tgt_idx = src_idices[i], tgt_idices[i]
        
        if 'original_size_mask' not in teacher_pl[src_idx]:
            teacher_pl[src_idx]['original_size_mask'] = torch.zeros((0, h, w), dtype=torch.float32, device=device)
        if 'original_size_mask' not in teacher_pl[tgt_idx]:
            teacher_pl[tgt_idx]['original_size_mask'] = torch.zeros((0, h, w), dtype=torch.float32, device=device)

        src = extract_instances(teacher_pl, data_unl, src_idx, h, w, dynamic, device)
        tgt = extract_instances(teacher_pl, data_unl, tgt_idx, h, w, dynamic, device)

        if src[0]:  # valid_src_masks not empty
            teacher_pl, data_unl = paste_valid_instances(teacher_pl, data_unl, tgt_idx, *src, device)
        if tgt[0]:  # valid_tgt_masks not empty
            teacher_pl, data_unl = paste_valid_instances(teacher_pl, data_unl, src_idx, *tgt, device)

    strong_aug = StrongAugmentation()
    data_unl = [strong_aug(d) for d in data_unl]

    return data_unl, teacher_pl
