#!/usr/bin/env python3
"""
fb_dataset.py - standalone PyTorch loader for the generated datasets. Copy this single file into
your evaluation project; it only needs numpy, Pillow and torch.

Two folder layouts are understood:

    # 1) the generated dataset: pick a suppressor variant and an alpha
    ds = FunnyBirdsSuppressor('datasets/FunnyBirds_Suppressor', split='test', variant='corr_s4', alpha=0.1)

    # 2) a self-contained export made with  generate.py --export  (variant and alpha are inside)
    ds = FunnyBirdsSuppressor('export/corr_s4__alpha_0.100', split='test')

    item = ds[0]
    item['image']            (3,H,W) float in [0,1]
    item['class_label']      int
    item['concept_vector']   (26,)  one-hot per part variant, zeros for missing parts
    item['concept_indices']  (5,)   variant index per part, -1 = missing
    item['attribute_vector'] (A,)   factorised concepts (shape/colour ...), see ds.attribute_names
    item['part_present']     (5,)   part exists in the scene
    item['part_visible']     (5,)   part is visible in the image
    item['masks']            (K,H,W) ground-truth masks, order = ds.mask_names (default: beak, eye, foot, tail, wing,
                             body, bird, bg_objects, bg_canvas, informative). Datasets made with signal.source
                             'layered' also have soft smoothed masks '<name>_smooth' (beak ... bird, informative);
                             request them with mask_names=[..., 'bird_smooth'] and list them with ds.all_mask_names
    item['alpha']            signal strength of this dataset
    optional: item['part_map'] (3,H,W) uint8 colours, item['clean'] (3,H,W) noise-free image,
              item['raw'] (3,H,W) unsmoothed composite (layered datasets only)
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def alpha_tag(alpha):
    return f'alpha_{float(alpha):.3f}'


def list_variants(root):
    d = os.path.join(root, 'variants')
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def list_alphas(root, variant):
    d = os.path.join(root, 'variants', variant)
    return sorted(float(a.replace('alpha_', '')) for a in os.listdir(d) if a.startswith('alpha_'))


def _json(path):
    with open(path) as f:
        return json.load(f)


class FunnyBirdsSuppressor(Dataset):
    def __init__(self, root, split='train', variant=None, alpha=None, get_masks=True,
                 get_part_map=False, get_clean=False, get_raw=False, mask_names=None, transform=None):
        self.root, self.split, self.transform = root, split, transform
        self.get_masks, self.get_part_map, self.get_clean, self.get_raw = get_masks, get_part_map, get_clean, get_raw

        self.scenes = os.path.join(root, 'scenes')
        if variant is None:                                   # exported layout
            self.img_dir = os.path.join(root, 'images')
            info_path = os.path.join(root, 'variant.json')
        else:                                                 # generated layout
            if alpha is None:
                raise ValueError('give alpha together with variant')
            vdir = os.path.join(root, 'variants', variant, alpha_tag(alpha))
            self.img_dir, info_path = vdir, os.path.join(vdir, 'variant.json')
        self.variant_info = _json(info_path)
        self.alpha = float(self.variant_info['alpha'])
        self.variant = self.variant_info['name']

        self.index = _json(os.path.join(self.scenes, f'index_{split}.json'))
        with np.load(os.path.join(self.scenes, f'labels_{split}.npz')) as z:
            self.labels = {k: z[k] for k in z.files}
        meta = _json(os.path.join(self.scenes, 'labels_meta.json'))
        self.concept_names = meta['concept_names']
        self.attribute_names = meta['attribute_names']
        self.part_names = meta['parts']
        self.informative_parts = meta['informative_parts']
        self.class_to_concept = torch.tensor(meta['class_to_concept'], dtype=torch.float32)
        self.num_classes = self.class_to_concept.shape[0]
        self.num_concepts = self.class_to_concept.shape[1]
        self.all_mask_names = meta['mask_names']
        self.mask_names = list(mask_names) if mask_names else [m for m in self.all_mask_names if not m.endswith('_smooth')]

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _mask(a):
        """bool -> 0/1, uint8 soft (smoothed) masks -> 0..1 floats."""
        return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)

    def _png(self, path):
        return torch.from_numpy(np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8).copy()).permute(2, 0, 1)

    def __getitem__(self, i):
        e = self.index[i]
        sid, cls = e['id'], e['class_idx']
        name = f'{sid:06d}'
        image = self._png(os.path.join(self.img_dir, self.split, str(cls), name + '.png')).float() / 255.0
        if self.transform is not None:
            image = self.transform(image)
        L = self.labels
        item = {
            'image': image,
            'class_label': torch.tensor(int(L['class_label'][i]), dtype=torch.long),
            'concept_vector': torch.from_numpy(L['concept_vector'][i]),
            'concept_indices': torch.from_numpy(L['concept_indices'][i]),
            'attribute_vector': torch.from_numpy(L['attribute_vector'][i]),
            'part_present': torch.from_numpy(L['part_present'][i].astype(np.float32)),
            'part_visible': torch.from_numpy(L['part_visible'][i].astype(np.float32)),
            'part_visible_pixels': torch.from_numpy(L['part_visible_pixels'][i]),
            'alpha': torch.tensor(self.alpha, dtype=torch.float32),
            'index': i,
            'scene_id': sid,
        }
        if self.get_masks:
            with np.load(os.path.join(self.scenes, 'masks', self.split, str(cls), name + '.npz')) as z:
                item['masks'] = torch.from_numpy(np.stack([self._mask(z[k]) for k in self.mask_names]))
        if self.get_part_map:
            item['part_map'] = self._png(os.path.join(self.scenes, 'part_map', self.split, str(cls), name + '.png'))
        if self.get_clean:
            item['clean'] = self._png(os.path.join(self.scenes, 'clean', self.split, str(cls), name + '.png')).float() / 255.0
        if self.get_raw:
            item['raw'] = self._png(os.path.join(self.scenes, 'raw', self.split, str(cls), name + '.png')).float() / 255.0
        return item

    def class_concepts(self, class_idx):
        """Canonical (full) concept vector of a class."""
        return self.class_to_concept[class_idx]
