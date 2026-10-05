#!/usr/bin/env python3
"""
scenes.py - everything about the content of a scene: classes, random scene parameters,
concept labels (several formats) and binary part masks.

Label formats produced for every image (all derived from what is actually in the scene):
    concept_vector       one-hot over all part variants, concatenated (26 for the default parts.json);
                         all zeros for a part that was dropped from the bird
    concept_indices      one integer per part (index of the variant, -1 if the part is missing)
    attribute_vector     factorised concepts, e.g. "wing:model=wing01.glb", "wing:color=green"
                         (only attributes with >= 2 different values)
    part_present         bool per part: is the part part of the bird in this scene
    part_visible_pixels  masked pixel count per part (a part can be present but hidden by the camera angle)
    part_visible         bool per part: pixel count above masks.min_visible_frac
and per dataset: class_to_concept matrix (canonical concepts of each class) and the informative parts
(parts whose variant differs between classes - only those carry class information).

Masks (bool, H x W): beak, eye, foot, tail, wing, body, bird, bg_objects, bg_canvas, informative.
"""
import json
import math

import numpy as np
import scipy.ndimage as ndi

PART_ORDER = ['beak', 'eye', 'foot', 'tail', 'wing']

# Exact RGB colours used by the FunnyBirds part_map render mode
PART_COLORS = {
    'beak': [(255, 255, 0)],
    'eye': [(255, 255, 253), (255, 255, 254)],
    'foot': [(255, 0, 1), (255, 0, 2)],
    'tail': [(0, 0, 255)],
    'wing': [(0, 255, 1), (0, 255, 2)],
    'body': [(170, 170, 170)],
}
BASE_MASKS = ['beak', 'eye', 'foot', 'tail', 'wing', 'body', 'bird', 'bg_objects', 'bg_canvas']
MASK_NAMES = BASE_MASKS + ['informative']
SMOOTH_BASE = ['beak', 'eye', 'foot', 'tail', 'wing', 'body', 'bird', 'informative']
SMOOTH_NAMES = [n + '_smooth' for n in SMOOTH_BASE]

BG_COLORS = ['red', 'green', 'blue', 'yellow']
BG_KEYS = ['bg_objects', 'bg_radius', 'bg_pitch', 'bg_roll', 'bg_scale_x', 'bg_scale_y', 'bg_scale_z',
           'bg_rot_x', 'bg_rot_y', 'bg_rot_z', 'bg_color']


# --------------------------------------------------------------------------- classes and scenes
def load_parts(path):
    with open(path, 'r') as f:
        parts = json.load(f)
    missing = [p for p in PART_ORDER if p not in parts]
    if missing:
        raise KeyError(f'parts.json is missing parts: {missing}')
    return parts


def part_dims(parts):
    return {p: len(parts[p]) for p in PART_ORDER}


def make_classes(nr_classes, parts, rng):
    """Unique random combinations of one variant per part."""
    dims = part_dims(parts)
    if nr_classes > math.prod(dims.values()):
        raise ValueError('more classes requested than distinct part combinations exist')
    seen, classes = set(), []
    while len(classes) < nr_classes:
        combo = tuple(int(rng.integers(0, dims[p])) for p in PART_ORDER)
        if combo in seen:
            continue
        seen.add(combo)
        classes.append({'class_idx': len(classes), 'parts': dict(zip(PART_ORDER, combo))})
    return classes


def informative_parts(classes):
    return {p: len({c['parts'][p] for c in classes}) > 1 for p in PART_ORDER}


def _join(values):
    return ''.join(f'{v},' for v in values)


def sample_scene(class_info, parts, split, dcfg, rng):
    """Random camera, light, dropped parts and 3D distractor objects for one image."""
    cam = dcfg['camera']
    r = {
        'camera_distance': int(rng.integers(cam['distance_range'][0], cam['distance_range'][1] + 1)),
        'camera_pitch': float(rng.uniform(*cam['pitch_range'])),
        'camera_roll': float(rng.uniform(*cam['roll_range'])),
        'light_distance': 300,
        'light_pitch': float(rng.uniform(0, 2 * math.pi)),
        'light_roll': float(rng.uniform(0, 2 * math.pi)),
    }
    dropped = set()
    if split == 'train' and dcfg['drop_parts_in_train'] and rng.random() < 0.5:
        n_del = int(rng.integers(0, len(PART_ORDER) + 1))
        dropped = {PART_ORDER[i] for i in rng.choice(len(PART_ORDER), n_del, replace=False)}
    present = {}
    for p in PART_ORDER:
        present[p] = p not in dropped
        for key, val in parts[p][class_info['parts'][p]].items():
            r[f'{p}_{key}'] = val if present[p] else 'placeholder'

    lo, hi = int(dcfg['min_bg_parts']), int(dcfg['max_bg_parts'])
    n_bg = int(rng.integers(lo, hi)) if hi > lo else lo
    u = lambda a, b: [float(x) for x in rng.uniform(a, b, n_bg)]
    two_pi = 2 * math.pi
    r['bg_objects'] = _join(int(x) for x in rng.integers(0, 5, n_bg))
    r['bg_radius'] = _join(int(x) for x in rng.integers(100, 201, n_bg))
    r['bg_pitch'] = _join(u(0, two_pi))
    r['bg_roll'] = _join(u(0, two_pi))
    r['bg_scale_x'] = _join(int(x) for x in rng.integers(5, 21, n_bg))
    r['bg_scale_y'] = _join(int(x) for x in rng.integers(5, 21, n_bg))
    r['bg_scale_z'] = _join(int(x) for x in rng.integers(5, 21, n_bg))
    r['bg_rot_x'] = _join(u(0, two_pi))
    r['bg_rot_y'] = _join(u(0, two_pi))
    r['bg_rot_z'] = _join(u(0, two_pi))
    r['bg_color'] = _join(BG_COLORS[int(i)] for i in rng.integers(0, 4, n_bg))

    return {'class_idx': class_info['class_idx'], 'render': r, 'present': present,
            'n_bg_objects': n_bg, 'noise_seed': int(rng.integers(0, 2 ** 31 - 1))}


def create_params(split, per_class, classes, parts, dcfg, rng):
    out = []
    for cls in classes:
        for _ in range(per_class):
            sc = sample_scene(cls, parts, split, dcfg, rng)
            sc['id'] = len(out)
            out.append(sc)
    return out


# --------------------------------------------------------------------------- label formats
def concept_names(parts):
    names = []
    for p in PART_ORDER:
        for i, inst in enumerate(parts[p]):
            names.append(f'{p}{i}_' + '_'.join(str(v).replace('.glb', '') for v in inst.values()))
    return names


def attribute_spec(parts):
    """Factorised concepts: list of (part, attribute, value) for attributes with >= 2 values."""
    spec = []
    for p in PART_ORDER:
        for key in sorted({k for inst in parts[p] for k in inst}):
            vals = []
            for inst in parts[p]:
                if key in inst and inst[key] not in vals:
                    vals.append(inst[key])
            if len(vals) >= 2:
                spec += [(p, key, v) for v in vals]
    return spec


def build_label_info(parts, classes, layered=False):
    dims = part_dims(parts)
    spec = attribute_spec(parts)
    c2c = np.zeros((len(classes), sum(dims.values())), dtype=np.float32)
    for c in classes:
        off = 0
        for p in PART_ORDER:
            c2c[c['class_idx'], off + c['parts'][p]] = 1.0
            off += dims[p]
    return {
        'parts': PART_ORDER,
        'part_dims': dims,
        'concept_names': concept_names(parts),
        'attribute_spec': spec,
        'attribute_names': [f"{p}:{k}={str(v).replace('.glb', '')}" for p, k, v in spec],
        'informative': informative_parts(classes),
        'class_to_concept': c2c,
        'mask_names': MASK_NAMES + (SMOOTH_NAMES if layered else []),
    }


def sample_labels(scene, parts, info):
    dims = info['part_dims']
    present = np.array([scene['present'][p] for p in PART_ORDER], dtype=bool)
    # recover the variant index from the class (stored in the class_to_concept matrix row)
    row = info['class_to_concept'][scene['class_idx']]
    off, indices, vec = 0, [], np.zeros(len(row), dtype=np.float32)
    for j, p in enumerate(PART_ORDER):
        idx = int(np.argmax(row[off:off + dims[p]]))
        if present[j]:
            vec[off + idx] = 1.0
            indices.append(idx)
        else:
            indices.append(-1)
        off += dims[p]
    attr = np.zeros(len(info['attribute_spec']), dtype=np.float32)
    for a, (p, key, val) in enumerate(info['attribute_spec']):
        j = PART_ORDER.index(p)
        if present[j] and parts[p][indices[j]].get(key) == val:
            attr[a] = 1.0
    return {'concept_vector': vec, 'concept_indices': np.array(indices, dtype=np.int64),
            'attribute_vector': attr, 'part_present': present}


# --------------------------------------------------------------------------- masks
def extract_masks(part_map, dilate_radius=0):
    """Binary masks from a part_map render (H, W, 3 uint8). Colours must match exactly."""
    arr = np.asarray(part_map)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    struct = None
    if dilate_radius > 0:
        struct = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), int(dilate_radius))

    masks = {}
    for name, colors in PART_COLORS.items():
        m = np.zeros(r.shape, dtype=bool)
        for (cr, cg, cb) in colors:
            m |= (r == cr) & (g == cg) & (b == cb)
        if struct is not None and m.any():
            m = ndi.binary_dilation(m, structure=struct)
        masks[name] = m
    masks['bird'] = np.logical_or.reduce([masks[p] for p in PART_COLORS])
    bg = (r == 204) & (g == 204) & (b >= 204)       # 0xcccccc + object index
    if struct is not None and bg.any():
        bg = ndi.binary_dilation(bg, structure=struct)
    masks['bg_objects'] = bg
    masks['bg_canvas'] = (r == 0) & (g == 0) & (b == 0)
    return masks


def finalize_masks(masks, informative):
    """Add the 'informative' mask: union of the masks of parts that differ between classes."""
    inf = np.zeros_like(masks['bird'])
    for p, is_inf in informative.items():
        if is_inf:
            inf |= masks[p]
    out = dict(masks)
    out['informative'] = inf
    return out


def visible_pixels(masks):
    return np.array([int(masks[p].sum()) for p in PART_ORDER], dtype=np.int64)


# --------------------------------------------------------------------------- layered signal masks
def distractor_mask(scene_pm):
    """Pixels that show a 3D distractor object in the part map of bird + distractors."""
    arr = np.asarray(scene_pm)
    return (arr[..., 0] == 204) & (arr[..., 1] == 204) & (arr[..., 2] >= 204)


def _blur(mask, sigma_px):
    m = mask.astype(np.float32)
    return m if sigma_px <= 0 else ndi.gaussian_filter(m, sigma=float(sigma_px))


def layered_masks(bird_pm, scene_pm, informative, sigma_px, threshold):
    """Masks of the layered signal: smooth the bird, then paste the distractors on top.

    bird_pm    part map of the bird alone       (nothing occludes it, defines the bird)
    scene_pm   part map of bird + distractors   (defines which pixels a distractor covers)
    Returns (masks, part_map):
      masks['beak'] ... masks['bird']   unsmoothed, VISIBLE in the final image (bool)
      masks['bg_objects']               distractor pixels (bool)
      masks['bg_canvas']                empty background (bool)
      masks['informative']              union of the informative parts (bool)
      masks['<name>_smooth']            same regions after smoothing with the width of H, uint8 0..255
                                        (soft mask, 0 outside the support of the smoothed bird and where a
                                        distractor covers the pixel; the part masks add up to bird_smooth)
      part_map                          composite part map of the final image (uint8)
    """
    D = distractor_mask(scene_pm)
    amodal = extract_masks(bird_pm, 0)
    free = ~D
    names = PART_ORDER + ['body']
    masks = {n: amodal[n] & free for n in names}
    masks['bird'] = amodal['bird'] & free
    masks['bg_objects'] = D
    masks['bg_canvas'] = amodal['bg_canvas'] & free
    inf = np.zeros_like(D)
    for p in PART_ORDER:
        if informative[p]:
            inf |= masks[p]
    masks['informative'] = inf

    # Smoothed masks: Gaussian blur of the binary mask with the width of H. The support is where the smoothed
    # BIRD mask is >= threshold; all masks share it, so the soft part masks add up to the soft bird mask.
    bird_blur = _blur(amodal['bird'], sigma_px)
    support = (bird_blur >= float(threshold)) * free
    soft = {n: _blur(amodal[n], sigma_px) * support for n in names}
    soft['bird'] = bird_blur * support
    soft['informative'] = np.clip(sum((soft[p] for p in PART_ORDER if informative[p]),
                                      np.zeros(D.shape, np.float32)), 0.0, 1.0)
    for n in SMOOTH_BASE:
        masks[n + '_smooth'] = np.round(soft[n] * 255.0).astype(np.uint8)
    part_map = np.where(D[..., None], np.asarray(scene_pm), np.asarray(bird_pm)).astype(np.uint8)
    return masks, part_map
