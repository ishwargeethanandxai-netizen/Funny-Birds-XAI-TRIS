#!/usr/bin/env python3
"""
verify_masks.py - look at the masks on top of the image, one figure per mask.

Each figure has one COLUMN per class (one example bird per class) and three ROWS:
    row 1  the image, zoomed in on the bird
    row 2  the SHARP mask drawn in purple on that image
    row 3  the SMOOTHED mask in purple (the stronger the purple, the higher the mask value),
           with the border of the sharp mask in white, so the soft halo can be compared

Every tile is cut out around the bird and enlarged by a whole number, so each block you see is exactly
one pixel of the dataset. For the part masks (beak, eye, ...) the example of each class is a bird where
that part is visible.

    python verify_masks.py --all-variants --alpha 0.75       # clean set + one folder per variant (nearest alpha if 0.75 is missing)
    python verify_masks.py                                   # all masks, clean image
    python verify_masks.py --only beak,wing                  # only these masks
    python verify_masks.py --source noisy --variant corr_s4 --alpha 0.75   # one variant on its noisy image

The masks are stored once per scene and are identical for every variant; only the noise differs. So the
clean figures are made once, and each variant folder shows the same masks on that variant's noisy image.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fb_dataset import FunnyBirdsSuppressor, list_alphas, list_variants

PROJECT = os.path.dirname(os.path.abspath(__file__))
PARTS = ['beak', 'eye', 'foot', 'tail', 'wing', 'body']
NAMES = PARTS + ['bird', 'informative']
PURPLE = np.array([170, 40, 255], dtype=np.float32) / 255.0
TILE, GAP, LEFT, TOP = 256, 8, 190, 46


def font(size):
    for p in ('DejaVuSans.ttf', '/usr/share/fonts/TTF/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tint(img, strength, alpha=0.75):
    """Blend purple into the image where `strength` (0..1) is high."""
    a = (alpha * strength)[..., None]
    return img * (1 - a) + PURPLE * a


def border(mask):
    """One-pixel inner border of a binary mask."""
    m = mask > 0.5
    p = np.pad(m, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def crop_box(region, size, margin=0.4, min_side=48):
    """Square box around the bird (region = boolean mask), clipped to the image."""
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return 0, size, 0, size
    side = int(min(size, max(min_side, max(np.ptp(ys) + 1, np.ptp(xs) + 1) * (1 + 2 * margin))))
    cy, cx = (ys.max() + ys.min()) // 2, (xs.max() + xs.min()) // 2
    y0 = int(np.clip(cy - side // 2, 0, size - side))
    x0 = int(np.clip(cx - side // 2, 0, size - side))
    return y0, y0 + side, x0, x0 + side


def to_tile(arr, box):
    """Crop, enlarge by a whole number (no blur, no distortion), centre on a black square tile."""
    y0, y1, x0, x1 = box
    crop = (np.clip(arr[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)
    scale = max(1, TILE // crop.shape[0])
    big = Image.fromarray(crop).resize((crop.shape[1] * scale, crop.shape[0] * scale), Image.NEAREST)
    tile = Image.new('RGB', (TILE, TILE), (0, 0, 0))
    tile.paste(big, ((TILE - big.width) // 2, (TILE - big.height) // 2))
    return tile


def choose(ds, cls, name):
    """Index of a typical sample of class `cls` in which `name` is visible (median visible size)."""
    L = ds.labels
    idx = np.where(L['class_label'] == cls)[0]
    if name in ds.part_names:
        j = ds.part_names.index(name)
        vis = idx[L['part_visible'][idx, j]]
        if len(vis):
            order = vis[np.argsort(L['part_visible_pixels'][vis, j])]
            return int(order[len(order) // 2]), True
        return int(idx[len(idx) // 2]), False
    return int(idx[len(idx) // 2]), True


def make_figures(ds, names, out_dir, source, split, note=''):
    """Write one figure per mask name into out_dir."""
    mi = {n: i for i, n in enumerate(ds.mask_names)}
    classes = sorted(set(int(c) for c in ds.labels['class_label']))
    os.makedirs(out_dir, exist_ok=True)
    f_big, f_small = font(20), font(15)
    rows = [('image', ''), ('sharp mask', '(purple = mask)'), ('smoothed mask', '(purple = value,\nwhite = sharp border)')]

    for name in names:
        W = LEFT + len(classes) * (TILE + GAP)
        H = TOP + 3 * (TILE + GAP)
        canvas = Image.new('RGB', (W, H), (18, 18, 24))
        d = ImageDraw.Draw(canvas)
        d.text((10, 2), f'mask: {name}', fill=(255, 255, 255), font=f_big)
        d.text((10, 26), note.replace(', ', '\n'), fill=(190, 190, 200), font=f_small)
        for r, (t1, t2) in enumerate(rows):
            y = TOP + r * (TILE + GAP)
            d.text((10, y + TILE // 2 - 22), t1, fill=(255, 255, 255), font=f_big)
            d.text((10, y + TILE // 2 + 4), t2, fill=(190, 190, 200), font=f_small)
        for c, cls in enumerate(classes):
            i, visible = choose(ds, cls, name)
            it = ds[i]
            img = np.transpose(np.asarray(it['clean'] if source == 'clean' else it['image']), (1, 2, 0))
            M = np.asarray(it['masks'])
            sharp, soft = M[mi[name]], M[mi[name + '_smooth']]
            box = crop_box((M[mi['bird']] > 0.5) | (M[mi['bird_smooth']] > 0), img.shape[0])

            sm = tint(img, soft)
            sm[border(sharp)] = 1.0
            x = LEFT + c * (TILE + GAP)
            for r, arr in enumerate([img, tint(img, sharp), sm]):
                canvas.paste(to_tile(arr, box), (x, TOP + r * (TILE + GAP)))
            d.text((x, TOP - 24), f'class {cls}  (scene {it["scene_id"]})', fill=(255, 255, 255), font=f_small)
            if not visible:
                d.text((x + 8, TOP + 8), f'{name} not visible\nin any {split} bird\nof this class', fill=(255, 120, 120), font=f_small)
        canvas.save(os.path.join(out_dir, f'{name}.png'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default=os.path.join(PROJECT, 'datasets', 'full_layered_v1'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--all-variants', action='store_true',
                    help='clean figures once + one folder per variant on its noisy image (uses --alpha, or the nearest alpha)')
    ap.add_argument('--source', choices=['clean', 'noisy'], default='clean',
                    help='single-variant mode: clean = noise-free signal image, noisy = image of --variant/--alpha')
    ap.add_argument('--variant', default='corr_s4')
    ap.add_argument('--alpha', type=float, default=0.75)
    ap.add_argument('--only', default=None, help='comma separated mask names, default: all')
    ap.add_argument('--out', default=os.path.join(PROJECT, 'verification'))
    a = ap.parse_args()

    names = a.only.split(',') if a.only else NAMES
    mask_names = sorted(set(names + ['bird', 'bird_smooth'] + [n + '_smooth' for n in names]))
    root = os.path.join(a.out, os.path.basename(os.path.normpath(a.dataset)))

    def load(variant, alpha):
        return FunnyBirdsSuppressor(a.dataset, split=a.split, variant=variant, alpha=alpha, get_clean=True,
                                    mask_names=mask_names)

    if a.all_variants:
        jobs = []
        for v in list_variants(a.dataset):
            alphas = list_alphas(a.dataset, v)
            alpha = min(alphas, key=lambda x: abs(x - a.alpha))
            jobs.append((v, alpha))
        v0, a0 = jobs[0]
        make_figures(load(v0, a0), names, os.path.join(root, 'clean'), 'clean', a.split, 'clean (no noise)')
        print(f'clean figures -> {os.path.join(root, "clean")}')
        for v, alpha in jobs:
            tag = '' if abs(alpha - a.alpha) < 1e-9 else f'  (alpha {a.alpha:g} not available, nearest used)'
            out = os.path.join(root, f'{v}__alpha_{alpha:.3f}')
            make_figures(load(v, alpha), names, out, 'noisy', a.split, f'{v}, alpha {alpha:g}')
            print(f'{v:16s} alpha {alpha:<5g} -> {out}{tag}')
    else:
        ds = load(a.variant, a.alpha)
        if a.source == 'clean':
            out, note = os.path.join(root, 'clean'), 'clean (no noise)'
        else:
            out, note = os.path.join(root, f'{a.variant}__alpha_{a.alpha:.3f}'), f'{a.variant}, alpha {a.alpha:g}'
        make_figures(ds, names, out, a.source, a.split, note)
        print(f'saved {len(names)} figures to {out}')


if __name__ == '__main__':
    main()