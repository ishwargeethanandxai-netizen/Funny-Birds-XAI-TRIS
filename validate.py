#!/usr/bin/env python3
"""
validate.py - sanity checks and previews for a generated dataset.

    python validate.py --config configs/quick_test.yaml [--samples 60]

Checks (errors make the exit code 1, warnings do not):
  * labels / index / masks agree in length; class balance
  * a part marked missing has an empty mask; how many present parts are hidden by the camera
  * stored masks equal masks re-extracted from the part_map; share of part_map pixels with unknown colour
  * every variant/alpha has all its images; images can be rebuilt bit-exactly from clean image + noise seed
  * fraction of saturated pixels per image set
  * suppressor strength per noise family: correlation across samples between the mean noise INSIDE the bird
    and the mean noise in a ring AROUND it. ~0 for white noise; high values = the background carries
    information about the noise on the bird, i.e. suppressor variables exist.
Writes validation/report.json and PNG sheets (one per variant, plus a mask overlay) into the dataset folder.
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

import config
import scenes
import suppressors

OVERLAY = {'beak': (255, 255, 0), 'eye': (255, 255, 255), 'foot': (255, 0, 0), 'tail': (0, 0, 255),
           'wing': (0, 255, 0), 'body': (150, 150, 150), 'bg_objects': (255, 0, 255)}


def _png(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


def _json(path):
    with open(path) as f:
        return json.load(f)


def check_scenes(split, cfg, P, rng, k):
    index = _json(P.index(split))
    with np.load(P.labels(split)) as z:
        L = {key: z[key] for key in z.files}
    n = len(index)
    res = {'n_scenes': n, 'errors': [], 'warnings': []}

    for key, arr in L.items():
        if len(arr) != n:
            res['errors'].append(f'labels[{key}] has {len(arr)} rows, index has {n}')
    res['class_counts'] = np.bincount(L['class_label']).tolist()
    if len(set(res['class_counts'])) > 1:
        res['warnings'].append(f"classes are not balanced: {res['class_counts']}")

    bad = int(((~L['part_present']) & (L['part_visible_pixels'] > 0)).sum())
    if bad:
        res['errors'].append(f'{bad} parts are marked missing but have mask pixels')
    hidden = int((L['part_present'] & ~L['part_visible']).sum())
    res['present_but_not_visible'] = hidden
    res['present_but_not_visible_frac'] = round(hidden / max(int(L['part_present'].sum()), 1), 4)

    sel = rng.choice(n, size=min(n, k), replace=False)
    mism, unknown, bird, overlap, gap = 0, [], [], 0, []
    for i in sel:
        e = index[i]
        pm = _png(P.part_map(split, e['class_idx'], e['id']))
        fresh = scenes.extract_masks(pm, cfg['masks']['dilate_radius'])
        with np.load(P.masks(split, e['class_idx'], e['id'])) as z:
            stored = {key: z[key] for key in z.files}
        if any(not np.array_equal(fresh[key], stored[key]) for key in scenes.BASE_MASKS):
            mism += 1
        if (stored['bird'] & stored['bg_objects']).any():
            overlap += 1
        if 'bird_smooth' in stored:
            gap.append(float((stored['bird'] & (stored['bird_smooth'] == 0)).sum() / max(stored['bird'].sum(), 1)))
        known = np.logical_or.reduce([stored[p] for p in scenes.PART_COLORS] + [stored['bg_objects'], stored['bg_canvas']])
        unknown.append(1.0 - known.mean())
        bird.append(stored['bird'].mean())
    if mism:
        res['errors'].append(f'{mism}/{len(sel)} stored masks differ from the part_map')
    if overlap:
        res['errors'].append(f'{overlap}/{len(sel)} scenes have bird pixels that are also marked as distractor')
    if gap:
        res['smooth_mask_gap'] = round(float(np.mean(gap)), 5)
        if np.mean(gap) > 0.01:
            res['warnings'].append('>1% of bird pixels have an empty smoothed mask (threshold too high for thin parts?)')
    res['unknown_colour_frac'] = round(float(np.mean(unknown)), 5)
    if np.mean(unknown) > 0.002:
        res['warnings'].append('>0.2% of part_map pixels have an unknown colour (antialiasing or wrong colours?)')
    res['bird_area_frac'] = {'min': round(float(np.min(bird)), 4), 'mean': round(float(np.mean(bird)), 4)}
    return res, index


def ring_correlation(spec, items, size):
    inside, ring_v = [], []
    gap, width = max(1, round(0.01 * size)), max(2, round(0.08 * size))
    for bird, seed in items:
        noise = suppressors.make_noise(spec, (size, size, 3), seed).mean(-1)
        if spec.get('exclude_bird'):
            noise = noise * (~bird)
        near = ndi.binary_dilation(bird, iterations=gap)
        ring = ndi.binary_dilation(bird, iterations=gap + width) & ~near
        if bird.sum() == 0 or ring.sum() == 0:
            continue
        inside.append(noise[bird].mean())
        ring_v.append(noise[ring].mean())
    if len(inside) < 3:
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        c = np.corrcoef(inside, ring_v)[0, 1]
    return 0.0 if np.isnan(c) else round(float(c), 3)


def check_variants(cfg, P, indexes, rng, k):
    sig_cfg = {'source': cfg['signal']['source'], 'sigma_smooth_frac': cfg['signal']['sigma_smooth_frac'],
               'threshold_support': cfg['signal']['threshold_support'], 'size': cfg['render']['size']}
    stats = _json(P.norm_stats)
    size = int(cfg['render']['size'])
    out, errors = [], []
    ref_split = 'train' if 'train' in indexes else next(iter(indexes))
    for spec in cfg['suppressors']:
        sel = {s: rng.choice(len(ix), size=min(len(ix), k), replace=False) for s, ix in indexes.items()}
        items = []
        for i in sel[ref_split]:
            e = indexes[ref_split][i]
            with np.load(P.masks(ref_split, e['class_idx'], e['id'])) as z:
                items.append((z['bird'], e['noise_seed']))
        ring = ring_correlation(spec, items, size)
        for alpha in spec['alphas']:
            row = {'variant': spec['name'], 'alpha': alpha, 'ring_corr': ring,
                   'nominal_snr_db': suppressors.nominal_snr_db(spec, alpha)}
            if row['nominal_snr_db'] is not None and np.isinf(row['nominal_snr_db']):
                row['nominal_snr_db'] = None
            missing, maxdiff, sat = 0, 0, []
            for split, index in indexes.items():
                missing += sum(not os.path.exists(P.image(spec['name'], alpha, split, e['class_idx'], e['id']))
                               for e in index)
                for i in sel[split]:
                    e = index[i]
                    path = P.image(spec['name'], alpha, split, e['class_idx'], e['id'])
                    if not os.path.exists(path):
                        continue
                    img = _png(path)
                    sat.append(float(((img == 0) | (img == 255)).mean()))
                    signal = suppressors.prepare_signal(_png(P.clean(split, e['class_idx'], e['id'])), sig_cfg)
                    with np.load(P.masks(split, e['class_idx'], e['id'])) as z:
                        bird = z['bird']
                    noise = suppressors.make_noise(spec, (size, size, 3), e['noise_seed'])
                    ref = suppressors.compose(signal, spec, alpha, noise, stats, bird)
                    maxdiff = max(maxdiff, int(np.abs(ref.astype(int) - img.astype(int)).max()))
            row.update({'missing_images': missing, 'max_rebuild_diff': maxdiff,
                        'saturated_frac': round(float(np.mean(sat)), 4) if sat else None})
            if missing:
                errors.append(f"{spec['name']} alpha={alpha}: {missing} images missing")
            if maxdiff > 1:
                errors.append(f"{spec['name']} alpha={alpha}: images cannot be rebuilt from clean image + seed "
                              f"(max diff {maxdiff}) - config changed since generation?")
            out.append(row)
    return out, errors


def _thumb(arr, t):
    return Image.fromarray(arr).resize((t, t), Image.BILINEAR)


def _label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 6 * len(text), 12], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 255))


def make_sheets(cfg, P, indexes, cols, thumb=128):
    os.makedirs(P.validation, exist_ok=True)
    split = 'test' if 'test' in indexes else next(iter(indexes))
    index = indexes[split]
    pick = [index[i] for i in np.linspace(0, len(index) - 1, min(cols, len(index))).astype(int)]

    for spec in cfg['suppressors']:
        alphas = sorted(spec['alphas'], reverse=True)
        sheet = Image.new('RGB', (len(pick) * thumb, (len(alphas) + 1) * thumb))
        for c, e in enumerate(pick):
            t = _thumb(_png(P.clean(split, e['class_idx'], e['id'])), thumb)
            _label(t, 'clean')
            sheet.paste(t, (c * thumb, 0))
            for r, a in enumerate(alphas, 1):
                p = P.image(spec['name'], a, split, e['class_idx'], e['id'])
                if os.path.exists(p):
                    t = _thumb(_png(p), thumb)
                    _label(t, f'a={a:g}')
                    sheet.paste(t, (c * thumb, r * thumb))
        sheet.save(os.path.join(P.validation, f"sheet_{spec['name']}.png"))

    sheet = Image.new('RGB', (len(pick) * thumb, 2 * thumb))
    for c, e in enumerate(pick):
        clean = _png(P.clean(split, e['class_idx'], e['id']))
        over = clean.astype(np.float32)
        with np.load(P.masks(split, e['class_idx'], e['id'])) as z:
            for name, col in OVERLAY.items():
                m = z[name]
                over[m] = 0.4 * over[m] + 0.6 * np.array(col, dtype=np.float32)
        sheet.paste(_thumb(clean, thumb), (c * thumb, 0))
        sheet.paste(_thumb(over.astype(np.uint8), thumb), (c * thumb, thumb))
    sheet.save(os.path.join(P.validation, 'masks_overlay.png'))


def main():
    ap = config.add_common_args(argparse.ArgumentParser(description='Validate a generated dataset'))
    ap.add_argument('--samples', type=int, default=60, help='images checked per split / variant')
    ap.add_argument('--cols', type=int, default=6, help='columns in the preview sheets')
    args = ap.parse_args()
    cfg = config.get_config(args)
    P = config.Paths(cfg)
    rng = np.random.default_rng(0)

    report, errors, indexes = {'scenes': {}}, [], {}
    for split in ('train', 'test'):
        if not os.path.exists(P.index(split)):
            continue
        res, index = check_scenes(split, cfg, P, rng, args.samples)
        report['scenes'][split] = res
        indexes[split] = index
        errors += [f'{split}: {e}' for e in res['errors']]
        print(f"[{split}] {res['n_scenes']} scenes | classes {res['class_counts']} | "
              f"bird area {res['bird_area_frac']} | unknown colour {res['unknown_colour_frac']} | "
              f"present-but-hidden parts {res['present_but_not_visible_frac']:.1%}")
        for w in res['warnings']:
            print(f'   warning: {w}')
    if not indexes:
        sys.exit('no scenes found - run generate.py first')

    rows, verr = check_variants(cfg, P, indexes, rng, args.samples)
    errors += verr
    report['variants'] = rows
    print(f"\n{'variant':<22}{'alpha':>7}{'SNR dB':>9}{'ring corr':>11}{'missing':>9}{'rebuild diff':>14}{'saturated':>11}")
    for r in rows:
        snr = '-' if r['nominal_snr_db'] is None else f"{r['nominal_snr_db']:.1f}"
        rc = '-' if r['ring_corr'] is None else f"{r['ring_corr']:.2f}"
        print(f"{r['variant']:<22}{r['alpha']:>7g}{snr:>9}{rc:>11}{r['missing_images']:>9}"
              f"{r['max_rebuild_diff']:>14}{str(r['saturated_frac']):>11}")

    make_sheets(cfg, P, indexes, args.cols)
    report['errors'] = errors
    os.makedirs(P.validation, exist_ok=True)
    with open(os.path.join(P.validation, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nsheets + report.json -> {P.validation}')
    if errors:
        print('\nERRORS:\n  ' + '\n  '.join(errors))
        sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
