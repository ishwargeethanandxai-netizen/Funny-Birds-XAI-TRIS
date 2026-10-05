#!/usr/bin/env python3
"""
generate.py - build the FunnyBirds suppressor datasets from a YAML config.

    python generate.py --config configs/quick_test.yaml
    python generate.py --config configs/quick_test.yaml --stage variants        # only re-blend noise
    python generate.py --config my.yaml --export ./export --export-mode hardlink

Stage 1 "scenes"   (slow, needs the render server) - written ONCE, shared by all variants:
    scenes/clean/        clean signal images
    scenes/part_map/     part segmentation maps
    scenes/masks/        binary masks (beak, eye, foot, tail, wing, body, bird, bg_objects, bg_canvas, informative)
    scenes/labels_<split>.npz, index_<split>.json, labels_meta.json, params_<split>.json, norm_stats.json
Stage 2 "variants" (fast, no renderer) - one image folder per suppressor family and alpha:
    variants/<name>/alpha_0.100/<split>/<class>/<id>.png  + variant.json
Optional export: self-contained folder per (variant, alpha) built from hard links.

Everything is resumable: files that already exist are skipped.
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml
from PIL import Image

import config
import render
import scenes
import suppressors


# --------------------------------------------------------------------------- small IO helpers
def _mkdir_for(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_png(arr, path):
    _mkdir_for(path)
    tmp = path + '.tmp'
    Image.fromarray(arr).save(tmp, format='PNG')
    os.replace(tmp, path)


def save_npz(path, **arrays):
    _mkdir_for(path)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def save_json(obj, path):
    _mkdir_for(path)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_png(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


def run_parallel(fn, items, workers, label):
    """Run fn over items with a thread pool, print progress, return the results in order."""
    results, t0, n = [], time.time(), len(items)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for i, res in enumerate(ex.map(fn, items), 1):
            results.append(res)
            if i % 25 == 0 or i == n:
                rate = i / max(time.time() - t0, 1e-9)
                print(f'  [{label}] {i}/{n}  ({rate:.1f}/s)', flush=True)
    return results


# --------------------------------------------------------------------------- setup
def load_or_make_classes(P, parts, cfg):
    d = cfg['dataset']
    if os.path.exists(P.classes) and not d['regen_classes']:
        with open(P.classes) as f:
            classes = json.load(f)
        print(f'[+] loaded {len(classes)} classes from classes.json')
        if len(classes) != d['nr_classes']:
            print(f"    note: config asks for {d['nr_classes']} classes - set dataset.regen_classes=true to change")
        return classes
    classes = scenes.make_classes(d['nr_classes'], parts, np.random.default_rng([d['seed'], 0]))
    save_json(classes, P.classes)
    print(f'[+] created {len(classes)} classes')
    return classes


def load_or_make_params(P, split, per_class, classes, parts, cfg, split_no):
    d = cfg['dataset']
    path = P.params(split)
    if os.path.exists(path) and not d['regen_params']:
        with open(path) as f:
            return json.load(f)
    rng = np.random.default_rng([d['seed'], 1, split_no])
    params = scenes.create_params(split, per_class, classes, parts, d, rng)
    save_json(params, path)
    print(f'[+] created {path} ({len(params)} scenes)')
    return params


# --------------------------------------------------------------------------- stage 1: scenes
def render_one_layered(scene, split, cfg, P, info):
    """Layered signal: smooth the bird alone (H), paste the distractors on top, keep all masks."""
    sid, cls = scene['id'], scene['class_idx']
    paths = [P.clean(split, cls, sid), P.raw(split, cls, sid), P.part_map(split, cls, sid), P.masks(split, cls, sid)]
    if all(os.path.exists(p) for p in paths):
        return 'skipped'

    rcfg, scfg = cfg['render'], cfg['signal']
    sigma = float(scfg['sigma_smooth_frac']) * int(rcfg['size'])
    for _ in range(int(rcfg['attempts'])):
        L = render.render_layers(rcfg, scene['render'])
        D = scenes.distractor_mask(L['scene_pm'])
        bird = scenes.extract_masks(L['bird_pm'], 0)['bird'] & ~D
        if bird.mean() >= rcfg['min_bird_frac']:
            break
    else:
        print(f'  [!] scene {split}/{sid}: bird not visible after {rcfg["attempts"]} renders - dropped')
        return 'failed'

    masks, part_map = scenes.layered_masks(L['bird_pm'], L['scene_pm'], info['informative'],
                                           sigma, scfg['threshold_support'])
    smooth_bird = suppressors.smooth_image(L['bird_img'], sigma, scfg['threshold_support'])
    clean = np.where(D[..., None], L['scene_img'].astype(np.float32) / 255.0, smooth_bird)
    raw = np.where(D[..., None], L['scene_img'], L['bird_img'])

    save_png((clean * 255.0 + 0.5).astype(np.uint8), paths[0])
    save_png(raw.astype(np.uint8), paths[1])
    save_png(part_map, paths[2])
    if cfg['masks']['save_pngs']:
        for name, m in masks.items():
            save_png((m.astype(np.uint8) * 255) if m.dtype == bool else m, P.mask_png(split, cls, sid, name))
    save_npz(paths[3], **masks)                      # written last: marks the scene as complete
    return 'done'


def render_one(scene, split, cfg, P, info):
    if cfg['signal']['source'] == 'layered':
        return render_one_layered(scene, split, cfg, P, info)
    sid, cls = scene['id'], scene['class_idx']
    paths = [P.clean(split, cls, sid), P.part_map(split, cls, sid), P.masks(split, cls, sid)]
    if all(os.path.exists(p) for p in paths):
        return 'skipped'

    rcfg, mcfg = cfg['render'], cfg['masks']
    masks = None
    for _ in range(int(rcfg['attempts'])):
        signal, part_map = render.render_scene(rcfg, scene['render'], cfg['signal']['source'])
        m = scenes.extract_masks(part_map, mcfg['dilate_radius'])
        if m['bird'].mean() >= rcfg['min_bird_frac']:
            masks = m
            break
    if masks is None:
        print(f'  [!] scene {split}/{sid}: bird not visible after {rcfg["attempts"]} renders - dropped')
        return 'failed'

    masks = scenes.finalize_masks(masks, info['informative'])
    save_png(signal, paths[0])
    save_png(part_map, paths[1])
    if mcfg['save_pngs']:
        for name, m in masks.items():
            save_png((m * 255).astype(np.uint8), P.mask_png(split, cls, sid, name))
    save_npz(paths[2], **masks)                      # written last: marks the scene as complete
    return 'done'


def build_index_and_labels(split, params, parts, info, cfg, P):
    """Index + label arrays for all scenes that were rendered successfully."""
    h = w = int(cfg['render']['size'])
    min_vis = float(cfg['masks']['min_visible_frac']) * h * w
    index, rows = [], []
    for sc in params:
        if not os.path.exists(P.masks(split, sc['class_idx'], sc['id'])):
            continue
        with np.load(P.masks(split, sc['class_idx'], sc['id'])) as z:
            vis = scenes.visible_pixels({p: z[p] for p in scenes.PART_ORDER})
        lab = scenes.sample_labels(sc, parts, info)
        lab['part_visible_pixels'] = vis
        lab['part_visible'] = (vis >= min_vis) & lab['part_present']
        lab['class_label'] = np.int64(sc['class_idx'])
        lab['ids'] = np.int64(sc['id'])
        rows.append(lab)
        index.append({'id': sc['id'], 'class_idx': sc['class_idx'], 'noise_seed': sc['noise_seed'],
                      'n_bg_objects': sc['n_bg_objects']})
    if not rows:
        raise RuntimeError(f'no scenes were rendered for split {split}')
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    save_npz(P.labels(split), **arrays)
    save_json(index, P.index(split))
    return index


def stage_scenes(cfg, P, parts, classes, info, splits):
    render.ensure_server(cfg['render'])
    index_by_split = {}
    for split_no, (split, per_class) in enumerate(splits):
        params = load_or_make_params(P, split, per_class, classes, parts, cfg, split_no)
        print(f'\n=== scenes: {split} ({len(params)} scenes) ===')
        res = run_parallel(lambda sc: render_one(sc, split, cfg, P, info), params,
                           cfg['render']['workers'], f'render {split}')
        print(f"  done={res.count('done')} skipped={res.count('skipped')} failed={res.count('failed')}")
        index_by_split[split] = build_index_and_labels(split, params, parts, info, cfg, P)
    return index_by_split


# --------------------------------------------------------------------------- stage 2: variants
def signal_config(cfg):
    s = cfg['signal']
    return {'source': s['source'], 'sigma_smooth_frac': s['sigma_smooth_frac'],
            'threshold_support': s['threshold_support'], 'size': cfg['render']['size']}


def ensure_norm_stats(cfg, P, index_by_split):
    sig_cfg = signal_config(cfg)
    if os.path.exists(P.norm_stats):
        with open(P.norm_stats) as f:
            old = json.load(f)
        if old.get('signal') == sig_cfg:
            return old
    split = 'train' if 'train' in index_by_split else next(iter(index_by_split))
    idx = index_by_split[split]
    take = np.linspace(0, len(idx) - 1, min(len(idx), int(cfg['signal']['stats_samples']))).astype(int)
    sigs = (suppressors.prepare_signal(load_png(P.clean(split, idx[i]['class_idx'], idx[i]['id'])), sig_cfg)
            for i in take)
    stats = suppressors.signal_stats(sigs)
    stats.update({'signal': sig_cfg, 'split': split, 'n_images': int(len(take))})
    save_json(stats, P.norm_stats)
    print(f"[+] signal statistics from {len(take)} {split} images: mean={np.round(stats['mean'], 3).tolist()} "
          f"std={stats['std']:.3f}")
    return stats


def variant_one(entry, split, cfg, P, stats, sig_cfg):
    sid, cls = entry['id'], entry['class_idx']
    specs = cfg['suppressors']
    todo = [(s, a) for s in specs for a in s['alphas'] if not os.path.exists(P.image(s['name'], a, split, cls, sid))]
    if not todo:
        return 0
    size = int(cfg['render']['size'])
    signal = suppressors.prepare_signal(load_png(P.clean(split, cls, sid)), sig_cfg)
    bird = None
    if any(s.get('exclude_bird') for s, _ in todo):
        with np.load(P.masks(split, cls, sid)) as z:
            bird = z['bird']
    noises = {}
    for spec, alpha in todo:
        if spec['name'] not in noises:
            noises[spec['name']] = suppressors.make_noise(spec, (size, size, 3), entry['noise_seed'])
        img = suppressors.compose(signal, spec, alpha, noises[spec['name']], stats, bird)
        save_png(img, P.image(spec['name'], alpha, split, cls, sid))
    return len(todo)


def stage_variants(cfg, P, index_by_split):
    stats = ensure_norm_stats(cfg, P, index_by_split)
    sig_cfg = signal_config(cfg)
    size = int(cfg['render']['size'])
    for spec in cfg['suppressors']:
        suppressors.prepare(spec, (size, size, 3))          # calibrate before threads start
        for alpha in spec['alphas']:
            snr = suppressors.nominal_snr_db(spec, alpha)
            save_json({
                'name': spec['name'], 'type': spec['type'], 'alpha': alpha,
                'mode': 'multiplicative' if spec['type'] in suppressors.MULTIPLICATIVE else 'additive',
                'nominal_snr_db': None if (snr is None or np.isinf(snr)) else round(snr, 2),
                'params': {k: v for k, v in spec.items() if k not in ('alphas', 'name')},
                'signal': sig_cfg, 'norm': {'mean': stats['mean'], 'std': stats['std']},
            }, os.path.join(P.variant_dir(spec['name'], alpha), 'variant.json'))
    for split, index in index_by_split.items():
        print(f'\n=== variants: {split} ({len(index)} scenes x {sum(len(s["alphas"]) for s in cfg["suppressors"])} images) ===')
        res = run_parallel(lambda e: variant_one(e, split, cfg, P, stats, sig_cfg), index,
                           cfg['render']['workers'], f'blend {split}')
        print(f'  wrote {sum(res)} new images')


# --------------------------------------------------------------------------- export
def _link(src, dst, mode):
    _mkdir_for(dst)
    if os.path.exists(dst):
        return
    if mode == 'symlink':
        os.symlink(os.path.abspath(src), dst)
    elif mode == 'hardlink':
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)          # different filesystem
    else:
        shutil.copy2(src, dst)


def _link_tree(src_dir, dst_dir, mode, skip=('variant.json',)):
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith('.tmp') or f in skip:
                continue
            src = os.path.join(root, f)
            _link(src, os.path.join(dst_dir, os.path.relpath(src, src_dir)), mode)


def export(cfg, P, dest, mode):
    """Self-contained folder per (variant, alpha): <dest>/<variant>__alpha_0.100/{scenes,images,variant.json,...}"""
    for spec in cfg['suppressors']:
        for alpha in spec['alphas']:
            out = os.path.join(dest, f"{spec['name']}__{config.alpha_tag(alpha)}")
            vdir = P.variant_dir(spec['name'], alpha)
            _link_tree(vdir, os.path.join(out, 'images'), mode)
            _link(os.path.join(vdir, 'variant.json'), os.path.join(out, 'variant.json'), 'copy')
            _link_tree(P.scenes, os.path.join(out, 'scenes'), mode, skip=())
            for f in (P.classes, P.parts):
                _link(f, os.path.join(out, os.path.basename(f)), mode)
            print(f'[export] {out}')


# --------------------------------------------------------------------------- main
def main():
    ap = config.add_common_args(argparse.ArgumentParser(description='Generate FunnyBirds suppressor datasets'))
    ap.add_argument('--stage', choices=['all', 'scenes', 'variants'], default='all')
    ap.add_argument('--export', type=str, default=None, help='folder for self-contained per-variant exports')
    ap.add_argument('--export-mode', choices=['hardlink', 'symlink', 'copy'], default='hardlink')
    args = ap.parse_args()
    cfg = config.get_config(args)
    P = config.Paths(cfg)
    os.makedirs(P.root, exist_ok=True)

    parts = scenes.load_parts(cfg['dataset']['parts_json'])
    shutil.copyfile(cfg['dataset']['parts_json'], P.parts)
    classes = load_or_make_classes(P, parts, cfg)
    info = scenes.build_label_info(parts, classes, layered=cfg['signal']['source'] == 'layered')
    save_json({
        'parts': info['parts'], 'part_dims': info['part_dims'], 'mask_names': info['mask_names'],
        'concept_names': info['concept_names'], 'attribute_names': info['attribute_names'],
        'attribute_spec': [list(a) for a in info['attribute_spec']],
        'informative_parts': info['informative'],
        'class_to_concept': info['class_to_concept'].tolist(),
    }, P.labels_meta)
    with open(P.config_used, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    d = cfg['dataset']
    splits = [(s, n) for s, n in (('train', d['train_per_class']), ('test', d['test_per_class'])) if n > 0]
    print(f"[+] {len(classes)} classes | informative parts: "
          f"{[p for p, v in info['informative'].items() if v]} | splits: {splits}")

    if args.stage in ('all', 'scenes'):
        index_by_split = stage_scenes(cfg, P, parts, classes, info, splits)
    else:
        index_by_split = {}
        for split, _ in splits:
            with open(P.index(split)) as f:
                index_by_split[split] = json.load(f)
    if args.stage in ('all', 'variants'):
        stage_variants(cfg, P, index_by_split)
    if args.export:
        export(cfg, P, config.resolve_path(args.export), args.export_mode)
    print(f'\n[+] finished -> {P.root}')


if __name__ == '__main__':
    sys.exit(main())
