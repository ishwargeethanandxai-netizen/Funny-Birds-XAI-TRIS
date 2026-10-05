#!/usr/bin/env python3
"""
config.py - YAML configuration, defaults, CLI overrides and shared path helpers.

Every setting has a default below, so a YAML file only needs the keys you want to change.
Relative paths are resolved against the folder that contains this file (the project root),
so scripts work from any working directory. ${ENV_VARS} and ~ are expanded.

Command-line overrides (work in generate.py and validate.py):
    python generate.py --config configs/quick_test.yaml --set render.size=128 dataset.seed=3
"""
import argparse
import copy
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TWO_PI = 6.283185307179586

DEFAULTS = {
    'dataset': {
        'name': 'FunnyBirds_Suppressor',
        'out_dir': './datasets',
        'parts_json': './Clones/funnybirds/render/parts.json',
        'seed': 0,
        'nr_classes': 10,
        'train_per_class': 50,
        'test_per_class': 10,
        'min_bg_parts': 0,            # 3D distractor objects per image: min .. max-1
        'max_bg_parts': 35,
        'drop_parts_in_train': True,  # FunnyBirds behaviour: half of the train birds lose random parts
        'regen_classes': False,       # True: overwrite classes.json
        'regen_params': False,        # True: overwrite params_<split>.json (new random scenes)
        'camera': {
            'distance_range': [200, 400],
            'pitch_range': [0.0, TWO_PI],
            'roll_range': [0.0, TWO_PI],
        },
    },
    'render': {
        'server_url': 'http://localhost:8081',
        'render_dir': './Clones/funnybirds/render',
        'chromium_path': None,        # e.g. /usr/bin/chromium ; None = puppeteer default
        'autostart_server': True,
        'size': 256,                  # final image side length in px
        'supersample': 2,             # render at size*supersample, then downscale
        'workers': 1,                 # parallel render requests
        'timeout': 120,
        'max_retries': 5,             # HTTP retries per request
        'attempts': 3,                # re-renders when the bird is (almost) invisible
        'min_bird_frac': 0.002,       # minimum bird area (fraction of image) to accept a render
    },
    'signal': {
        'source': 'default',          # default = bird + distractors on blue | foreground = bird only on black |
                                      # layered = smooth the bird alone (H), then paste the distractors, then add noise
        'sigma_smooth_frac': 0.006,   # filter H (XAI-TRIS): sigma as fraction of image size (0.006*256 = 1.5 px)
        'threshold_support': 0.05,
        'stats_samples': 500,         # train images used to estimate signal mean/std
    },
    'masks': {
        'dilate_radius': 0,           # part maps are rendered without antialiasing, so 0 is exact
        'min_visible_frac': 0.0002,   # a part counts as "visible" above this fraction of image area
        'save_pngs': False,           # also write one PNG per mask
    },
    'suppressors': [
        {'type': 'corr', 'sigma_frac': 0.04, 'alphas': [0.1, 0.3, 0.6, 1.0]},
    ],
}


def deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def resolve_path(p):
    if p is None:
        return None
    p = os.path.expanduser(os.path.expandvars(str(p)))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(PROJECT_ROOT, p))


def alpha_tag(alpha):
    """Folder name for an alpha value. Must match fb_dataset.py."""
    return f'alpha_{float(alpha):.3f}'


def _set_dotted(cfg, dotted, value):
    keys = dotted.split('.')
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _finalise(cfg):
    import suppressors  # local import: suppressors.py does not import config

    d, r, s = cfg['dataset'], cfg['render'], cfg['signal']
    d['out_dir'] = resolve_path(d['out_dir'])
    d['parts_json'] = resolve_path(d['parts_json'])
    r['render_dir'] = resolve_path(r['render_dir'])
    r['chromium_path'] = resolve_path(r['chromium_path'])

    if s['source'] not in ('default', 'foreground', 'layered'):
        raise ValueError("signal.source must be 'default', 'foreground' or 'layered'")
    if s['source'] == 'layered' and cfg['masks']['dilate_radius'] != 0:
        raise ValueError("signal.source 'layered' needs masks.dilate_radius: 0")
    if int(r['size']) < 16 or int(r['supersample']) < 1:
        raise ValueError('render.size must be >= 16 and render.supersample >= 1')
    if d['max_bg_parts'] < d['min_bg_parts']:
        raise ValueError('dataset.max_bg_parts must be >= dataset.min_bg_parts')

    if not cfg['suppressors']:
        raise ValueError('config needs at least one entry under "suppressors"')
    names = set()
    for spec in cfg['suppressors']:
        if spec.get('type') not in suppressors.FAMILIES:
            raise ValueError(f"unknown suppressor type {spec.get('type')!r}; "
                             f"choose from {sorted(suppressors.FAMILIES)}")
        alphas = [float(a) for a in spec.get('alphas', [])]
        if not alphas or any(not 0.0 < a <= 1.0 for a in alphas):
            raise ValueError(f"suppressor {spec['type']}: 'alphas' must be a non-empty list in (0, 1]")
        spec['alphas'] = alphas
        if spec['type'] == 'natural':
            if 'dir' not in spec:
                raise ValueError("suppressor type 'natural' needs a 'dir' with background images")
            spec['dir'] = resolve_path(spec['dir'])
        spec['name'] = spec.get('name') or suppressors.default_name(spec)
        if spec['name'] in names:
            raise ValueError(f"duplicate suppressor name {spec['name']!r}; give one a unique 'name'")
        names.add(spec['name'])
    return cfg


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = path if os.path.exists(path) else resolve_path(path)
        with open(p, 'r') as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, value = item.partition('=')
        _set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return _finalise(cfg)


def add_common_args(parser):
    parser.add_argument('--config', type=str, default=None, help='YAML config (defaults are in config.py)')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='override config values, e.g. render.size=128 dataset.seed=1')
    return parser


def get_config(args):
    return load_config(args.config, args.set)


class Paths:
    """All file locations of one dataset, in one place."""

    def __init__(self, cfg):
        self.root = os.path.join(cfg['dataset']['out_dir'], cfg['dataset']['name'])
        self.scenes = os.path.join(self.root, 'scenes')
        self.variants = os.path.join(self.root, 'variants')
        self.classes = os.path.join(self.root, 'classes.json')
        self.parts = os.path.join(self.root, 'parts.json')
        self.labels_meta = os.path.join(self.scenes, 'labels_meta.json')
        self.norm_stats = os.path.join(self.scenes, 'norm_stats.json')
        self.config_used = os.path.join(self.root, 'config_used.yaml')
        self.validation = os.path.join(self.root, 'validation')

    @staticmethod
    def _f(sid):
        return f'{int(sid):06d}'

    def params(self, split):
        return os.path.join(self.scenes, f'params_{split}.json')

    def index(self, split):
        return os.path.join(self.scenes, f'index_{split}.json')

    def labels(self, split):
        return os.path.join(self.scenes, f'labels_{split}.npz')

    def clean(self, split, cls, sid):
        return os.path.join(self.scenes, 'clean', split, str(cls), self._f(sid) + '.png')

    def raw(self, split, cls, sid):
        """Unsmoothed composite (layered mode only)."""
        return os.path.join(self.scenes, 'raw', split, str(cls), self._f(sid) + '.png')

    def part_map(self, split, cls, sid):
        return os.path.join(self.scenes, 'part_map', split, str(cls), self._f(sid) + '.png')

    def masks(self, split, cls, sid):
        return os.path.join(self.scenes, 'masks', split, str(cls), self._f(sid) + '.npz')

    def mask_png(self, split, cls, sid, name):
        return os.path.join(self.scenes, 'masks', split, str(cls), f'{self._f(sid)}_{name}.png')

    def variant_dir(self, variant, alpha):
        return os.path.join(self.variants, variant, alpha_tag(alpha))

    def image(self, variant, alpha, split, cls, sid):
        return os.path.join(self.variant_dir(variant, alpha), split, str(cls), self._f(sid) + '.png')


if __name__ == '__main__':
    ap = add_common_args(argparse.ArgumentParser(description='Print the resolved configuration'))
    import json
    print(json.dumps(get_config(ap.parse_args()), indent=2))
