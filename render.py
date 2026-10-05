#!/usr/bin/env python3
"""
render.py - client for the FunnyBirds Node/puppeteer render server.

- starts the server if it is not running (and stops it again when Python exits)
- renders the signal image (normal or bird-only) and the part_map of one scene
- renders at size*supersample px and downsamples: area-filter for images, nearest for part maps
  (nearest keeps the exact part colours that the masks are extracted from)
- retries failed requests; safe to call from several threads (render.workers)
"""
import atexit
import io
import os
import signal as _signal
import subprocess
import time
from base64 import b64decode

import numpy as np
import requests
from PIL import Image

from scenes import BG_KEYS

_RS = getattr(Image, 'Resampling', Image)
_server_proc = None


def server_alive(url):
    try:
        requests.get(url, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def _stop_server():
    if _server_proc is not None and _server_proc.poll() is None:
        try:
            os.killpg(os.getpgid(_server_proc.pid), _signal.SIGTERM)
        except (ProcessLookupError, PermissionError, AttributeError):
            _server_proc.terminate()


def ensure_server(rcfg):
    """Make sure the render server answers; start it when allowed."""
    global _server_proc
    if server_alive(rcfg['server_url']):
        return
    if not rcfg['autostart_server']:
        raise RuntimeError(f"render server not reachable at {rcfg['server_url']} and autostart_server is false")

    rdir = rcfg['render_dir']
    if not os.path.isdir(os.path.join(rdir, 'node_modules')):
        raise RuntimeError(f'{rdir}/node_modules not found - run "npm install" in that folder')
    if not os.path.isfile(os.path.join(rdir, 'js', 'three.js-master', 'examples', 'data', 'XAI', 'bird01.glb')):
        raise RuntimeError(f'bird models not found in {rdir}/js - restore them: git checkout -- render/js')

    env = os.environ.copy()
    if rcfg.get('chromium_path'):
        env['PUPPETEER_EXECUTABLE_PATH'] = rcfg['chromium_path']
    log = open(os.path.join(rdir, 'server.log'), 'a')
    _server_proc = subprocess.Popen(['node', 'server.js'], cwd=rdir, env=env, stdout=log, stderr=log,
                                    start_new_session=True)
    atexit.register(_stop_server)
    t0 = time.time()
    while time.time() - t0 < 40:
        if server_alive(rcfg['server_url']):
            print(f"[render] server started (pid {_server_proc.pid}) on {rcfg['server_url']}")
            return
        if _server_proc.poll() is not None:
            raise RuntimeError(f'render server exited immediately - see {rdir}/server.log')
        time.sleep(0.5)
    raise RuntimeError(f'render server did not come up within 40 s - see {rdir}/server.log')


def _url(rcfg, params, mode):
    q = {'render_mode': mode, 'size': int(rcfg['size']), 'scale': int(rcfg['supersample'])}
    q.update(params)
    # values are plain numbers / names / comma lists, so no URL-encoding is needed (matches the original code)
    return rcfg['server_url'].rstrip('/') + '/render?' + '&'.join(f'{k}={v}' for k, v in q.items())


def _fetch(rcfg, params, mode):
    url = _url(rcfg, params, mode)
    size = int(rcfg['size'])
    last = None
    for attempt in range(int(rcfg['max_retries'])):
        try:
            r = requests.get(url, timeout=rcfg['timeout'])
            r.raise_for_status()
            img = Image.open(io.BytesIO(b64decode(r.content))).convert('RGB')
            if img.size != (size, size):
                img = img.resize((size, size), _RS.NEAREST if mode == 'part_map' else _RS.BOX)
            return np.asarray(img, dtype=np.uint8)
        except Exception as e:  # network error, bad image, server hiccup
            last = e
            time.sleep(1.0 + attempt)
            try:
                ensure_server(rcfg)
            except Exception:
                pass
    raise RuntimeError(f'render request failed after {rcfg["max_retries"]} tries: {last}')


def render_scene(rcfg, params, source='default'):
    """Returns (signal uint8 HxWx3, part_map uint8 HxWx3) for one scene.

    source='default'    bird + 3D distractor objects on the blue FunnyBirds background
    source='foreground' bird only on black; the part_map is then rendered without distractors too,
                        so the bg_objects mask stays consistent with the image
    """
    signal = _fetch(rcfg, params, 'default' if source == 'default' else 'foreground')
    pm_params = params if source == 'default' else {**params, **{k: '' for k in BG_KEYS}}
    part_map = _fetch(rcfg, pm_params, 'part_map')
    return signal, part_map


def render_layers(rcfg, params):
    """Four renders for the 'layered' signal (see generate.py). All uint8 HxWx3.

    bird_img   bird only, black background           -> smoothed with H, becomes the signal
    bird_pm    part map of the bird only (no distractors, nothing occluding it)
    scene_img  bird + distractors on a black background -> only the distractor pixels are used
    scene_pm   part map of bird + distractors        -> tells which pixels show a distractor
    """
    blank = {**params, **{k: '' for k in BG_KEYS}}
    return {
        'bird_img': _fetch(rcfg, params, 'foreground'),
        'bird_pm': _fetch(rcfg, blank, 'part_map'),
        'scene_img': _fetch(rcfg, params, 'black'),
        'scene_pm': _fetch(rcfg, params, 'part_map'),
    }
