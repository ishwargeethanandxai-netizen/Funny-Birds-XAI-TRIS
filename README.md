# FunnyBirds-Suppressor

Generates synthetic bird image datasets (with class labels, concept labels and part masks) with different amounts and types of background noise.

## Setup

**1. Python environment**

```bash
conda create -n funnybirds_test python=3.11 -y
conda activate funnybirds_test
pip install numpy scipy pillow requests pyyaml
```

**2. Renderer** (run from the repository root; needs Node.js and Chromium)

```bash
cd Clones/funnybirds
(cd render && PUPPETEER_SKIP_DOWNLOAD=1 npm install)
patch -p1 < ../../render_patch.diff
cd ../..
export PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium    # path to your Chromium/Chrome
```

## Generate

Small test:

```bash
python generate.py --config configs/layered_test.yaml
python validate.py --config configs/layered_test.yaml
```

Full dataset:

```bash
nohup env RENDER_WAIT_MS=600 python -u generate.py --config configs/full_layered.yaml > full_layered.log 2>&1 &
tail -f full_layered.log
python validate.py --config configs/full_layered.yaml
```

Output is written to `datasets/<dataset.name>/`. Re-running the same command resumes an interrupted run.
