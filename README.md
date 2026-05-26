# AIM Benchmark Evaluation Code

This repository contains the evaluation scripts used to compute the AIM benchmark metrics after a model has produced edited images.

## Installation

Create a Python environment and install the evaluation dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If your machine needs a CUDA-specific PyTorch build, install `torch` and `torchvision` from the official PyTorch selector first, then run `pip install -r requirements.txt`.

`longclip.py` uses the Long-CLIP source adapter included under `LongCLIP/model/`. The LongCLIP-L checkpoint is not committed because it is about 1.71 GB. Download it before computing `CLIP-T`:

```bash
python scripts/setup_longclip.py
```

This creates:

```text
LongCLIP/checkpoints/longclip-L.pt
```

## Expected Directory Layout

Each evaluated model should use the following layout:

```text
MODEL_NAME/
  edited_output/
    fear_cluster_2_fear_04287_original.jpg
    fear_cluster_2_fear_04287_edited_contentment.png
  eval/
```

The metadata file is expected at `Instructions.json` by default. Ground-truth edited images are expected under `benchmark/` by default.

## Using The Benchmark Files

The benchmark is driven by `Instructions.json`. Each record contains:

- `original_image`: the source image filename in `benchmark/`.
- `edited_image`: the benchmark ground-truth edited image filename in `benchmark/`.
- `edit_prompt`: the instruction that should be given to the editing model.
- `target_emotion.emotion`: the target emotion used in output filenames and emotion-based evaluation.
- `target_emotion.VAD`: the ground-truth VAD target used by `vad.py`.

For a full evaluation, create one output folder per model:

```text
MODEL_NAME/
  edited_output/
  eval/
```

For every item in `Instructions.json`, put two files in `MODEL_NAME/edited_output/`:

1. A copy or symlink of the original source image, renamed from:

```text
benchmark/<base>.jpg
```

to:

```text
MODEL_NAME/edited_output/<base>_original.jpg
```

2. The model-edited output image, named as:

```text
MODEL_NAME/edited_output/<base>_edited_<target_emotion>.png
```

Here, `<base>` is `original_image` without the `.jpg` extension, and `<target_emotion>` is `target_emotion.emotion` from the same `Instructions.json` record.

Example:

```json
{
  "original_image": "fear_cluster_2_fear_04287.jpg",
  "edited_image": "fear_cluster_2_fear_04287_contentment_instruction_7.png",
  "target_emotion": {
    "emotion": "contentment"
  }
}
```

Use these model-side filenames:

```text
MODEL_NAME/edited_output/fear_cluster_2_fear_04287_original.jpg
MODEL_NAME/edited_output/fear_cluster_2_fear_04287_edited_contentment.png
```

Do not name model outputs after the benchmark `edited_image` field. That field points to the ground-truth image inside `benchmark/`, and `vie.py` uses it only for the `S_GTC` comparison.

The scripts match these filename patterns:

- Source image for pairwise evaluation: `<base>_original.jpg`
- Model edited image: `<base>_edited_<target_emotion>.png`
- Ground-truth edited image: `benchmark/<edited_image>` from `Instructions.json`

`vie.py` needs both the source image and the model-edited image in `MODEL_NAME/edited_output/`. `longclip.py`, `aesthetic.py`, `emo_predict.py`, and `vad.py` use the model-edited image files; keeping the source image there as well makes the folder compatible with every metric script.

## Metric Coverage

| Paper metric | Script | Main output |
| --- | --- | --- |
| `CLIP-T` | `longclip.py` | `MODEL_NAME/eval/longclip_scores.json` |
| `SC`, `PQ`, `S_GTC` | `vie.py` | `MODEL_NAME/eval/evaluation_scores.json` |
| `A` | `aesthetic.py` | `MODEL_NAME/eval/aesthetic.json` |
| `ACC`, `F1` | `emo_predict.py` | `MODEL_NAME/eval/evaluation_results.json`, `MODEL_NAME/eval/emotion_metrics.json` |
| `D_VAD` | `vad.py` | `MODEL_NAME/eval/vad_metrics.json` |
| `Overall` | `overall_auto.py` | `raw_benchmark_data.csv`, `final_leaderboard.csv` |

## Example Commands

```bash
python vie.py --models Qwen-Image-Edit-Plus --instructions Instructions.json --gt-dir benchmark
python longclip.py --model Qwen-Image-Edit-Plus --instructions Instructions.json
python aesthetic.py --model Qwen-Image-Edit-Plus
python emo_predict.py --model Qwen-Image-Edit-Plus
python vad.py --model Qwen-Image-Edit-Plus --metadata Instructions.json
python overall_auto.py --models Qwen-Image-Edit-Plus
```

API-based scripts read keys from the environment or from a `.env` file. OpenAI-model evaluations use `OPENAI_API_KEY` and default to `https://api.openai.com/v1`. Gemini-model evaluations use `GEMINI_API_KEY` or `GOOGLE_API_KEY` and default to Google's OpenAI-compatible endpoint.

## External Metric Assets

`longclip.py` expects the included LongCLIP source adapter under `LongCLIP/model/` and a checkpoint at `LongCLIP/checkpoints/longclip-L.pt` unless a different checkpoint path is passed with `--checkpoint`. The checkpoint is not committed to this repository; use `python scripts/setup_longclip.py` or download it manually from the LongCLIP-L release.

`aesthetic.py` uses `improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth` by default and OpenAI CLIP `ViT-L/14` through the `clip` package. Pass `--clip-model` if you want to use a local CLIP checkpoint path.
