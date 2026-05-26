import argparse
import json
import os
from warnings import filterwarnings

import clip
import numpy as np
import torch
import torch.nn as nn
import tqdm
from PIL import Image, ImageFile


filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_FORMATS = {".png"}
DEFAULT_WEIGHT_PATH = "./improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"
DEFAULT_CLIP_MODEL = "ViT-L/14"


class MLP(nn.Module):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_image_files(directory):
    image_files = []
    if os.path.isfile(directory):
        if os.path.splitext(directory)[1].lower() in SUPPORTED_FORMATS:
            return [directory]
        print(f"Warning: unsupported image format: {directory}")
        return []

    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_FORMATS:
                image_files.append(os.path.join(root, file))
    return sorted(image_files)


def parse_emotions_from_filename(filename):
    parts = filename.split("_")
    if not parts:
        return "unknown", "unknown"

    original_emo = parts[0]
    target_emo = original_emo

    if "_instruction_" in filename:
        try:
            target_emo = parts[-3]
        except IndexError:
            target_emo = "parse_error"

    return original_emo, target_emo


def predict_aesthetic_score(image_path, clip_model, preprocess, mlp_model, device):
    try:
        pil_image = Image.open(image_path).convert("RGB")
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        tensor_type = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
        prediction = mlp_model(torch.from_numpy(im_emb_arr).to(device).type(tensor_type))
        return prediction.item()
    except Exception as exc:
        print(f"Failed to process {image_path}: {exc}")
        return None


def get_average_aesthetic_score(
    modelname,
    weight_path=DEFAULT_WEIGHT_PATH,
    clip_model_path=DEFAULT_CLIP_MODEL,
):
    img_directory = os.path.join(modelname, "edited_output")

    print("=" * 70)
    print("Aesthetic score evaluation")
    print(f"Model: {modelname}")
    print("=" * 70)

    print("\n[1/4] Loading MLP model...")
    model = MLP(768)
    if not os.path.exists(weight_path):
        print(f"Error: MLP weight file not found: {weight_path}")
        return None
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model.to(device)
    model.eval()

    print("\n[2/4] Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_model_path, device=device)

    print(f"\n[3/4] Scanning directory: {img_directory}")
    image_files = get_image_files(img_directory)
    if not image_files:
        print("Error: no supported PNG images were found.")
        return None
    print(f"Found {len(image_files)} PNG images")

    print("\n[4/4] Running aesthetic scoring...")
    results = []
    for img_path in tqdm.tqdm(image_files, desc="Aesthetic scoring"):
        score = predict_aesthetic_score(img_path, clip_model, preprocess, model, device)
        if score is not None:
            results.append(
                {
                    "name": os.path.basename(img_path),
                    "score": round(score, 4),
                }
            )

    if not results:
        print("No images were scored successfully.")
        return None

    scores = [r["score"] for r in results]
    avg_score = float(np.mean(scores))
    print("\n" + "=" * 70)
    print("Aesthetic scoring complete")
    print("=" * 70)
    print(f"Processed images: {len(results)}")
    print(f"Average score: {avg_score:.4f}")

    eval_dir = os.path.join(modelname, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    output_json = os.path.join(eval_dir, "aesthetic.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to: {output_json}")

    return avg_score


def parse_args():
    parser = argparse.ArgumentParser(description="Compute aesthetic scores for edited images.")
    parser.add_argument("--model", default="Qwen-Image-Edit-Plus", help="Model directory name.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHT_PATH, help="Aesthetic MLP checkpoint path.")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="CLIP model name or local path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    avg = get_average_aesthetic_score(args.model, args.weights, args.clip_model)
    print(f"{args.model} average aesthetic score: {avg}")
