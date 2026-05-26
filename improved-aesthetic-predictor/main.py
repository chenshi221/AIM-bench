import argparse
import csv
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

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


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
    if os.path.isfile(directory):
        ext = os.path.splitext(directory)[1].lower()
        return [directory] if ext in SUPPORTED_FORMATS else []

    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_FORMATS:
                image_files.append(os.path.join(root, file))
    return sorted(image_files)


def predict_aesthetic_score(image_path, clip_model, preprocess, mlp_model, device):
    try:
        pil_image = Image.open(image_path).convert("RGB")
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        embedding = normalized(image_features.cpu().detach().numpy())
        tensor_type = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
        prediction = mlp_model(torch.from_numpy(embedding).to(device).type(tensor_type))
        return prediction.item()
    except Exception as exc:
        print(f"Failed to process {image_path}: {exc}")
        return None


def run_batch(input_path, weights, clip_model_name, output_json, output_csv):
    print("Aesthetic score predictor")
    print(f"Input: {input_path}")

    model = MLP(768)
    state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model.to(device)
    model.eval()

    clip_model, preprocess = clip.load(clip_model_name, device=device)
    image_files = get_image_files(input_path)
    if not image_files:
        print("Error: no supported images were found.")
        return []

    print(f"Found images: {len(image_files)}")
    results = []
    for image_path in tqdm.tqdm(image_files, desc="Aesthetic scoring"):
        score = predict_aesthetic_score(image_path, clip_model, preprocess, model, device)
        if score is not None:
            results.append({"image_path": image_path, "score": round(score, 4)})

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "score"])
        writer.writeheader()
        writer.writerows(results)

    if results:
        scores = [item["score"] for item in results]
        print(f"Processed images: {len(results)}")
        print(f"Average score: {np.mean(scores):.4f}")
        print(f"Max score: {np.max(scores):.4f}")
        print(f"Min score: {np.min(scores):.4f}")
        print(f"Standard deviation: {np.std(scores):.4f}")
    else:
        print("No images were scored successfully.")

    print(f"Saved JSON results to: {output_json}")
    print(f"Saved CSV results to: {output_csv}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Batch aesthetic scoring.")
    parser.add_argument("--input", required=True, help="Image file or directory to score.")
    parser.add_argument("--weights", default="sac+logos+ava1-l14-linearMSE.pth", help="Aesthetic MLP checkpoint path.")
    parser.add_argument("--clip-model", default="ViT-L/14", help="CLIP model name or local path.")
    parser.add_argument("--output-json", default="aesthetic_scores.json", help="JSON output path.")
    parser.add_argument("--output-csv", default="aesthetic_scores.csv", help="CSV output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch(args.input, args.weights, args.clip_model, args.output_json, args.output_csv)
