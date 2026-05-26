import argparse
import json
import os
import warnings

import torch
from PIL import Image
from tqdm import tqdm

from LongCLIP.longclip_wrapper import LongCLIPWrapper


warnings.filterwarnings("ignore", category=UserWarning)


def build_prompts_database(instruction_file_path):
    print(f"Building prompt database from: {instruction_file_path}")
    try:
        with open(instruction_file_path, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: instruction file not found: {instruction_file_path}")
        return None

    prompts_db = {}
    for task in all_tasks:
        try:
            original_filename = task["original_image"]
            target_emotion = task["target_emotion"]["emotion"]
            base_name, _ = os.path.splitext(original_filename)
            edited_filename = f"{base_name}_edited_{target_emotion}.png"
            prompts_db[edited_filename] = task["edit_prompt"]
        except (KeyError, IndexError):
            continue

    print(f"Prompt database size: {len(prompts_db)}")
    return prompts_db


def evaluate_with_longclip(
    model_name,
    longclip_model_path="./LongCLIP/checkpoints/longclip-L.pt",
    instruction_file="Instructions.json",
):
    edited_images_dir = os.path.join(model_name, "edited_output")
    eval_dir = os.path.join(model_name, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    clip_scores_file = os.path.join(eval_dir, "longclip_scores.json")

    print("=" * 70)
    print("LongCLIP instruction-alignment evaluation")
    print(f"Model: {model_name}")
    print("=" * 70)

    if not os.path.exists(longclip_model_path):
        print(f"Error: LongCLIP checkpoint not found: {longclip_model_path}")
        return None
    if not os.path.isdir(edited_images_dir):
        print(f"Error: edited image directory not found: {edited_images_dir}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        longclip_scorer = LongCLIPWrapper(longclip_model_path, device)
    except Exception as exc:
        print(f"Error: failed to load LongCLIP model: {exc}")
        return None

    prompts_database = build_prompts_database(instruction_file)
    if not prompts_database:
        return None

    image_files = [
        f
        for f in os.listdir(edited_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and "edited" in f
    ]

    all_scores = []
    with torch.no_grad():
        for filename in tqdm(image_files, desc="LongCLIP scoring"):
            prompt_text = prompts_database.get(filename)
            if not prompt_text:
                continue
            image_path = os.path.join(edited_images_dir, filename)
            try:
                image = Image.open(image_path).convert("RGB")
                outputs = longclip_scorer(text=[prompt_text], images=image)
                similarity = (outputs.image_embeds @ outputs.text_embeds.T).squeeze().item()
                all_scores.append(
                    {
                        "edited_filename": filename,
                        "longclip_similarity_score": round(similarity, 4),
                    }
                )
            except Exception as exc:
                print(f"Warning: failed to process {filename}: {exc}")

    if not all_scores:
        print("No valid LongCLIP scores were produced.")
        return None

    sorted_scores = sorted(all_scores, key=lambda x: x["edited_filename"])
    with open(clip_scores_file, "w", encoding="utf-8") as f:
        json.dump(sorted_scores, f, indent=4)
    print(f"Saved LongCLIP results to: {clip_scores_file}")

    scores = [item["longclip_similarity_score"] for item in sorted_scores]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Average LongCLIP similarity score: {avg_score:.4f}")
    return avg_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate instruction alignment with LongCLIP.")
    parser.add_argument("--model", default="Qwen-Image-Edit-Plus", help="Model directory name.")
    parser.add_argument("--checkpoint", default="./LongCLIP/checkpoints/longclip-L.pt", help="LongCLIP checkpoint path.")
    parser.add_argument("--instructions", default="Instructions.json", help="Instruction JSON path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_with_longclip(args.model, args.checkpoint, args.instructions)
