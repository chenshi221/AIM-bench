import argparse
import collections
import json
import os

import numpy as np


DEFAULT_MODEL_LIST = [
    "Doubao-Seedream-4.0",
    "Doubao-Seededit-3-0-i2i",
    "bagel",
    "Qwen-Image-Edit-Plus",
    "Qwen-Image-Edit",
    "IP2P",
    "Omnigen2",
    "Step-1X",
    "DreamOmni2",
    "Uniworldv2",
    "Flux-kontext-max",
    "Flux-kontext-pro",
    "Flux-kontext-dev",
]

DEFAULT_VLM_MODEL_NAME = "gpt-4o"
POSITIVE_EMOTIONS = {"amusement", "awe", "contentment", "excitement"}
NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}


def calculate_vad_distance(result_entry):
    if "predicted_vad" not in result_entry or "error" in result_entry["predicted_vad"]:
        return None

    gt_vad = result_entry.get("ground_truth_vad", {})
    pred_vad = result_entry.get("predicted_vad", {})

    required = ["valence", "arousal", "dominance"]
    if not all(k in gt_vad for k in required) or not all(k in pred_vad for k in required):
        return None

    gt_vec = np.array([gt_vad["valence"], gt_vad["arousal"], gt_vad["dominance"]])
    pred_vec = np.array([pred_vad["valence"], pred_vad["arousal"], pred_vad["dominance"]])
    return float(np.linalg.norm(gt_vec - pred_vec))


def analyze_model_results(model_name, vlm_model_name):
    results_filename = f"evaluation_vad_results_{vlm_model_name.replace('/', '_')}.json"
    results_filepath = os.path.join(model_name, "eval", results_filename)

    if not os.path.exists(results_filepath):
        print(f"Warning: missing VAD result file for {model_name}: {results_filepath}")
        return None

    try:
        with open(results_filepath, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error: failed to load VAD result file for {model_name}: {results_filepath}")
        return None

    distances = {"positive": [], "negative": []}
    for entry in results_data:
        emotion = entry.get("target_emotion")
        distance = calculate_vad_distance(entry)
        if emotion and distance is not None:
            if emotion in POSITIVE_EMOTIONS:
                distances["positive"].append(distance)
            elif emotion in NEGATIVE_EMOTIONS:
                distances["negative"].append(distance)

    return {
        "positive": float(np.mean(distances["positive"])) if distances["positive"] else 0.0,
        "negative": float(np.mean(distances["negative"])) if distances["negative"] else 0.0,
    }


def run_analysis(model_list, vlm_model_name, output_json=None):
    print("=" * 80)
    print("VAD distance by sentiment group")
    print("=" * 80)

    all_models_summary = collections.OrderedDict()
    for model_name in model_list:
        print(f"Processing model: {model_name}")
        summary = analyze_model_results(model_name, vlm_model_name)
        if summary:
            all_models_summary[model_name] = summary
            print(f"  done. positive={summary['positive']:.4f}, negative={summary['negative']:.4f}")
        else:
            all_models_summary[model_name] = {"positive": "N/A", "negative": "N/A"}

    max_model_name_len = max(len(name) for name in all_models_summary.keys()) if all_models_summary else 10
    header = f"{'Model':<{max_model_name_len}}   {'Positive':<10} {'Negative':<10}"
    print("\n" + header)
    print("-" * len(header))

    for model_name, summary in all_models_summary.items():
        pos_dist = summary["positive"]
        neg_dist = summary["negative"]
        pos_str = f"{pos_dist:.4f}" if isinstance(pos_dist, float) else str(pos_dist)
        neg_str = f"{neg_dist:.4f}" if isinstance(neg_dist, float) else str(neg_dist)
        print(f"{model_name:<{max_model_name_len}}   {pos_str:<10} {neg_str:<10}")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_models_summary, f, indent=2)
        print(f"\nSaved VAD analysis JSON to: {output_json}")

    return all_models_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate VAD distance results.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODEL_LIST, help="Model directories to analyze.")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL_NAME, help="VAD VLM model name used in result filenames.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.models, args.vlm_model, args.output_json)
