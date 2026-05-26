import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


DEFAULT_MODEL_LIST = [
    "Doubao-Seedream-4.0",
    "Doubao-Seededit-3-0-i2i",
    "bagel",
    "Qwen-Image-Edit-Plus",
    "IP2P",
    "Omnigen2",
    "Step-1X",
    "DreamOmni2",
    "Uniworldv2",
    "Flux-kontext-max",
    "Flux-kontext-pro",
    "Flux-kontext-dev",
    "qwen_image_edit_2509_1",
    "qwen_image_edit_2509_lora_1",
]

ORG_MAP = {
    "Doubao-Seedream-4.0": "ByteDance",
    "Doubao-Seededit-3-0-i2i": "ByteDance",
    "Qwen-Image-Edit-Plus": "Alibaba",
    "Qwen-Image-Edit": "Alibaba",
    "Flux-kontext-max": "Black Forest Labs",
    "Flux-kontext-pro": "Black Forest Labs",
    "Flux-kontext-dev": "Black Forest Labs",
    "Step-1X": "StepFun",
    "Uniworldv2": "PKU",
    "DreamOmni2": "CUHK",
    "bagel": "ByteDance",
    "Omnigen2": "BAAI",
    "IP2P": "UCB",
    "qwen_image_edit_2509_1": "Alibaba",
    "qwen_image_edit_2509_lora_1": "Ours",
}

METRIC_BOUNDS = {
    "CLIP-T": [0.20, 0.30],
    "SC": [5.0, 10.0],
    "A": [0.0, 10.0],
    "PQ": [5.0, 10.0],
    "S_GTC": [5.0, 10.0],
    "ACC": [0.0, 100.0],
    "F1": [0.0, 100.0],
    "D_VAD": [2.0, 3.0],
}

HIGHER_IS_BETTER = ["CLIP-T", "SC", "A", "PQ", "S_GTC", "ACC", "F1"]
LOWER_IS_BETTER = ["D_VAD"]
DISPLAY_COLS = ["Method", "Organization", "Overall"] + HIGHER_IS_BETTER + LOWER_IS_BETTER


def load_json_safe(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def get_longclip_score(model_name, fallback_eval_file):
    filepath = os.path.join(model_name, "eval", "longclip_scores.json")
    data = load_json_safe(filepath)
    if data:
        values = [item.get("longclip_similarity_score") for item in data if item.get("longclip_similarity_score") is not None]
        if values:
            return mean_or_zero(values)

    fallback = load_json_safe(fallback_eval_file) or {}
    return fallback.get(model_name, {}).get("longCLIP_score", 0.0)


def get_aesthetic_score(model_name, fallback_eval_file):
    filepath = os.path.join(model_name, "eval", "aesthetic.json")
    data = load_json_safe(filepath)
    if data:
        values = [item.get("score") for item in data if item.get("score") is not None]
        if values:
            return mean_or_zero(values)

    fallback = load_json_safe(fallback_eval_file) or {}
    return fallback.get(model_name, {}).get("aesthetic_score", 0.0)


def get_sc_pq_gtc(model_name):
    filepath = os.path.join(model_name, "eval", "evaluation_scores.json")
    data = load_json_safe(filepath)
    if not data:
        return 0.0, 0.0, 0.0

    sc_vals, pq_vals, gtc_vals = [], [], []
    for item in data:
        if item.get("sc_scores"):
            sc_vals.append(min(item["sc_scores"]))
        if item.get("pq_scores"):
            pq_vals.append(min(item["pq_scores"]))
        if item.get("gtc_scores"):
            gtc_vals.append(min(item["gtc_scores"]))

    return mean_or_zero(sc_vals), mean_or_zero(pq_vals), mean_or_zero(gtc_vals)


def get_vad_distance(model_name):
    metrics_file = os.path.join(model_name, "eval", "vad_metrics.json")
    metrics = load_json_safe(metrics_file)
    if metrics and "vad_avg_distance" in metrics:
        return metrics["vad_avg_distance"]

    pattern = os.path.join(model_name, "eval", "evaluation_vad_results_*.json")
    files = glob.glob(pattern)
    if not files:
        return None

    data = load_json_safe(files[0])
    if not data:
        return None

    distances = []
    for item in data:
        gt_vad = item.get("ground_truth_vad", {})
        pred_vad = item.get("predicted_vad", {})
        required = ["valence", "arousal", "dominance"]
        if "error" in pred_vad:
            continue
        if all(k in gt_vad for k in required) and all(k in pred_vad for k in required):
            gt_vec = np.array([gt_vad["valence"], gt_vad["arousal"], gt_vad["dominance"]])
            pred_vec = np.array([pred_vad["valence"], pred_vad["arousal"], pred_vad["dominance"]])
            distances.append(float(np.linalg.norm(gt_vec - pred_vec)))
    return mean_or_zero(distances) if distances else None


def macro_f1_from_predictions(results):
    emotions = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
    confusion = {target: {pred: 0 for pred in emotions} for target in emotions}
    for item in results:
        target = item.get("target_emotion")
        pred = item.get("predicted_emotion")
        if target in emotions and pred in emotions:
            confusion[target][pred] += 1

    f1_scores = []
    for emotion in emotions:
        tp = confusion[emotion][emotion]
        fp = sum(confusion[other][emotion] for other in emotions) - tp
        fn = sum(confusion[emotion].values()) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return mean_or_zero(f1_scores) * 100


def get_acc_f1(model_name):
    metrics_file = os.path.join(model_name, "eval", "emotion_metrics.json")
    metrics = load_json_safe(metrics_file)
    if metrics:
        return metrics.get("emotion_accuracy", 0.0), metrics.get("macro_f1_score", 0.0)

    results_file = os.path.join(model_name, "eval", "evaluation_results.json")
    data = load_json_safe(results_file)
    if not data:
        return 0.0, 0.0

    valid = [item for item in data if item.get("predicted_emotion") not in ["error", "exception"]]
    if not valid:
        return 0.0, 0.0

    acc = sum(1 for item in valid if item.get("is_correct")) / len(valid) * 100
    f1 = macro_f1_from_predictions(valid)
    return acc, f1


def extract_raw_data(model_list, fallback_eval_file, output_csv):
    print("Extracting raw benchmark metrics...")
    rows = []
    for model in model_list:
        print(f"Processing: {model}")
        sc, pq, gtc = get_sc_pq_gtc(model)
        acc, f1 = get_acc_f1(model)
        vad_distance = get_vad_distance(model)
        rows.append(
            {
                "Method": model,
                "Organization": ORG_MAP.get(model, "Unknown"),
                "CLIP-T": get_longclip_score(model, fallback_eval_file),
                "SC": sc,
                "A": get_aesthetic_score(model, fallback_eval_file),
                "PQ": pq,
                "S_GTC": gtc,
                "ACC": acc,
                "F1": f1,
                "D_VAD": vad_distance if vad_distance is not None else 100.0,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved raw metric CSV to: {output_csv}")
    return df


def normalize_score(value, min_value, max_value, higher_is_better=True):
    value = np.clip(value, min_value, max_value)
    if higher_is_better:
        return (value - min_value) / (max_value - min_value)
    return (max_value - value) / (max_value - min_value)


def build_leaderboard(df, output_csv):
    normalized_cols = []
    for col in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        if col not in df.columns:
            print(f"Warning: missing metric column: {col}")
            continue

        min_bound, max_bound = METRIC_BOUNDS[col]
        norm_col_name = f"{col}_norm"
        higher_is_better = col in HIGHER_IS_BETTER
        df[norm_col_name] = df[col].apply(lambda x: normalize_score(x, min_bound, max_bound, higher_is_better))
        normalized_cols.append(norm_col_name)

    df["Overall"] = df[normalized_cols].mean(axis=1) * 100
    final_df = df.sort_values(by="Overall", ascending=False)
    final_df[DISPLAY_COLS].to_csv(output_csv, index=False)

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n" + "=" * 80)
    print("Model Benchmark Leaderboard")
    print("=" * 80)
    print(final_df[DISPLAY_COLS].to_string(index=False))
    print(f"\nSaved final leaderboard to: {output_csv}")
    return final_df


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate AIM benchmark metrics and compute Overall.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODEL_LIST, help="Model directories to aggregate.")
    parser.add_argument("--fallback-eval-file", default="./eval_results/all_models_eval.json", help="Optional fallback JSON for CLIP-T and aesthetic scores.")
    parser.add_argument("--raw-output", default="raw_benchmark_data.csv", help="Raw metric CSV output path.")
    parser.add_argument("--leaderboard-output", default="final_leaderboard.csv", help="Final leaderboard CSV output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_df = extract_raw_data(args.models, args.fallback_eval_file, args.raw_output)
    build_leaderboard(raw_df, args.leaderboard_output)
