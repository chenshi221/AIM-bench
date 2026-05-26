import argparse
import collections
import json
import os


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
    "qwen_image_edit_2509",
    "qwen_image_edit_2509_lora",
]

POSITIVE_EMOTIONS = {"amusement", "awe", "contentment", "excitement"}
NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}
ALL_EMOTIONS = sorted(POSITIVE_EMOTIONS | NEGATIVE_EMOTIONS)


def calculate_metrics(results, relevant_emotions):
    if not results:
        return {"accuracy": 0.0, "macro_f1_score": 0.0}

    correct_count = sum(1 for result in results if result["is_correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count else 0.0

    confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(int))
    for result in results:
        target = result["target_emotion"]
        predicted = result["predicted_emotion"]
        if target in relevant_emotions:
            confusion_matrix[target][predicted] += 1

    f1_scores = []
    for emotion in relevant_emotions:
        tp = confusion_matrix[emotion].get(emotion, 0)
        fp = sum(confusion_matrix[other].get(emotion, 0) for other in relevant_emotions) - tp
        fn = sum(confusion_matrix[emotion].values()) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)

    macro_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {
        "accuracy": accuracy * 100,
        "macro_f1_score": macro_f1_score * 100,
    }


def analyze_model_results_by_sentiment(model_name):
    results_filepath = os.path.join(model_name, "eval", "evaluation_results.json")

    if not os.path.exists(results_filepath):
        print(f"Warning: missing emotion result file for {model_name}: {results_filepath}")
        return None

    try:
        with open(results_filepath, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error: failed to load emotion result file for {model_name}: {results_filepath}")
        return None

    valid_results = [r for r in all_results if r.get("predicted_emotion") not in ["error", "exception"]]
    positive_results = [r for r in valid_results if r["target_emotion"] in POSITIVE_EMOTIONS]
    negative_results = [r for r in valid_results if r["target_emotion"] in NEGATIVE_EMOTIONS]

    return {
        "positive": calculate_metrics(positive_results, POSITIVE_EMOTIONS),
        "negative": calculate_metrics(negative_results, NEGATIVE_EMOTIONS),
    }


def run_analysis(model_list, output_json=None):
    print("=" * 80)
    print("Emotion-classification metrics by sentiment group")
    print("=" * 80)

    all_models_summary = collections.OrderedDict()
    for model_name in model_list:
        print(f"Processing model: {model_name}")
        summary = analyze_model_results_by_sentiment(model_name)
        if summary:
            all_models_summary[model_name] = summary
            print("  done")
        else:
            all_models_summary[model_name] = {
                "positive": {"accuracy": "N/A", "macro_f1_score": "N/A"},
                "negative": {"accuracy": "N/A", "macro_f1_score": "N/A"},
            }

    max_model_name_len = max(len(name) for name in all_models_summary.keys()) if all_models_summary else 10
    header1 = f"{'Model':<{max_model_name_len}} | {'Positive emotions':^25} | {'Negative emotions':^25}"
    header2 = f"{'':<{max_model_name_len}} | {'ACC (%)':>10} | {'F1 (%)':>10} | {'ACC (%)':>10} | {'F1 (%)':>10}"
    separator = f"{'-' * max_model_name_len}-|-{'-' * 25}-|-{'-' * 25}"

    print("\n" + header1)
    print(header2)
    print(separator)

    for model_name, summary in all_models_summary.items():
        pos_acc = summary["positive"]["accuracy"]
        pos_f1 = summary["positive"]["macro_f1_score"]
        neg_acc = summary["negative"]["accuracy"]
        neg_f1 = summary["negative"]["macro_f1_score"]

        pos_acc_str = f"{pos_acc:.2f}" if isinstance(pos_acc, float) else str(pos_acc)
        pos_f1_str = f"{pos_f1:.2f}" if isinstance(pos_f1, float) else str(pos_f1)
        neg_acc_str = f"{neg_acc:.2f}" if isinstance(neg_acc, float) else str(neg_acc)
        neg_f1_str = f"{neg_f1:.2f}" if isinstance(neg_f1, float) else str(neg_f1)
        print(f"{model_name:<{max_model_name_len}} | {pos_acc_str:>10} | {pos_f1_str:>10} | {neg_acc_str:>10} | {neg_f1_str:>10}")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_models_summary, f, indent=2)
        print(f"\nSaved emotion analysis JSON to: {output_json}")

    return all_models_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate emotion ACC and F1 results.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODEL_LIST, help="Model directories to analyze.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.models, args.output_json)
