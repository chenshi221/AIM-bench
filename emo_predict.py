import argparse
import base64
import collections
import json
import os
import re
import time
import concurrent.futures

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

EMOTIONS_LIST = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
]
FILE_REGEX = re.compile(r".*_edited_(" + "|".join(EMOTIONS_LIST) + r")\.png$")

load_dotenv()
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
GEMINI_API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODEL = "qwen_image_edit_2509_lora"
DEFAULT_API_MODEL = "gemini-2.5-pro"
MAX_CONCURRENT_REQUESTS = 40
SAVE_CHECKPOINT_INTERVAL = 20
MAX_RETRIES = 3
RETRY_DELAY = 2

PROMPT_TEXT = """
# ROLE
You are an expert in visual psychology and affective computing, specializing in visual sentiment analysis.

# TASK
Your task is to analyze the core emotion conveyed by the given image. You must classify the image into one of the eight predefined emotion categories below.

# EMOTION CATEGORIES
- amusement
- anger
- awe
- contentment
- disgust
- excitement
- fear
- sadness

# ANALYSIS GUIDELINES
- Objectively analyze the image's content: subjects, actions, facial expressions, environment, colors, and composition.
- Choose the single best-fitting emotion from the provided list.
- If the image is emotionally neutral or ambiguous, choose the closest emotion and explain the ambiguity in your reasoning.
- Your reasoning must directly connect specific visual elements from the image to your chosen emotion.

# OUTPUT FORMAT
You MUST provide your response strictly in the following JSON format. Do not include any introductory text, explanations, or markdown formatting outside of the JSON block itself.

{
  "emotion": "<chosen_emotion_string>",
  "confidence": <score_float_from_0.0_to_1.0>,
  "reasoning": "A concise analysis directly linking specific visual elements to the chosen emotion."
}
"""


def get_api_config(api_model_name):
    if api_model_name.lower().startswith("gemini"):
        for env_var in GEMINI_API_KEY_ENV_VARS:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key, GEMINI_BASE_URL, env_var
        return None, GEMINI_BASE_URL, "GEMINI_API_KEY or GOOGLE_API_KEY"
    return os.getenv(OPENAI_API_KEY_ENV_VAR), OPENAI_BASE_URL, OPENAI_API_KEY_ENV_VAR


def get_image_files_and_emotions(directory):
    matched_files = []
    for filename in sorted(os.listdir(directory)):
        match = FILE_REGEX.match(filename)
        if match:
            matched_files.append({"filename": filename, "target_emotion": match.group(1)})
    return matched_files


def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            from io import BytesIO

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return None


def validate_response(data):
    if not isinstance(data, dict):
        return False, "Response is not a dictionary"
    if "error" in data:
        return False, data["error"]
    required = ["emotion", "confidence", "reasoning"]
    missing = [k for k in required if k not in data]
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"
    if data["emotion"] not in EMOTIONS_LIST:
        return False, f"Invalid emotion: {data['emotion']}"
    try:
        if not (0.0 <= float(data["confidence"]) <= 1.0):
            return False, f"Confidence out of range: {data['confidence']}"
    except (ValueError, TypeError):
        return False, f"Invalid confidence format: {data['confidence']}"
    return True, None


def analyze_image_emotion(client, image_path, api_model_name, retry_count=0):
    encoded_data = encode_image_to_base64(image_path)
    if not encoded_data:
        return {"error": f"Failed to encode image: {image_path}"}

    try:
        response = client.chat.completions.create(
            model=api_model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_TEXT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_data}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in response", content, 0)

        parsed_data = json.loads(json_match.group(0))
        is_valid, error_msg = validate_response(parsed_data)
        if not is_valid:
            raise ValueError(f"Invalid response structure: {error_msg}")
        return parsed_data

    except Exception as exc:
        if retry_count < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return analyze_image_emotion(client, image_path, api_model_name, retry_count + 1)
        return {"error": f"{type(exc).__name__}: {exc} (after {MAX_RETRIES} retries)"}


def save_results(all_data, filepath):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4)
    except Exception as exc:
        print(f"Error: failed to save {filepath}: {exc}")


def process_single_image(client, image_info, api_model_name, image_dir):
    filename = image_info["filename"]
    target_emotion = image_info["target_emotion"]
    image_path = os.path.join(image_dir, filename)

    try:
        if not os.path.exists(image_path):
            analysis_result = {"error": "File not found"}
        else:
            analysis_result = analyze_image_emotion(client, image_path, api_model_name)

        result = {
            "image_name": filename,
            "target_emotion": target_emotion,
        }

        if "error" in analysis_result:
            result.update(
                {
                    "predicted_emotion": "error",
                    "is_correct": False,
                    "details": analysis_result,
                }
            )
            return {"status": "error", "data": result}

        predicted = analysis_result.get("emotion")
        result.update(
            {
                "predicted_emotion": predicted,
                "is_correct": target_emotion == predicted,
                "details": analysis_result,
            }
        )
        return {"status": "success", "data": result}

    except Exception as exc:
        return {
            "status": "error",
            "data": {
                "image_name": filename,
                "target_emotion": target_emotion,
                "predicted_emotion": "exception",
                "is_correct": False,
                "details": {"error": str(exc)},
            },
        }


def macro_f1_score(valid_results):
    confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(int))
    for item in valid_results:
        confusion_matrix[item["target_emotion"]][item["predicted_emotion"]] += 1

    f1_scores = []
    for emotion in EMOTIONS_LIST:
        tp = confusion_matrix[emotion].get(emotion, 0)
        fp = sum(confusion_matrix[other].get(emotion, 0) for other in EMOTIONS_LIST) - tp
        fn = sum(confusion_matrix[emotion].values()) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def calculate_and_save_accuracy(results_filepath, summary_filepath, metrics_filepath, api_model_name):
    print("\n" + "=" * 70)
    print("Generating emotion-classification report")
    print("=" * 70)

    try:
        with open(results_filepath, "r", encoding="utf-8") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: failed to load results file: {results_filepath}")
        return

    valid_results = [r for r in results if r.get("predicted_emotion") not in ["error", "exception"]]
    if not valid_results:
        print("Error: no valid results are available for reporting.")
        return

    total_correct = sum(1 for r in valid_results if r["is_correct"])
    total_processed = len(valid_results)
    overall_accuracy = (total_correct / total_processed) * 100 if total_processed else 0.0
    f1 = macro_f1_score(valid_results) * 100

    emotion_stats = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(int))

    for item in valid_results:
        target = item["target_emotion"]
        predicted = item["predicted_emotion"]
        emotion_stats[target]["total"] += 1
        confusion_matrix[target][predicted] += 1
        if item["is_correct"]:
            emotion_stats[target]["correct"] += 1

    report = [
        "=" * 70,
        f"Emotion-classification evaluation report ({api_model_name})",
        "=" * 70,
        f"Valid images: {total_processed}",
        f"Correct predictions: {total_correct}",
        f"Overall accuracy: {overall_accuracy:.2f}%",
        f"Macro F1: {f1:.2f}%",
        "\nPer-emotion accuracy:",
        "-" * 70,
    ]

    for emotion in EMOTIONS_LIST:
        stats = emotion_stats[emotion]
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] else 0.0
        report.append(f"- {emotion:<12}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

    report.extend(["\nConfusion matrix (rows: target, columns: prediction)", "=" * 70])
    header = f"{'':<12}" + "".join([f"{emo[:4]:>6}" for emo in EMOTIONS_LIST])
    report.append(header)
    report.append("-" * len(header))

    for target_emo in EMOTIONS_LIST:
        row = f"{target_emo:<12}"
        for pred_emo in EMOTIONS_LIST:
            row += f"{confusion_matrix[target_emo][pred_emo]:>6}"
        report.append(row)

    report_str = "\n".join(report)
    print(report_str)

    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write(report_str)
    with open(metrics_filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "emotion_accuracy": round(overall_accuracy, 4),
                "macro_f1_score": round(f1, 4),
                "valid_images": total_processed,
            },
            f,
            indent=2,
        )
    print(f"Saved summary report to: {summary_filepath}")
    print(f"Saved metric JSON to: {metrics_filepath}")


def run_evaluation(model, api_model_name):
    image_dir = os.path.join(model, "edited_output")
    results_dir = os.path.join(model, "eval")
    results_file = os.path.join(results_dir, "evaluation_results.json")
    summary_file = os.path.join(results_dir, "evaluation_summary.txt")
    metrics_file = os.path.join(results_dir, "emotion_metrics.json")

    print("=" * 70)
    print("VLM emotion-classification evaluation")
    print(f"Model: {model}")
    print("=" * 70)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isdir(image_dir):
        print(f"Error: image directory not found: {image_dir}")
        return

    image_infos = get_image_files_and_emotions(image_dir)
    if not image_infos:
        print(f"Error: no images matched the expected pattern: {FILE_REGEX.pattern}")
        return

    print(f"Matched images: {len(image_infos)}")
    print(f"Example: {image_infos[0]['filename']} (target: {image_infos[0]['target_emotion']})")

    processed_images = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            for item in existing_results:
                processed_images[item.get("image_name")] = item
            print(f"Loaded existing results: {len(processed_images)}")
        except json.JSONDecodeError:
            print(f"Warning: existing result file is not valid JSON: {results_file}")

    images_to_process = [
        info
        for info in image_infos
        if info["filename"] not in processed_images
        or processed_images.get(info["filename"], {}).get("predicted_emotion") in ["error", "exception"]
    ]

    if not images_to_process:
        print("All images already have valid predictions. Generating the report only.")
        calculate_and_save_accuracy(results_file, summary_file, metrics_file, api_model_name)
        return

    print(f"Pending images: {len(images_to_process)} / {len(image_infos)}")
    print(f"API model: {api_model_name}, concurrency: {MAX_CONCURRENT_REQUESTS}")

    api_key, base_url, key_env_name = get_api_config(api_model_name)
    if api_key is None:
        print(f"Error: {key_env_name} is not set in the environment.")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    success_count = 0
    failed_count = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            future_to_info = {
                executor.submit(process_single_image, client, info, api_model_name, image_dir): info
                for info in images_to_process
            }

            pbar = tqdm(concurrent.futures.as_completed(future_to_info), total=len(images_to_process), desc="Emotion scoring")
            for future in pbar:
                try:
                    result = future.result()
                    processed_images[result["data"]["image_name"]] = result["data"]

                    if result["status"] == "success":
                        success_count += 1
                    else:
                        failed_count += 1
                        error_msg = result["data"].get("details", {}).get("error", "Unknown error")
                        tqdm.write(f"  x {result['data']['image_name']}: {error_msg}")

                    if (success_count + failed_count) % SAVE_CHECKPOINT_INTERVAL == 0:
                        save_results(list(processed_images.values()), results_file)

                except Exception as exc:
                    info = future_to_info[future]
                    failed_count += 1
                    tqdm.write(f"  x {info['filename']}: {exc}")
                    processed_images[info["filename"]] = {
                        "image_name": info["filename"],
                        "target_emotion": info["target_emotion"],
                        "predicted_emotion": "exception",
                        "is_correct": False,
                        "details": {"error": f"Executor exception: {exc}"},
                    }

    except KeyboardInterrupt:
        print("\nInterrupted. Saving current progress...")
    finally:
        save_results(list(processed_images.values()), results_file)
        print(f"Saved emotion predictions to: {results_file}")
        print(f"Batch result: success={success_count}, failed={failed_count}")

    calculate_and_save_accuracy(results_file, summary_file, metrics_file, api_model_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate emotion classification for edited images.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model directory name.")
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL, help="VLM API model name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.model, args.api_model)
