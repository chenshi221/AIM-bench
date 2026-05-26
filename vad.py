import argparse
import base64
import collections
import concurrent.futures
import json
import os
import re
import time

import numpy as np
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
EDITED_FILE_REGEX = re.compile(r"^(.*?)_edited_(" + "|".join(EMOTIONS_LIST) + r")\.png$")

load_dotenv()
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
GEMINI_API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODEL = "qwen_image_edit_2509_lora"
DEFAULT_METADATA_FILE = "Instructions.json"
DEFAULT_API_MODEL = "gpt-4o"
MAX_CONCURRENT_REQUESTS = 20
SAVE_CHECKPOINT_INTERVAL = 20
MAX_RETRIES = 3
RETRY_DELAY = 2

PROMPT_TEXT = """
# ROLE
You are a meticulous expert in affective computing and psychology, specializing in visual sentiment analysis.
# TASK
Your task is to analyze the core emotion conveyed by the given image and provide a precise rating using the VAD (Valence-Arousal-Dominance) three-dimensional emotion model.
# RATING DIMENSIONS DEFINED
The rating scale for each dimension is from 1.0 to 9.0 (floating point numbers allowed), where 5.0 is perfectly neutral.
1.  **Valence (V)**: 1.0 (Extremely Unpleasant) to 9.0 (Extremely Pleasant).
2.  **Arousal (A)**: 1.0 (Extremely Calm) to 9.0 (Extremely Excited).
3.  **Dominance (D)**: 1.0 (Completely Powerless) to 9.0 (Completely In Control).
# ANALYSIS GUIDELINES
- Objectively analyze the image's content: subjects, actions, facial expressions, environment, colors, and composition.
- Your reasoning must directly connect specific visual elements from the image to your V, A, and D scores.
# OUTPUT FORMAT
You MUST provide your response strictly in the following JSON format. Do not include any introductory text or markdown.
{
  "valence": <score_float>,
  "arousal": <score_float>,
  "dominance": <score_float>,
  "reasoning": "A concise analysis linking visual elements to VAD scores."
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


def load_and_prepare_ground_truth(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        ground_truth_map = {}
        for entry in metadata:
            base_name = os.path.splitext(entry["original_image"])[0]
            if "target_emotion" in entry and "VAD" in entry["target_emotion"]:
                ground_truth_map[base_name] = entry["target_emotion"]
            else:
                print(f"Warning: missing target_emotion.VAD for metadata entry: {base_name}")

        return ground_truth_map
    except FileNotFoundError:
        print(f"Error: metadata file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: failed to parse metadata JSON: {filepath}")
        return None


def find_images_and_create_tasks(directory, ground_truth_map):
    tasks = []
    print(f"Scanning edited images in: {directory}")
    for filename in sorted(os.listdir(directory)):
        match = EDITED_FILE_REGEX.match(filename)
        if not match:
            continue
        base_name = match.group(1)
        emotion = match.group(2)
        if base_name in ground_truth_map:
            tasks.append(
                {
                    "image_filename": filename,
                    "target_emotion_info": ground_truth_map[base_name],
                    "parsed_emotion": emotion,
                }
            )
        else:
            print(f"Warning: no metadata match for image {filename} with base name {base_name}")
    return tasks


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


def validate_vad_response(data):
    if not isinstance(data, dict):
        return False, "Response is not a dictionary"
    if "error" in data:
        return False, data["error"]
    required = ["valence", "arousal", "dominance"]
    missing = [k for k in required if k not in data]
    if missing:
        return False, f"Missing VAD fields: {', '.join(missing)}"
    try:
        for key in required:
            if not (1.0 <= float(data[key]) <= 9.0):
                return False, f"{key} score out of range: {data[key]}"
    except (ValueError, TypeError):
        return False, "Invalid score format"
    return True, None


def analyze_image_with_vad(client, image_path, api_model_name, retry_count=0):
    encoded_data = encode_image_to_base64(image_path)
    if not encoded_data:
        return {"error": "Failed to encode image"}

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
        is_valid, error_msg = validate_vad_response(parsed_data)
        if not is_valid:
            raise ValueError(f"Invalid response: {error_msg}")
        return {
            k: v
            for k, v in parsed_data.items()
            if k in ["valence", "arousal", "dominance", "reasoning"]
        }
    except Exception as exc:
        if retry_count < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return analyze_image_with_vad(client, image_path, api_model_name, retry_count + 1)
        return {"error": f"{type(exc).__name__}: {exc} (after {MAX_RETRIES} retries)"}


def save_results(data, filepath):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as exc:
        print(f"Error: failed to save {filepath}: {exc}")


def generate_summary_report(results_filepath, summary_filepath, metrics_filepath, api_model_name):
    print("\n" + "=" * 70)
    print("Generating VAD report")
    print("=" * 70)

    try:
        with open(results_filepath, "r", encoding="utf-8") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: failed to load results file: {results_filepath}")
        return

    valid_results = [r for r in results if "error" not in r.get("predicted_vad", {})]
    if not valid_results:
        print("Error: no valid VAD results are available for reporting.")
        return

    metrics_by_emotion = collections.defaultdict(lambda: {"distances": [], "errors": collections.defaultdict(list)})
    overall_distances = []
    overall_errors = collections.defaultdict(list)

    for result in valid_results:
        gt_vad = result["ground_truth_vad"]
        pred_vad = result["predicted_vad"]
        emotion = result["target_emotion"]

        gt_vec = np.array([gt_vad["valence"], gt_vad["arousal"], gt_vad["dominance"]])
        pred_vec = np.array([pred_vad["valence"], pred_vad["arousal"], pred_vad["dominance"]])
        distance = float(np.linalg.norm(gt_vec - pred_vec))

        metrics_by_emotion[emotion]["distances"].append(distance)
        overall_distances.append(distance)

        for dim in ["valence", "arousal", "dominance"]:
            error = abs(gt_vad[dim] - pred_vad[dim])
            metrics_by_emotion[emotion]["errors"][dim].append(error)
            overall_errors[dim].append(error)

    vad_avg_distance = float(np.mean(overall_distances))
    report = [
        "=" * 80,
        f"VAD prediction report - API model: {api_model_name}",
        "=" * 80,
        f"Valid images: {len(valid_results)}",
        "\nOverall metrics",
        f"Mean Euclidean distance: {vad_avg_distance:.4f}",
        "\nMean absolute error",
        f"{'Dimension':<12} | {'MAE':>8}",
        "-" * 24,
    ]
    for dim in ["valence", "arousal", "dominance"]:
        report.append(f"{dim.capitalize():<12} | {np.mean(overall_errors[dim]):>8.4f}")

    report.append("\nPer-target-emotion metrics")
    header = f"{'Emotion':<12} | {'Mean distance':>14} | {'Valence MAE':>13} | {'Arousal MAE':>13} | {'Dominance MAE':>15} | {'Count':>7}"
    report.append(header)
    report.append("-" * len(header))

    for emotion in sorted(metrics_by_emotion.keys()):
        data = metrics_by_emotion[emotion]
        report.append(
            f"{emotion:<12} | "
            f"{np.mean(data['distances']):>14.4f} | "
            f"{np.mean(data['errors']['valence']):>13.4f} | "
            f"{np.mean(data['errors']['arousal']):>13.4f} | "
            f"{np.mean(data['errors']['dominance']):>15.4f} | "
            f"{len(data['distances']):>7}"
        )

    report_str = "\n".join(report)
    print(report_str)

    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write(report_str)
    with open(metrics_filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vad_avg_distance": round(vad_avg_distance, 4),
                "valid_images": len(valid_results),
            },
            f,
            indent=2,
        )
    print(f"Saved VAD summary to: {summary_filepath}")
    print(f"Saved VAD metric JSON to: {metrics_filepath}")


def process_task(client, task, image_dir, api_model_name):
    image_filename = task["image_filename"]
    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        predicted_vad = {"error": "Image file not found"}
    else:
        predicted_vad = analyze_image_with_vad(client, image_path, api_model_name)

    return {
        "edited_image": image_filename,
        "target_emotion": task["target_emotion_info"]["emotion"],
        "ground_truth_vad": task["target_emotion_info"]["VAD"],
        "predicted_vad": predicted_vad,
    }


def run_evaluation(model, metadata_file, api_model_name):
    image_dir = os.path.join(model, "edited_output")
    results_dir = os.path.join(model, "eval")
    results_file = os.path.join(results_dir, f"evaluation_vad_results_{api_model_name.replace('/', '_')}.json")
    summary_file = os.path.join(results_dir, f"evaluation_vad_summary_{api_model_name.replace('/', '_')}.txt")
    metrics_file = os.path.join(results_dir, "vad_metrics.json")

    print("=" * 70)
    print("VAD evaluation")
    print(f"Model: {model}")
    print("=" * 70)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isdir(image_dir):
        print(f"Error: image directory not found: {image_dir}")
        return

    api_key, base_url, key_env_name = get_api_config(api_model_name)
    if not api_key:
        print(f"Error: {key_env_name} is not set in the environment.")
        return

    ground_truth_map = load_and_prepare_ground_truth(metadata_file)
    if not ground_truth_map:
        return

    tasks_to_run = find_images_and_create_tasks(image_dir, ground_truth_map)
    if not tasks_to_run:
        print("Error: no images matched the edited-image naming pattern and metadata.")
        return

    print(f"Created VAD tasks: {len(tasks_to_run)}")
    client = OpenAI(api_key=api_key, base_url=base_url)

    processed_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    processed_results[item["edited_image"]] = item
            print(f"Loaded existing VAD results: {len(processed_results)}")
        except json.JSONDecodeError:
            print(f"Warning: existing result file is not valid JSON: {results_file}")

    final_tasks = [
        task
        for task in tasks_to_run
        if task["image_filename"] not in processed_results
        or "error" in processed_results[task["image_filename"]].get("predicted_vad", {})
    ]

    if not final_tasks:
        print("All images already have valid VAD predictions. Generating the report only.")
        generate_summary_report(results_file, summary_file, metrics_file, api_model_name)
        return

    print(f"Pending VAD tasks: {len(final_tasks)} / {len(tasks_to_run)}")
    print(f"API model: {api_model_name}, concurrency: {MAX_CONCURRENT_REQUESTS}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            future_to_task = {
                executor.submit(process_task, client, task, image_dir, api_model_name): task
                for task in final_tasks
            }
            pbar = tqdm(concurrent.futures.as_completed(future_to_task), total=len(final_tasks), desc="VAD scoring")

            for i, future in enumerate(pbar, 1):
                try:
                    result = future.result()
                    processed_results[result["edited_image"]] = result
                    if "error" in result["predicted_vad"]:
                        tqdm.write(f"  x {result['edited_image']}: {result['predicted_vad']['error']}")
                except Exception as exc:
                    task = future_to_task[future]
                    filename = task["image_filename"]
                    tqdm.write(f"  x {filename}: {exc}")
                    processed_results[filename] = {
                        "edited_image": filename,
                        "predicted_vad": {"error": f"Executor exception: {exc}"},
                    }

                if i % SAVE_CHECKPOINT_INTERVAL == 0:
                    save_results(list(processed_results.values()), results_file)
                    tqdm.write(f"Saved checkpoint with {len(processed_results)} records.")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving current progress...")
    finally:
        save_results(list(processed_results.values()), results_file)
        print(f"Saved VAD details to: {results_file}")

    generate_summary_report(results_file, summary_file, metrics_file, api_model_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAD distance for edited images.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model directory name.")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_FILE, help="Instruction metadata JSON path.")
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL, help="VLM API model name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.model, args.metadata, args.api_model)
