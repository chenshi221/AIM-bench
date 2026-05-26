import argparse
import base64
import collections
import concurrent.futures
import json
import os
import re
import sys
from contextlib import contextmanager

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

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

DEFAULT_INSTRUCTION_FILE = "Instructions.json"
DEFAULT_GT_IMAGE_DIR = "./benchmark"
DEFAULT_API_MODEL = "gpt-4o"
MAX_CONCURRENT_REQUESTS = 200
SAVE_CHECKPOINT_INTERVAL = 20

load_dotenv()
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
GEMINI_API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"

SC_EVALUATION_PROMPT = """
# ROLE
You are an expert in evaluating image editing.

# TASK
You will be given an original image, an edited version of it, and the text instruction used for the edit. Your task is to evaluate how successfully the editing instruction has been executed.

# RATING DIMENSIONS (Scale 0-10)
1. **Editing Success (Score 1)**: How well does the edited image follow the instruction?
   - 0: The instruction is completely ignored.
   - 10: The instruction is perfectly and accurately executed.
2. **Editing Fidelity (Score 2)**: How well are the unedited parts of the original image preserved?
   - 0: The edited image is completely different from the original, showing extreme over-editing.
   - 10: Only the areas relevant to the instruction are changed, preserving the original's identity perfectly.

# OUTPUT FORMAT
You MUST provide your response strictly as JSON:
{
  "scores": [score1, score2],
  "reason": "A concise analysis explaining the scores by referencing specific visual changes between the original and edited images."
}
"""

PQ_EVALUATION_PROMPT = """
# ROLE
You are an expert in generated-image quality assessment.

# TASK
You will be given a single generated image. Your task is to evaluate its perceptual quality based on its realism and technical flaws.

# RATING DIMENSIONS (Scale 0-10)
1. **Naturalness (Score 1)**: How natural and realistic does the image look?
   - 0: The image looks completely unnatural.
   - 10: The image is indistinguishable from a real photograph in terms of naturalness.
2. **Freedom from Artifacts (Score 2)**: How free is the image from generation artifacts?
   - 0: The image is filled with severe artifacts.
   - 10: The image is perfectly clean and has no visible artifacts.

# OUTPUT FORMAT
You MUST provide your response strictly as JSON:
{
  "scores": [score1, score2],
  "reason": "A concise analysis explaining the scores by pointing out specific visual elements related to naturalness and artifacts in the image."
}
"""

GTC_EVALUATION_PROMPT = """
# ROLE
You are an expert in comparing generated images against a ground-truth standard.

# TASK
You will be given a ground-truth image and a model-generated image, both created from the same editing instruction. Your task is to evaluate how well the model output matches the ground truth.

# RATING DIMENSIONS (Scale 0-10)
1. **Semantic Match (Score 1)**: How well does the model image capture the meaning and core idea of the edit shown in the ground-truth image?
   - 0: It completely fails to capture the same intent.
   - 10: It perfectly captures the same semantic change, even if stylistically different.
2. **Visual Similarity (Score 2)**: How visually similar is the model image to the ground truth, considering object placement, color, and style?
   - 0: The image is visually completely different.
   - 10: The image is visually identical or nearly identical to the ground truth.

# OUTPUT FORMAT
You MUST provide your response strictly as JSON:
{
  "scores": [score1, score2],
  "reason": "A concise analysis explaining the scores by comparing specific visual elements between the ground truth and the model-generated images."
}
"""


def get_api_config(api_model):
    if api_model.lower().startswith("gemini"):
        for env_var in GEMINI_API_KEY_ENV_VARS:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key, GEMINI_BASE_URL, env_var
        return None, GEMINI_BASE_URL, "GEMINI_API_KEY or GOOGLE_API_KEY"
    return os.getenv(OPENAI_API_KEY_ENV_VAR), OPENAI_BASE_URL, OPENAI_API_KEY_ENV_VAR


def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            from io import BytesIO

            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return None


def parse_api_response(response_content):
    try:
        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in response", response_content, 0)
        parsed = json.loads(json_match.group(0))
        scores = parsed.get("scores")
        if isinstance(scores, list) and len(scores) == 2 and all(isinstance(x, (int, float)) for x in scores):
            return {"scores": [int(s) for s in scores], "reason": parsed.get("reason", "")}
        return {"error": f"Invalid score format: {scores}"}
    except json.JSONDecodeError:
        return {"error": f"Failed to parse JSON response: {response_content}"}


def get_sc_scores_from_api(client, original_image_path, edited_image_path, instruction, api_model):
    b64_orig = encode_image_to_base64(original_image_path)
    b64_edit = encode_image_to_base64(edited_image_path)
    if not b64_orig or not b64_edit:
        return {"error": "Failed to encode original or edited image"}
    try:
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SC_EVALUATION_PROMPT},
                        {"type": "text", "text": f"Editing instruction: {instruction}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_orig}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_edit}"}},
                    ],
                }
            ],
            max_tokens=200,
            temperature=0.0,
        )
        return parse_api_response(response.choices[0].message.content)
    except Exception as exc:
        return {"error": f"SC API error: {exc}"}


def get_pq_scores_from_api(client, image_path, api_model):
    b64_img = encode_image_to_base64(image_path)
    if not b64_img:
        return {"error": "Failed to encode image"}
    try:
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PQ_EVALUATION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                }
            ],
            max_tokens=200,
            temperature=0.0,
        )
        return parse_api_response(response.choices[0].message.content)
    except Exception as exc:
        return {"error": f"PQ API error: {exc}"}


def get_gtc_scores_from_api(client, model_output_path, gt_image_path, instruction, api_model):
    b64_model = encode_image_to_base64(model_output_path)
    b64_gt = encode_image_to_base64(gt_image_path)
    if not b64_model or not b64_gt:
        return {"error": "Failed to encode model or ground-truth image"}
    try:
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GTC_EVALUATION_PROMPT},
                        {"type": "text", "text": f"Editing instruction: {instruction}"},
                        {"type": "text", "text": "Image 1: Ground-truth image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_gt}"}},
                        {"type": "text", "text": "Image 2: Model-generated image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_model}"}},
                    ],
                }
            ],
            max_tokens=200,
            temperature=0.0,
        )
        return parse_api_response(response.choices[0].message.content)
    except Exception as exc:
        return {"error": f"GTC API error: {exc}"}


def process_full_evaluation_task(client, task, model_image_dir, gt_image_dir, api_model):
    orig_path = os.path.join(model_image_dir, task["original_filename"])
    edit_path = os.path.join(model_image_dir, task["edited_filename"])

    update = {}

    sc_res = get_sc_scores_from_api(client, orig_path, edit_path, task["instruction"], api_model)
    update.update({"sc_scores": sc_res.get("scores"), "sc_reason": sc_res.get("reason", ""), "sc_error": sc_res.get("error")})

    pq_res = get_pq_scores_from_api(client, edit_path, api_model)
    update.update({"pq_scores": pq_res.get("scores"), "pq_reason": pq_res.get("reason", ""), "pq_error": pq_res.get("error")})

    if task.get("gt_filename"):
        gt_path = os.path.join(gt_image_dir, task["gt_filename"])
        if os.path.exists(gt_path):
            gtc_res = get_gtc_scores_from_api(client, edit_path, gt_path, task["instruction"], api_model)
            update.update({"gtc_scores": gtc_res.get("scores"), "gtc_reason": gtc_res.get("reason", ""), "gtc_error": gtc_res.get("error")})
        else:
            update.update({"gtc_error": f"Ground-truth file not found: {task['gt_filename']}"})
    return update


def create_task_list(model_image_dir, instruction_file):
    print(f"Building task list for: {model_image_dir}")
    if not os.path.exists(instruction_file):
        print(f"Error: instruction file not found: {instruction_file}")
        return None
    if not os.path.isdir(model_image_dir):
        print(f"Error: model image directory not found: {model_image_dir}")
        return None

    edited_file_regex = re.compile(r"^(.*?)_edited_([a-zA-Z]+)\.png$")
    orig_map = {
        filename.rsplit("_original.jpg", 1)[0]: filename
        for filename in os.listdir(model_image_dir)
        if filename.lower().endswith("_original.jpg")
    }
    edit_map = {
        match.group(1): filename
        for filename in os.listdir(model_image_dir)
        if (match := edited_file_regex.match(filename))
    }

    pairs = [{"base": base, "orig": original, "edit": edit_map[base]} for base, original in orig_map.items() if base in edit_map]
    print(f"Matched original-edited pairs: {len(pairs)}")

    with open(instruction_file, "r", encoding="utf-8") as f:
        inst_map = {os.path.splitext(entry["original_image"])[0]: entry for entry in json.load(f)}
    print(f"Loaded instruction records: {len(inst_map)}")

    tasks, skipped = [], 0
    for pair in pairs:
        instruction = inst_map.get(pair["base"])
        if instruction and "edit_prompt" in instruction:
            tasks.append(
                {
                    "base_name": pair["base"],
                    "original_filename": pair["orig"],
                    "edited_filename": pair["edit"],
                    "instruction": instruction["edit_prompt"],
                    "target_category": edited_file_regex.match(pair["edit"]).group(2),
                    "gt_filename": instruction.get("edited_image"),
                }
            )
        else:
            skipped += 1

    print(f"Created tasks: {len(tasks)}; skipped unmatched pairs: {skipped}")
    return tasks


def calculate_and_save_report(model_name, scores_filepath, report_filepath):
    print(f"\nGenerating SC/PQ/GTC report for: {model_name}")
    try:
        with open(scores_filepath, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: failed to load score file: {scores_filepath}")
        return None

    sc_pq_results = [r for r in all_results if r.get("sc_scores") and r.get("pq_scores")]
    gtc_results = [r for r in sc_pq_results if r.get("gtc_scores")]
    if not sc_pq_results:
        print("Error: no valid SC/PQ scores are available for reporting.")
        return None

    scores_by_cat = collections.defaultdict(lambda: collections.defaultdict(list))
    for result in sc_pq_results:
        category = result.get("target_category", "unknown")
        scores_by_cat[category]["sc_score"].append(min(result["sc_scores"]))
        scores_by_cat[category]["pq_score"].append(min(result["pq_scores"]))

    for result in gtc_results:
        category = result.get("target_category", "unknown")
        scores_by_cat[category]["gtc_score"].append(min(result["gtc_scores"]))

    def get_overall_avg(key):
        all_scores = [score for data in scores_by_cat.values() for score in data.get(key, [])]
        return float(np.mean(all_scores)) if all_scores else 0.0

    avg_sc = get_overall_avg("sc_score")
    avg_pq = get_overall_avg("pq_score")
    avg_gtc = get_overall_avg("gtc_score")

    report = [
        "=" * 80,
        f"Image-editing evaluation report - Model: {model_name}",
        "=" * 80,
        "Aggregation rule: min(score1, score2)",
        f"Valid SC/PQ images: {len(sc_pq_results)} / {len(all_results)}",
        f"Valid GTC images: {len(gtc_results)}",
        "\nOverall averages",
        f"SC-Score:  {avg_sc:.4f}",
        f"PQ-Score:  {avg_pq:.4f}",
        f"GTC-Score: {avg_gtc:.4f}" if gtc_results else "GTC-Score: N/A",
        "\nPer-target-emotion averages",
    ]

    header = f"{'Emotion':<12} | {'SC-Score':>10} | {'PQ-Score':>10} | {'GTC-Score':>11} | {'Count':>7}"
    report.append(header)
    report.append("-" * len(header))

    for category, data in sorted(scores_by_cat.items()):
        avg_gtc_str = f"{np.mean(data['gtc_score']):>11.4f}" if data["gtc_score"] else f"{'N/A':>11}"
        report.append(
            f"{category:<12} | "
            f"{np.mean(data['sc_score']):>10.4f} | "
            f"{np.mean(data['pq_score']):>10.4f} | "
            f"{avg_gtc_str} | "
            f"{len(data['sc_score']):>7}"
        )

    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Saved SC/PQ/GTC report to: {report_filepath}")

    return {"SC_Score": avg_sc, "PQ_Score": avg_pq, "GTC_Score": avg_gtc if gtc_results else None}


def process_model(model_name, instruction_file, gt_image_dir, api_model, api_key, base_url):
    print("\n" + "=" * 80)
    print(f"Evaluating model: {model_name}")
    print("=" * 80)

    model_image_dir = os.path.join(model_name, "edited_output")
    eval_dir = os.path.join(model_name, "eval")
    results_file = os.path.join(eval_dir, "evaluation_scores.json")
    summary_file = os.path.join(eval_dir, "evaluation_summary_report.txt")
    os.makedirs(eval_dir, exist_ok=True)

    tasks = create_task_list(model_image_dir, instruction_file)
    if not tasks:
        print(f"Skipping {model_name}: no valid tasks.")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    processed = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    processed[item["edited_filename"]] = item
            print(f"Loaded existing results: {len(processed)}")
        except json.JSONDecodeError:
            print(f"Warning: existing result file is not valid JSON: {results_file}")

    tasks_to_run = []
    for task in tasks:
        key = task["edited_filename"]
        if key in processed:
            previous = processed[key]
            needs_gtc = task.get("gt_filename") is not None
            has_basic = previous.get("sc_scores") and previous.get("pq_scores")
            has_gtc = (not needs_gtc) or previous.get("gtc_scores") or previous.get("gtc_error")
            if has_basic and has_gtc:
                continue
        tasks_to_run.append(task)

    if not tasks_to_run:
        print("All tasks already have complete scores. Generating the report only.")
    else:
        print(f"Pending tasks: {len(tasks_to_run)} / {len(tasks)}")
        final_results = collections.OrderedDict((task["edited_filename"], task) for task in tasks)
        final_results.update(processed)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                future_to_task = {
                    executor.submit(process_full_evaluation_task, client, task, model_image_dir, gt_image_dir, api_model): task
                    for task in tasks_to_run
                }
                pbar = tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks_to_run), desc=f"SC/PQ/GTC {model_name}")
                for i, future in enumerate(pbar, 1):
                    task = future_to_task[future]
                    try:
                        task.update(future.result())
                        for metric in ["sc", "pq", "gtc"]:
                            if task.get(f"{metric}_error"):
                                tqdm.write(f"  x {metric.upper()} {task['edited_filename']}: {task[f'{metric}_error']}")
                    except Exception as exc:
                        task.update({"sc_error": f"Executor exception: {exc}", "pq_error": f"Executor exception: {exc}"})
                    final_results[task["edited_filename"]] = task
                    if i % SAVE_CHECKPOINT_INTERVAL == 0:
                        with open(results_file, "w", encoding="utf-8") as f:
                            json.dump(list(final_results.values()), f, indent=2)

        except KeyboardInterrupt:
            print("\nInterrupted. Saving current progress...")
        finally:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(list(final_results.values()), f, indent=2)
            print(f"Saved SC/PQ/GTC details to: {results_file}")

    return calculate_and_save_report(model_name, results_file, summary_file)


def run_all_models(model_list, instruction_file, gt_image_dir, api_model):
    print("=" * 80)
    print("Running multi-model SC/PQ/GTC evaluation")
    print("=" * 80)

    api_key, base_url, key_env_name = get_api_config(api_model)
    if not api_key:
        print(f"Error: {key_env_name} is not set in the environment.")
        return

    all_models_summary = {}
    for model_name in model_list:
        summary_metrics = process_model(model_name, instruction_file, gt_image_dir, api_model, api_key, base_url)
        if summary_metrics:
            all_models_summary[model_name] = summary_metrics
        else:
            print(f"Model excluded from final summary: {model_name}")

    print("\n" + "=" * 80)
    print("Final SC/PQ/GTC summary")
    print("=" * 80)

    md_table = [
        "| Model Name | SC-Score | PQ-Score | GTC-Score |",
        "|:-----------|:--------:|:--------:|:---------:|",
    ]

    sorted_models = sorted(all_models_summary.items(), key=lambda item: item[1].get("SC_Score", 0), reverse=True)
    for model_name, metrics in sorted_models:
        sc = f"{metrics.get('SC_Score', 0):.4f}"
        pq = f"{metrics.get('PQ_Score', 0):.4f}"
        gtc = f"{metrics.get('GTC_Score', 0):.4f}" if metrics.get("GTC_Score") is not None else "N/A"
        md_table.append(f"| {model_name} | {sc} | {pq} | {gtc} |")

    md_report_str = "\n".join(md_table)
    print(md_report_str)

    summary_json_path = "all_models_evaluation_summary.json"
    summary_md_path = "all_models_evaluation_summary.md"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_models_summary, f, indent=4)
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_report_str)
    print(f"Saved summary JSON to: {summary_json_path}")
    print(f"Saved summary Markdown to: {summary_md_path}")


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def test_execution_plan(model_list, instruction_file):
    print("\n" + "#" * 100)
    print("SC/PQ/GTC execution plan")
    print("#" * 100)

    if not os.path.exists(instruction_file):
        print(f"Error: instruction file not found: {instruction_file}")
        return

    header_fmt = "| {name:<30} | {total:>8} | {done:>8} | {todo:>10} | {status:<18} |"
    print("-" * 88)
    print(header_fmt.format(name="Model Name", total="Total", done="Done", todo="Remaining", status="Status"))
    print("-" * 88)

    total_todo_all = 0
    total_images_all = 0

    for model_name in model_list:
        model_image_dir = os.path.join(model_name, "edited_output")
        results_file = os.path.join(model_name, "eval", "evaluation_scores.json")

        if not os.path.isdir(model_image_dir):
            print(header_fmt.format(name=model_name, total="-", done="-", todo="-", status="No directory"))
            continue

        try:
            with suppress_stdout():
                tasks = create_task_list(model_image_dir, instruction_file)
        except Exception:
            tasks = []

        if not tasks:
            print(header_fmt.format(name=model_name, total="0", done="0", todo="0", status="No match"))
            continue

        total_count = len(tasks)
        total_images_all += total_count

        processed_data = {}
        if os.path.exists(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        processed_data[item["edited_filename"]] = item
            except Exception:
                pass

        done_count = 0
        for task in tasks:
            key = task["edited_filename"]
            if key in processed_data:
                previous = processed_data[key]
                has_basic = previous.get("sc_scores") and previous.get("pq_scores")
                needs_gtc = task.get("gt_filename") is not None
                has_gtc = (not needs_gtc) or previous.get("gtc_scores") or previous.get("gtc_error")
                if has_basic and has_gtc:
                    done_count += 1

        todo_count = total_count - done_count
        total_todo_all += todo_count
        if todo_count == 0:
            status_str = "All done"
        elif todo_count == total_count:
            status_str = "New task"
        else:
            pct = int((done_count / total_count) * 100)
            status_str = f"{pct}% processed"

        print(header_fmt.format(name=model_name, total=total_count, done=done_count, todo=todo_count, status=status_str))

    print("-" * 88)
    print(f"Total tasks: {total_images_all}")
    print(f"Pending tasks: {total_todo_all}")
    print("#" * 100 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SC, PQ, and GTC metrics for edited images.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODEL_LIST, help="Model directories to evaluate.")
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTION_FILE, help="Instruction JSON path.")
    parser.add_argument("--gt-dir", default=DEFAULT_GT_IMAGE_DIR, help="Ground-truth image directory.")
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL, help="VLM API model name.")
    parser.add_argument("--plan-only", action="store_true", help="Print pending-task counts without calling the API.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plan_only:
        test_execution_plan(args.models, args.instructions)
    else:
        run_all_models(args.models, args.instructions, args.gt_dir, args.api_model)
