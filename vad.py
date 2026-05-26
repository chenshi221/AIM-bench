import os
import json
import base64
import time
import re
import collections
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageFile
from tqdm import tqdm
import concurrent.futures

# 允许加载可能被截断的图像文件，增加代码的健壮性
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# --- 1. 配置区域 (请根据您的环境修改) ---
# ==============================================================================

# --- 输入/输出路径配置 ---
# 存放您所有 *_edited_emo.png 图片的文件夹
MODEL = "qwen_image_edit_2509_lora"  # <-- 请修改为您的模型文件夹名称
IMAGE_DIR = os.path.join(MODEL, "edited_output")

# 【重要】包含标准答案 (Ground Truth) VAD分数的元数据JSON文件
# 脚本会根据图片文件名在此文件中查找对应的VAD标准值
METADATA_JSON_FILE = "./src/Instructions.json"

# 结果输出目录
RESULTS_DIR = os.path.join(MODEL, "eval")

# --- API 配置 ---
load_dotenv()
API_KEY = os.getenv("DMXAPI_API_KEY_MY")
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL_NAME = "gpt-4o"

# --- 文件名匹配模式 ---
# 用于匹配编辑后图片并提取基础文件名和情感的正则表达式
EMOTIONS_LIST = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
EDITED_FILE_REGEX = re.compile(r"^(.*?)_edited_(" + "|".join(EMOTIONS_LIST) + r")\.png$")


# --- 输出文件名配置 ---
# 保存每张图片详细预测结果的JSON文件
RESULTS_FILE = os.path.join(RESULTS_DIR, f"evaluation_vad_results_{MODEL_NAME.replace('/', '_')}.json")
# 保存包含欧式距离和MAE的最终评测报告
SUMMARY_FILE = os.path.join(RESULTS_DIR, f"evaluation_vad_summary_{MODEL_NAME.replace('/', '_')}.txt")

# --- 运行控制 ---
MAX_CONCURRENT_REQUESTS = 20    # 并发请求数
SAVE_CHECKPOINT_INTERVAL = 20   # 每处理N张图片保存一次检查点
MAX_RETRIES = 3                 # 最大重试次数
RETRY_DELAY = 2                 # 重试间隔（秒）

# --- VAD模型的分析指令 (Prompt) ---
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
-   Objectively analyze the image's content: subjects, actions, facial expressions, environment, colors, and composition.
-   Your reasoning must directly connect specific visual elements from the image to your V, A, and D scores.
# OUTPUT FORMAT
You MUST provide your response strictly in the following JSON format. Do not include any introductory text or markdown.
{
  "valence": <score_float>,
  "arousal": <score_float>,
  "dominance": <score_float>,
  "reasoning": "A concise analysis linking visual elements to VAD scores."
}
"""

# ==============================================================================
# --- 2. 核心功能函数 ---
# ==============================================================================

def load_and_prepare_ground_truth(filepath):
    """加载元数据JSON文件，并将其转换为以基础文件名为键的字典以便快速查找。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        ground_truth_map = {}
        for entry in metadata:
            # 从 "fear_cluster_2_fear_04287.jpg" 中提取 "fear_cluster_2_fear_04287"
            base_name = os.path.splitext(entry['original_image'])[0]
            if 'target_emotion' in entry and 'VAD' in entry['target_emotion']:
                 ground_truth_map[base_name] = entry['target_emotion'] # 保存整个目标情感对象
            else:
                print(f"  [警告] 元数据条目 {base_name} 缺少 target_emotion.VAD 字段，将被忽略。")

        return ground_truth_map
    except FileNotFoundError:
        print(f"❌ [致命错误] 元数据文件未找到: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"❌ [致命错误] 无法解析JSON文件: {filepath}")
        return None

def find_images_and_create_tasks(directory, ground_truth_map):
    """扫描目录，匹配图片，并结合标准答案创建评测任务列表。"""
    tasks = []
    print("\n🔍 正在扫描图片目录并匹配元数据...")
    for filename in os.listdir(directory):
        match = EDITED_FILE_REGEX.match(filename)
        if match:
            base_name = match.group(1)
            emotion = match.group(2)
            if base_name in ground_truth_map:
                tasks.append({
                    "image_filename": filename,
                    "target_emotion_info": ground_truth_map[base_name],
                    "parsed_emotion": emotion # 从文件名解析出的情感，用于分类报告
                })
            else:
                print(f"  [警告] 找到图片 '{filename}'，但在元数据文件中没有找到匹配的基础名称 '{base_name}'。")
    return tasks

def encode_image_to_base64(image_path):
    """将图片编码为Base64字符串。"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            from io import BytesIO
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception:
        return None

def validate_vad_response(data):
    """验证VLM返回的VAD响应的格式和范围。"""
    if not isinstance(data, dict): return False, "响应不是一个字典"
    if "error" in data: return False, data["error"]
    required = ["valence", "arousal", "dominance"]
    if not all(k in data for k in required): return False, "缺少VAD关键字段"
    try:
        for key in required:
            if not (1.0 <= float(data[key]) <= 9.0):
                return False, f"{key} 分数 {data[key]} 超出 [1.0, 9.0] 范围"
    except (ValueError, TypeError):
        return False, "无效的分数格式"
    return True, None

def analyze_image_with_vad(client, image_path, retry_count=0):
    """使用VLM分析单张图片的VAD分数，带重试机制。"""
    encoded_data = encode_image_to_base64(image_path)
    if not encoded_data: return {"error": "图片编码失败"}

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT_TEXT}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_data}"}}]}],
            temperature=0.1
        )
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match: raise json.JSONDecodeError("响应中未找到JSON对象", content, 0)

        parsed_data = json.loads(json_match.group(0))
        is_valid, error_msg = validate_vad_response(parsed_data)
        if not is_valid: raise ValueError(f"无效响应: {error_msg}")
        return {k: v for k, v in parsed_data.items() if k in ["valence", "arousal", "dominance", "reasoning"]}
    except Exception as e:
        if retry_count < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return analyze_image_with_vad(client, image_path, retry_count + 1)
        return {"error": f"{type(e).__name__}: {e} (在 {MAX_RETRIES} 次重试后失败)"}

def save_results(data, filepath):
    """将数据安全地保存到JSON文件。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"\n[错误] 保存结果文件 '{filepath}' 失败: {e}")

# ==============================================================================
# --- 3. 评测与报告生成 ---
# ==============================================================================

def generate_summary_report(results_filepath, summary_filepath):
    """计算欧式距离和MAE，并生成最终的评测报告。"""
    print("\n" + "📊" * 35)
    print("正在生成VAD评测报告...")
    print("📊" * 35)

    try:
        with open(results_filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"❌ 无法加载或解析结果文件: {results_filepath}")
        return

    valid_results = [r for r in results if "error" not in r.get("predicted_vad", {})]
    if not valid_results:
        print("❌ 没有找到有效的评测结果来生成报告。")
        return

    # 初始化统计数据结构
    metrics_by_emotion = collections.defaultdict(lambda: {"distances": [], "errors": collections.defaultdict(list)})
    overall_distances = []
    overall_errors = collections.defaultdict(list)

    for res in valid_results:
        gt_vad = res["ground_truth_vad"]
        pred_vad = res["predicted_vad"]
        emotion = res["target_emotion"]

        # 计算欧式距离
        gt_vec = np.array([gt_vad['valence'], gt_vad['arousal'], gt_vad['dominance']])
        pred_vec = np.array([pred_vad['valence'], pred_vad['arousal'], pred_vad['dominance']])
        dist = np.linalg.norm(gt_vec - pred_vec)
        
        metrics_by_emotion[emotion]["distances"].append(dist)
        overall_distances.append(dist)

        # 计算MAE
        for dim in ["valence", "arousal", "dominance"]:
            error = abs(gt_vad[dim] - pred_vad[dim])
            metrics_by_emotion[emotion]["errors"][dim].append(error)
            overall_errors[dim].append(error)

    # --- 构建报告字符串 ---
    report = [
        "=" * 80,
        f"VAD 预测性能评测报告 - 模型: {MODEL_NAME}",
        "=" * 80,
        f"总评测图片数: {len(valid_results)}",
        
        "\n--- 1. 总体性能指标 ---",
        f"平均欧式距离 (Euclidean Distance): {np.mean(overall_distances):.4f}",
        f"  - 这个值代表预测的VAD三维向量在情感空间中与标准答案的平均距离，越小越好。",
        
        "\n平均绝对误差 (Mean Absolute Error):",
        f"  - {'维度':<12} | {'MAE':>8}",
        "  - " + "-" * 22,
    ]
    for dim in ["valence", "arousal", "dominance"]:
        mae = np.mean(overall_errors[dim])
        report.append(f"  - {dim.capitalize():<12} | {mae:>8.4f}")

    report.append("\n--- 2. 按目标情感分类的性能指标 ---")
    header = f"{'情感':<12} | {'平均欧式距离':>14} | {'Valence MAE':>13} | {'Arousal MAE':>13} | {'Dominance MAE':>15} | {'数量':>7}"
    report.append(header)
    report.append("-" * len(header))

    for emo in sorted(metrics_by_emotion.keys()):
        data = metrics_by_emotion[emo]
        count = len(data["distances"])
        avg_dist = np.mean(data["distances"])
        mae_v = np.mean(data["errors"]["valence"])
        mae_a = np.mean(data["errors"]["arousal"])
        mae_d = np.mean(data["errors"]["dominance"])
        report.append(f"{emo:<12} | {avg_dist:>14.4f} | {mae_v:>13.4f} | {mae_a:>13.4f} | {mae_d:>15.4f} | {count:>7}")

    report_str = "\n".join(report)
    print(report_str)

    try:
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"\n✅ 评测报告已保存至: {summary_filepath}")
    except Exception as e:
        print(f"\n❌ 保存评测报告失败: {e}")


# ==============================================================================
# --- 4. 主执行逻辑 ---
# ==============================================================================

def process_task(client, task):
    """处理单个评测任务。"""
    image_filename = task['image_filename']
    image_path = os.path.join(IMAGE_DIR, image_filename)

    if not os.path.exists(image_path):
        predicted_vad = {"error": "图片文件未找到"}
    else:
        predicted_vad = analyze_image_with_vad(client, image_path)

    return {
        "edited_image": image_filename,
        "target_emotion": task["target_emotion_info"]["emotion"],
        "ground_truth_vad": task["target_emotion_info"]["VAD"],
        "predicted_vad": predicted_vad
    }

def main():
    """主函数，用于运行VAD评测任务。"""
    print("=" * 30, "VAD 模型性能评测任务", "=" * 30)

    if not API_KEY:
        print("❌ [致命错误] 未在 .env 文件中找到API密钥 (DMXAPI_API_KEY_MY)。")
        return

    ground_truth_map = load_and_prepare_ground_truth(METADATA_JSON_FILE)
    if not ground_truth_map:
        return

    tasks_to_run = find_images_and_create_tasks(IMAGE_DIR, ground_truth_map)
    if not tasks_to_run:
        print("\n❌ 未找到任何符合命名规则且能在元数据中匹配到的图片。请检查文件名和元数据文件。")
        return
    
    print(f"✅ 成功创建 {len(tasks_to_run)} 个评测任务。")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 加载已有结果，实现断点续传
    processed_results = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    processed_results[item['edited_image']] = item
            print(f"✓ 已加载 {len(processed_results)} 条现有结果，将继续执行。")
        except json.JSONDecodeError:
            print(f"⚠️ 无法解析已有的结果文件，将重新开始。")

    # 筛选出需要处理的任务
    final_tasks = [
        task for task in tasks_to_run
        if task['image_filename'] not in processed_results or "error" in processed_results[task['image_filename']].get("predicted_vad", {})
    ]

    if not final_tasks:
        print("\n✅ 所有图片均已评测完毕！现在开始生成最终报告。")
        generate_summary_report(RESULTS_FILE, SUMMARY_FILE)
        return

    print(f"\n📊 待处理任务总数: {len(final_tasks)}")
    print(f"   使用模型: {MODEL_NAME}, 并发数: {MAX_CONCURRENT_REQUESTS}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            future_to_task = {executor.submit(process_task, client, task): task for task in final_tasks}
            pbar = tqdm(concurrent.futures.as_completed(future_to_task), total=len(final_tasks), desc="VAD评测进度")

            for i, future in enumerate(pbar, 1):
                try:
                    result = future.result()
                    processed_results[result['edited_image']] = result
                    if "error" in result['predicted_vad']:
                         tqdm.write(f"  ✗ 错误: {result['edited_image']}: {result['predicted_vad']['error']}")
                except Exception as exc:
                    task = future_to_task[future]
                    fname = task['image_filename']
                    tqdm.write(f"  ✗ 处理 {fname} 时发生严重错误: {exc}")
                    processed_results[fname] = {"edited_image": fname, "predicted_vad": {"error": f"执行器异常: {exc}"}}

                if i % SAVE_CHECKPOINT_INTERVAL == 0:
                    save_results(list(processed_results.values()), RESULTS_FILE)
                    tqdm.write(f"  💾 已保存检查点。当前共记录 {len(processed_results)} 条结果。")

    except KeyboardInterrupt:
        print("\n⚠️ 检测到用户中断，正在保存当前进度...")
    finally:
        save_results(list(processed_results.values()), RESULTS_FILE)
        print("\n" + "="*70)
        print("🎉 VAD分析处理完成！")
        print(f"   详细结果已保存至: {RESULTS_FILE}")
        print("="*70)

    # 生成最终的评测报告
    generate_summary_report(RESULTS_FILE, SUMMARY_FILE)

if __name__ == "__main__":
    main()