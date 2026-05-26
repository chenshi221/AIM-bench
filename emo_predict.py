import os
import json
import base64
import time
import re
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageFile
from tqdm import tqdm
import concurrent.futures
import collections

# 允许加载可能被截断的图像文件，增加代码的健壮性
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =============================================================================
# --- 1. 配置区域 (请根据您的环境修改) ---
# =============================================================================

# --- 输入/输出路径配置 ---
# 假设您的项目结构如下:
# /YourProject/
# |-- Qwen-Image-Edit-Plus/
# |   |-- edited_output/  <-- 图片放在这里
# |   |-- eval/           <-- 结果会输出到这里
# |-- this_script.py
# |-- .env
MODEL = "qwen_image_edit_2509_lora"
IMAGE_DIR = os.path.join(MODEL, "edited_output")  # 存放 *_edited_emo.png 图片的文件夹

# --- 文件名匹配模式配置 ---
# 正则表达式用于匹配文件名并提取情感 (emo) 部分
# 例如: 'xxx_edited_amusement.png' 会匹配成功, 并提取出 'amusement'
# 请确保您的文件名遵循这个格式
EMOTIONS_LIST = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
FILE_REGEX = re.compile(r".*_edited_(" + "|".join(EMOTIONS_LIST) + r")\.png$")

# --- API 配置 ---
load_dotenv()
API_KEY = os.getenv("DMXAPI_API_KEY_MY")
BASE_URL = "https://www.dmxapi.cn/v1"

# --- 模型配置 ---
MODEL_NAME = "gemini-2.5-pro"

# --- 输出文件名配置 ---
RESULTS_DIR = os.path.join(MODEL, "eval")
RESULTS_FILE = os.path.join(RESULTS_DIR, "evaluation_results.json")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "evaluation_summary.txt")


# --- 运行控制 ---
MAX_CONCURRENT_REQUESTS = 40  # 并发请求数
SAVE_CHECKPOINT_INTERVAL = 20  # 每处理N张图片保存一次

# --- 重试配置 ---
MAX_RETRIES = 3  # 每张图片最大重试次数
RETRY_DELAY = 2  # 重试间隔(秒)

# --- VLM模型的视觉情感分类指令 (Prompt) ---
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
  "reasoning": "A concise analysis directly linking specific visual elements to the chosen emotion. For example: 'The subject's wide grin and the playful, bright setting strongly suggest amusement.'"
}
"""

# =============================================================================
# --- 2. 核心功能函数 ---
# =============================================================================

def get_image_files_and_emotions(directory):
    """获取图片文件列表, 并从文件名中提取目标情感"""
    matched_files = []
    for filename in sorted(os.listdir(directory)):
        match = FILE_REGEX.match(filename)
        if match:
            target_emotion = match.group(1)
            matched_files.append({"filename": filename, "target_emotion": target_emotion})
    return matched_files

def encode_image_to_base64(image_path):
    """将图片文件编码为Base64字符串"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            from io import BytesIO
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception:
        return None

def validate_response(data):
    """验证API响应是否为有效的JSON且包含必要字段"""
    if not isinstance(data, dict):
        return False, "Response is not a dictionary"
    if "error" in data:
        return False, data["error"]
    required = ["emotion", "confidence", "reasoning"]
    if not all(k in data for k in required):
        return False, f"Missing fields: {', '.join(k for k in required if k not in data)}"
    if data["emotion"] not in EMOTIONS_LIST:
        return False, f"Invalid emotion: {data['emotion']}"
    try:
        if not (0.0 <= float(data["confidence"]) <= 1.0):
            return False, f"Confidence out of range: {data['confidence']}"
    except (ValueError, TypeError):
        return False, f"Invalid confidence format: {data['confidence']}"
    return True, None


def analyze_image_emotion(client, image_path, model_name, retry_count=0):
    """使用VLM分析单张图片的情感，带重试机制"""
    encoded_data = encode_image_to_base64(image_path)
    if not encoded_data:
        return {"error": f"Failed to encode image: {image_path}"}

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_data}"}}
                ]}
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in response", content, 0)

        parsed_data = json.loads(json_match.group(0))
        is_valid, error_msg = validate_response(parsed_data)
        if not is_valid:
            raise ValueError(f"Invalid response structure: {error_msg}")
        return parsed_data

    except (json.JSONDecodeError, ValueError, Exception) as e:
        if retry_count < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return analyze_image_emotion(client, image_path, model_name, retry_count + 1)
        return {"error": f"{type(e).__name__}: {str(e)} (after {MAX_RETRIES} retries)"}

def save_results(all_data, filepath):
    """将所有结果安全地保存到JSON文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"\n[错误] 保存结果文件 '{filepath}' 失败: {e}")

def process_single_image(client, image_info, model_name):
    """处理单张图片（用于多线程）"""
    filename = image_info["filename"]
    target_emotion = image_info["target_emotion"]
    image_path = os.path.join(IMAGE_DIR, filename)

    try:
        if not os.path.exists(image_path):
            analysis_result = {"error": "File not found"}
        else:
            analysis_result = analyze_image_emotion(client, image_path, model_name)

        result = {
            "image_name": filename,
            "target_emotion": target_emotion,
        }

        if "error" in analysis_result:
            result.update({
                "predicted_emotion": "error",
                "is_correct": False,
                "details": analysis_result
            })
            return {"status": "error", "data": result}
        else:
            predicted = analysis_result.get("emotion")
            result.update({
                "predicted_emotion": predicted,
                "is_correct": target_emotion == predicted,
                "details": analysis_result
            })
            return {"status": "success", "data": result}

    except Exception as exc:
        result = {
            "image_name": filename,
            "target_emotion": target_emotion,
            "predicted_emotion": "exception",
            "is_correct": False,
            "details": {"error": str(exc)}
        }
        return {"status": "error", "data": result}

# =============================================================================
# --- 3. 结果分析与报告 ---
# =============================================================================

def calculate_and_save_accuracy(results_filepath, summary_filepath):
    """从结果文件中计算准确率并生成总结报告"""
    print("\n" + "📊" * 35)
    print("生成评估报告...")
    print("📊" * 35)

    try:
        with open(results_filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"❌ 无法加载或解析结果文件: {results_filepath}")
        return

    valid_results = [r for r in results if r.get("predicted_emotion") not in ["error", "exception"]]
    if not valid_results:
        print("❌ 没有找到有效的处理结果，无法生成报告。")
        return

    total_correct = sum(1 for r in valid_results if r['is_correct'])
    total_processed = len(valid_results)
    overall_accuracy = (total_correct / total_processed) * 100 if total_processed > 0 else 0

    # 按情感分类统计
    emotion_stats = collections.defaultdict(lambda: {'correct': 0, 'total': 0})
    # 混淆矩阵
    confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(int))

    for r in valid_results:
        target = r['target_emotion']
        predicted = r['predicted_emotion']
        emotion_stats[target]['total'] += 1
        confusion_matrix[target][predicted] += 1
        if r['is_correct']:
            emotion_stats[target]['correct'] += 1

    # --- 生成报告文本 ---
    report = []
    report.append("="*70)
    report.append(f"情感分类模型评估报告 ({MODEL_NAME})")
    report.append("="*70)
    report.append(f"总图片数 (有效处理): {total_processed}")
    report.append(f"总正确数: {total_correct}")
    report.append(f"▶ 总体正确率: {overall_accuracy:.2f}%")
    report.append("\n" + "-"*70)
    report.append("各项情感准确率:")
    report.append("-" * 70)

    for emotion in EMOTIONS_LIST:
        stats = emotion_stats[emotion]
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        report.append(f"- {emotion:<12}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

    # --- 混淆矩阵 ---
    report.append("\n" + "="*70)
    report.append("混淆矩阵 (行: 真实情感, 列: 预测情感)")
    report.append("="*70)
    header = f"{'':<12}" + "".join([f"{emo[:4]:>6}" for emo in EMOTIONS_LIST])
    report.append(header)
    report.append("-" * len(header))

    for target_emo in EMOTIONS_LIST:
        row = f"{target_emo:<12}"
        for pred_emo in EMOTIONS_LIST:
            count = confusion_matrix[target_emo][pred_emo]
            row += f"{count:>6}"
        report.append(row)

    report_str = "\n".join(report)
    print(report_str)

    try:
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"\n✅ 评估报告已保存到: {summary_filepath}")
    except Exception as e:
        print(f"\n❌ 保存评估报告失败: {e}")


# =============================================================================
# --- 4. 主处理逻辑 ---
# =============================================================================

def main():
    """主执行函数"""
    print("="*30, "VLM情感分类评估任务", "="*30)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.isdir(IMAGE_DIR):
        print(f"❌ [致命错误] 找不到图片文件夹 '{IMAGE_DIR}'。")
        return

    image_infos = get_image_files_and_emotions(IMAGE_DIR)
    if not image_infos:
        print(f"❌ [致命错误] 在 '{IMAGE_DIR}' 中未找到符合 '{FILE_REGEX.pattern}' 格式的图片文件！")
        return

    print(f"✅ 在 '{IMAGE_DIR}' 中找到 {len(image_infos)} 张待评估图片。")
    print(f"   示例: {image_infos[0]['filename']} (目标情感: {image_infos[0]['target_emotion']})")

    # 加载已有结果
    processed_images = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            for item in existing_results:
                processed_images[item.get('image_name')] = item
            print(f"✓ 成功加载了 {len(processed_images)} 条已有结果。")
        except json.JSONDecodeError:
            print(f"⚠ 无法解析已有结果文件 '{RESULTS_FILE}'，将重新开始。")

    # 确定需要处理的图片 (未处理或之前处理失败的)
    images_to_process = [
        info for info in image_infos
        if info['filename'] not in processed_images or
           processed_images.get(info['filename'], {}).get('predicted_emotion') in ["error", "exception"]
    ]

    if not images_to_process:
        print("\n✅ 所有图片均已成功处理！现在开始生成评估报告。")
        calculate_and_save_accuracy(RESULTS_FILE, SUMMARY_FILE)
        return

    print(f"\n📊 待处理图片: {len(images_to_process)} / {len(image_infos)}")
    print(f"   模型: {MODEL_NAME}, 并发数: {MAX_CONCURRENT_REQUESTS}")
    print(f"   结果将保存至: {RESULTS_FILE}")

    if API_KEY is None:
        print("\n❌ [致命错误] API密钥 (DMXAPI_API_KEY_MY) 未在.env文件中设置。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    success_count = 0
    failed_count = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            future_to_info = {executor.submit(process_single_image, client, info, MODEL_NAME): info for info in images_to_process}

            pbar = tqdm(concurrent.futures.as_completed(future_to_info), total=len(images_to_process), desc="评估进度")
            for future in pbar:
                try:
                    result = future.result()
                    processed_images[result["data"]["image_name"]] = result["data"]

                    if result["status"] == "success":
                        success_count += 1
                    else:
                        failed_count += 1
                        error_msg = result["data"].get("details", {}).get("error", "未知错误")
                        tqdm.write(f"  ✗ {result['data']['image_name']}: {error_msg}")

                    if (success_count + failed_count) % SAVE_CHECKPOINT_INTERVAL == 0:
                        save_results(list(processed_images.values()), RESULTS_FILE)

                except Exception as exc:
                    info = future_to_info[future]
                    failed_count += 1
                    tqdm.write(f"  ✗ 处理 {info['filename']} 时发生致命异常: {exc}")
                    processed_images[info['filename']] = {
                        "image_name": info['filename'], "target_emotion": info['target_emotion'],
                        "predicted_emotion": "exception", "is_correct": False,
                        "details": {"error": f"Executor exception: {str(exc)}"}
                    }

    except KeyboardInterrupt:
        print("\n⚠ 检测到中断，正在保存当前进度...")
    finally:
        save_results(list(processed_images.values()), RESULTS_FILE)
        print("\n" + "="*70)
        print("🎉 情感分类处理完成！")
        print(f"  ▶ 本次成功: {success_count}, 失败: {failed_count}")
        print(f"  ▶ 结果文件: {RESULTS_FILE}")
        print("="*70)

    # 最后，无论处理是否中断，都尝试生成一次报告
    calculate_and_save_accuracy(RESULTS_FILE, SUMMARY_FILE)

if __name__ == "__main__":
    main()