import os
import json
import base64
import time
import re
import collections
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import numpy as np

# ================== Part 1: 配置区域 ==================
# --- API 配置 ---
load_dotenv()
API_KEY = os.getenv("DMXAPI_API_KEY_MY")
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL_NAME = "gpt-4o"

# --- 路径配置 ---
# 【重要】存放您所有 .jpg 和 *_edited_emo.png 图片的文件夹

MODEL = "qwen_image_edit_2509_lora"  # <-- 请根据实际模型名称修改此处
IMAGE_DIR = os.path.join(MODEL, "edited_output")

# 【重要】包含编辑指令的JSON文件
INSTRUCTION_FILE = r"./src/Instructions.json"

# --- 输出与断点续传配置 ---
os.makedirs("./pipeline", exist_ok=True)
EVALUATION_SCORES_FILE = os.path.join(MODEL, "eval", "vie_scores.json")
SUMMARY_REPORT_FILE = os.path.join(MODEL, "eval", "vie_evaluation_report.txt")

# --- 并发配置 ---
MAX_CONCURRENT_REQUESTS = 30

# --- 评估 Prompt (无需修改) ---
EVALUATION_PROMPT = """
RULES:
Two images will be provided: The first being the original image and the second being an edited version of the first. The objective is to evaluate how successfully the editing instruction has been executed in the second image. Note that sometimes the two images might look identical due to the failure of the image edit.

On a scale of 0 to 10:
- **Score 1 (Editing Success)**: A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instructions at all. 10 indicates that the scene in the edited image follows the editing instruction text perfectly.)
- **Score 2 (Degree of Overediting)**: A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimally edited yet effective version of the original.)

Your response MUST be a JSON object that adheres to the following structure:
```json
{
  "scores": [score1, score2],
  "reason": "A concise string explaining the reasoning behind the given scores, highlighting specific observations from the images."
}
Where:
score1 is the Editing Success score (0-10).
score2 is the Degree of Overediting score (0-10).
reason is a concise explanation for the scores.
Do not provide any other text or explanation outside of this JSON object.
"""

# ================== Part 2: 核心功能函数 (无需修改) ==================

def encode_image_to_base64(image_path):
    """将图片编码为Base64字符串。"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            from io import BytesIO
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        return None

def get_evaluation_scores_from_api(client, original_image_path, edited_image_path, instruction):
    """通过API调用获取编辑评估分数。"""
    base64_original = encode_image_to_base64(original_image_path)
    base64_edited = encode_image_to_base64(edited_image_path)

    if not base64_original: return {"error": f"无法编码原始图片: {os.path.basename(original_image_path)}."}
    if not base64_edited: return {"error": f"无法编码编辑后图片: {os.path.basename(edited_image_path)}."}
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EVALUATION_PROMPT},
                        {"type": "text", "text": f"Editing instruction: {instruction}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_original}", "detail": "low"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_edited}", "detail": "low"}}
                    ]
                }
            ],
            max_tokens=200,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("响应中未找到有效的JSON对象", content, 0)
            
            parsed_response = json.loads(json_match.group(0))
            scores = parsed_response.get('scores')
            reason = parsed_response.get('reason', '')

            if isinstance(scores, list) and len(scores) == 2 and all(isinstance(x, (int, float)) for x in scores):
                scores = [int(s) for s in scores]
                return {"scores": scores, "reason": reason}
            else:
                return {"error": f"解析出的分数格式不符合预期 [数字, 数字]: {scores}", "raw_response": content}

        except json.JSONDecodeError:
            return {"error": f"无法从响应中解析出有效的JSON对象: {content}"}
    except Exception as e:
        return {"error": f"发生API错误: {str(e)}"}

# ================== Part 3: 主执行逻辑 (create_task_list 已最终修正) ==================

def create_task_list():
    """
    【最终修正逻辑】
    以实际存在的文件对为主导，反向查找JSON中的指令来构建任务。
    """
    print("--- 步骤 1: 正在生成任务清单 ---")
    if not os.path.exists(INSTRUCTION_FILE):
        print(f"[致命错误] 指令文件未找到: {INSTRUCTION_FILE}"); return None
    if not os.path.isdir(IMAGE_DIR):
        print(f"[致命错误] 图片目录未找到: {IMAGE_DIR}"); return None

    # --- 步骤 A: 扫描图片目录，建立文件索引 ---
    original_files_map = {} # key: base_name, value: full_filename.jpg
    edited_files_map = {}   # key: base_name, value: full_filename_edited_emo.png
    edited_file_regex = re.compile(r"^(.*?)_edited_([a-zA-Z]+)\.png$")

    print(f"正在扫描目录 '{IMAGE_DIR}'...")
    for filename in os.listdir(IMAGE_DIR):
        match = edited_file_regex.match(filename)
        if match:
            base_name = match.group(1)
            edited_files_map[base_name] = filename
        elif filename.lower().endswith('_original.jpg'):
            # 【最终修正点】确保 base_name 是一个字符串
            # 从 'file_original.jpg' 提取 'file'
            base_name = filename.rsplit('_original.jpg', 1)[0]
            original_files_map[base_name] = filename
            
    print(f"扫描完成：找到 {len(original_files_map)} 个 _original.jpg 文件和 {len(edited_files_map)} 个 *_edited_emo.png 文件。")

    # --- 步骤 B: 寻找成对存在的文件 ---
    file_pairs = []
    for base_name, original_filename in original_files_map.items():
        if base_name in edited_files_map:
            file_pairs.append({
                "base_name": base_name,
                "original_filename": original_filename,
                "edited_filename": edited_files_map[base_name]
            })
    print(f"成功找到 {len(file_pairs)} 个实际存在的文件对。")

    # --- 步骤 C: 加载并索引指令JSON ---
    with open(INSTRUCTION_FILE, 'r', encoding='utf-8') as f:
        instruction_data = json.load(f)
    
    # 【最终修正点】确保从 'file.jpg' 提取 'file' 作为key
    instructions_map = {
        os.path.splitext(entry['original_image'])[0]: entry
        for entry in instruction_data if 'original_image' in entry
    }
    print(f"成功加载并索引了 {len(instructions_map)} 条指令。")
    # print一组示例
    sample_keys = list(instructions_map.keys())[:3]
    for key in sample_keys:
        print(f"示例指令键: {key} -> 指令: {instructions_map[key]['edit_prompt'][:50]}...")
    # --- 步骤 D: 结合文件对和指令，生成最终任务清单 ---
    tasks = []
    skipped_count = 0
    for pair in file_pairs:
        base_name = pair['base_name']
        instruction_entry = instructions_map.get(base_name)
        if instruction_entry and 'edit_prompt' in instruction_entry:
            target_emo = edited_file_regex.match(pair['edited_filename']).group(2)
            tasks.append({
                "base_name": base_name,
                "original_filename": pair['original_filename'],
                "edited_filename": pair['edited_filename'],
                "instruction": instruction_entry['edit_prompt'],
                "target_category": target_emo,
            })
        else:
            skipped_count += 1
            
    print(f"成功创建 {len(tasks)} 个评估任务。由于在JSON中找不到指令，跳过了 {skipped_count} 个文件对。")
    print("--- 任务清单生成完毕 ---\n")
    return tasks

def run_evaluation_workflow_concurrent():
    """主函数，负责执行整个并发评估流程。"""
    print("="*25, "开始执行图像编辑评估 (并发流程)", "="*25)

    if not API_KEY:
        print("[致命错误] API密钥 (DMXAPI_API_KEY_MY) 未在.env文件中设置。"); return

    tasks_to_process = create_task_list()
    if not tasks_to_process:
        print("未能生成任何任务，程序退出。"); return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 断点续传逻辑
    processed_files = set()
    existing_results_dict = {}
    if os.path.exists(EVALUATION_SCORES_FILE):
        try:
            with open(EVALUATION_SCORES_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            for item in existing_results:
                key = item.get('edited_filename')
                if key:
                    existing_results_dict[key] = item
                    if 'evaluation_scores' in item and item['evaluation_scores'] is not None:
                        processed_files.add(key)
            print(f"成功加载了 {len(existing_results)} 条已有记录。其中 {len(processed_files)} 条已有有效评估分。")
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告：无法解析已有的结果文件 '{EVALUATION_SCORES_FILE}'，将重新开始。错误: {e}")
    
    tasks_to_run = [task for task in tasks_to_process if task.get('edited_filename') not in processed_files]

    if not tasks_to_run:
        print("\n所有任务均已处理完毕，直接进入报告生成阶段。")
    else:
        print("-" * 50)
        print(f"总任务数: {len(tasks_to_process)}")
        print(f"已处理数量: {len(processed_files)}")
        print(f"剩余待处理: {len(tasks_to_run)}")
        print(f"并发请求数: {MAX_CONCURRENT_REQUESTS}")
        print("-" * 50)

    # 并发执行与结果保存
    final_results_dict = {task['edited_filename']: task for task in tasks_to_process}
    final_results_dict.update(existing_results_dict)
    newly_processed_tasks = []

    try:
        if tasks_to_run:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                future_to_task = {
                    executor.submit(
                        get_evaluation_scores_from_api, client,
                        os.path.join(IMAGE_DIR, task['original_filename']),
                        os.path.join(IMAGE_DIR, task['edited_filename']),
                        task['instruction']
                    ): task for task in tasks_to_run
                }

                for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks_to_run), desc="评估图像编辑效果 (API)"):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if 'error' in result:
                            task['evaluation_scores'] = None
                            task['evaluation_reason'] = ""
                            task['evaluation_error'] = result['error']
                            tqdm.write(f"\n[警告] 处理 '{task['edited_filename']}' 失败: {result['error']}")
                        else:
                            task['evaluation_scores'] = result['scores']
                            task['evaluation_reason'] = result.get('reason', '')
                        
                        newly_processed_tasks.append(task)
                    except Exception as exc:
                        task['evaluation_scores'] = None
                        task['evaluation_reason'] = ""
                        task['evaluation_error'] = f"处理API调用时发生未捕获异常: {str(exc)}"
                        tqdm.write(f"\n[严重错误] 处理 '{task['edited_filename']}' 时发生异常: {exc}")

    except KeyboardInterrupt:
        print("\n检测到用户中断 (Ctrl+C)，将在退出前保存当前进度...")
    finally:
        print("\n评估计算完成。正在保存结果...")
        try:
            for task in newly_processed_tasks:
                final_results_dict[task['edited_filename']] = task

            final_results_list = list(final_results_dict.values())
            with open(EVALUATION_SCORES_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=4, ensure_ascii=False)
            print(f"所有评估结果已成功保存至: '{EVALUATION_SCORES_FILE}'")
            
            calculate_and_save_report(EVALUATION_SCORES_FILE, SUMMARY_REPORT_FILE)

        except Exception as e:
            print(f"[错误] 保存结果文件或生成报告时失败: {e}")
        print("="*25, "评估流程完成", "="*25)

# ================== Part 4: 报告生成函数 (无需修改) ==================

# ================== Part 4: 报告生成函数 (已修正) ==================

def calculate_and_save_report(scores_filepath, report_filepath):
    """
    读取评估分数JSON文件，计算平均分和VIE-Score，并生成一个易于阅读的文本报告。
    """
    print("\n" + "📊" * 35)
    print("正在生成最终的评估总结报告...")
    print("📊" * 35)

    try:
        with open(scores_filepath, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"❌ 无法加载或解析评估分数文件: {scores_filepath}，无法生成报告。")
        return

    valid_results = [
        res for res in all_results
        if isinstance(res.get('evaluation_scores'), list) and len(res.get('evaluation_scores')) == 2
    ]

    if not valid_results:
        print("❌ 没有找到有效的评估分数，无法生成报告。")
        return

    scores_by_category = collections.defaultdict(list)
    # =========== 从这里开始是核心修正区域 ===========
    for res in valid_results:
        category = res.get('target_category', 'unknown')
        scores = res['evaluation_scores'] # scores 是一个列表, e.g., [8, 9]
        
        # 【修正 1】正确地从列表中提取单个分数
        score_success = scores[0]
        score_fidelity = scores[1]
        
        # 【修正 2】使用提取出的数字分数进行计算
        vie_score = (score_success + score_fidelity) / 2
        
        # 【修正 3】将单个数字分数存入字典
        scores_by_category[category].append({
            "success": score_success,
            "fidelity": score_fidelity,
            "vie_score": vie_score
        })
    # =========== 修正区域结束 ===========

    report = [
        "=" * 80,
        f"图像编辑质量评估报告 (VIE-Score) - 模型: {MODEL}", # 使用全局变量 MODEL
        "=" * 80,
        f"总评估图片数 (有效): {len(valid_results)} / {len(all_results)}",
        f"VIE-Score 计算公式: (编辑成功分 (Score 1) + 编辑保真分 (Score 2)) / 2",
        
        "\n--- 1. 总体平均分 ---"
    ]

    # 下面的代码现在可以正常工作了，因为 'success' 和 'fidelity' 都是数字
    all_success = [item['success'] for cat_data in scores_by_category.values() for item in cat_data]
    all_fidelity = [item['fidelity'] for cat_data in scores_by_category.values() for item in cat_data]
    all_vie = [item['vie_score'] for cat_data in scores_by_category.values() for item in cat_data]

    report.append(f"  - 平均编辑成功分 (Score 1):     {np.mean(all_success):.3f}")
    report.append(f"  - 平均编辑保真分 (Score 2):     {np.mean(all_fidelity):.3f} (分数越高代表对原图改动越小且有效)")
    report.append(f"  - 平均 VIE-Score (综合得分):    {np.mean(all_vie):.3f} (越高越好)")

    report.append("\n--- 2. 按目标情感类别分类的平均分 ---")
    header = f"{'情感类别':<15} | {'平均成功分':>12} | {'平均保真分':>12} | {'平均VIE-Score':>15} | {'数量':>7}"
    report.append(header)
    report.append("-" * len(header))

    for category, data in sorted(scores_by_category.items()):
        count = len(data)
        avg_success = np.mean([item['success'] for item in data])
        avg_fidelity = np.mean([item['fidelity'] for item in data])
        avg_vie = np.mean([item['vie_score'] for item in data])
        report.append(f"{category:<15} | {avg_success:>12.3f} | {avg_fidelity:>12.3f} | {avg_vie:>15.3f} | {count:>7}")

    report_str = "\n".join(report)
    print("\n--- 报告预览 ---")
    print(report_str)
    print("--- 报告预览结束 ---")

    try:
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"\n✅ 评估报告已成功保存至: '{report_filepath}'")
    except Exception as e:
        print(f"\n❌ 保存报告文件失败: {e}")

# ================== 脚本入口 ==================
if __name__ == "__main__":
    run_evaluation_workflow_concurrent()