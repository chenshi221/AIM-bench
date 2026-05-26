import os
import json
from PIL import Image
import torch
from tqdm import tqdm  # 假设在标准环境中，如果notebook则用 .notebook
import warnings

# ================== Part 0: 导入我们的新包装器 ==================
# 确保 longclip_wrapper.py 和 model/ 目录都在可访问的路径中
from LongCLIP.longclip_wrapper import LongCLIPWrapper

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

# ================== 封装函数 ==================
# 这个函数封装了整个评估逻辑，只需要传入 MODEL_NAME 即可调用
def evaluate_with_longclip(model_name):
    """
    使用 LongCLIP 对指定模型的编辑图像进行指令对齐评分。
    
    参数:
    model_name (str): 模型名称，例如 "Qwen-Image-Edit-Plus"，用于构建路径。
    """
    # ================== Part 1: 配置区域 ==================
    # 评估的模型
    MODEL_NAME = model_name

    # --- 模型路径 ---
    LONGCLIP_MODEL_PATH = r"./LongCLIP/checkpoints/longclip-L.pt" 

    # --- 输入路径 ---
    EDITED_IMAGES_DIR = os.path.join(MODEL_NAME, "edited_output")
    INSTRUCTION_FILE = r"./src/Instructions.json"

    # --- 输出配置 ---
    # 确认有eval目录，没有就创建
    if not os.path.exists(os.path.join(MODEL_NAME, "eval")):
        os.makedirs(os.path.join(MODEL_NAME, "eval"))
    CLIP_SCORES_FILE = os.path.join(MODEL_NAME, "eval", "longclip_scores.json")

    # ================== Part 2: 执行逻辑 (函数部分不变) ==================

    def build_prompts_database(instruction_file_path):
        """加载指令文件并构建一个易于查询的字典。"""
        print(f"正在从 '{instruction_file_path}' 构建指令数据库...")
        try:
            with open(instruction_file_path, 'r', encoding='utf-8') as f:
                all_tasks = json.load(f)
        except FileNotFoundError:
            print(f"[错误] 指令文件未找到: {instruction_file_path}")
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
        print(f"数据库构建完成，共找到 {len(prompts_db)} 条有效的指令。")
        return prompts_db

    def run_clip_scoring():
        print("="*25, "Part 1: 开始执行 LongCLIP 指令对齐评分", "="*25)

        if not os.path.exists(LONGCLIP_MODEL_PATH):
            print(f"[致命错误] LongCLIP模型文件不存在: {LONGCLIP_MODEL_PATH}")
            return
        if not os.path.isdir(EDITED_IMAGES_DIR):
            print(f"[致命错误] 待评估的图片目录不存在: {EDITED_IMAGES_DIR}")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # <--- MODIFIED: 使用我们的包装器加载模型
        # model 和 processor 被一个统一的 scorer 对象替代
        try:
            longclip_scorer = LongCLIPWrapper(LONGCLIP_MODEL_PATH, device)
        except Exception as e:
            print(f"[致命错误] 加载 LongCLIP 模型失败: {e}")
            return

        prompts_database = build_prompts_database(INSTRUCTION_FILE)
        if not prompts_database: return

        all_scores = []
        image_files = [f for f in os.listdir(EDITED_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'edited' in f]
        
        # <--- MODIFIED: 这里的 with torch.no_grad() 已被移入包装器内部，但保留也无妨
        with torch.no_grad():
            for filename in tqdm(image_files, desc="计算LongCLIP相似度分数"):
                prompt_text = prompts_database.get(filename)
                if not prompt_text: continue
                image_path = os.path.join(EDITED_IMAGES_DIR, filename)
                try:
                    image = Image.open(image_path).convert("RGB")
                    
                    outputs = longclip_scorer(text=[prompt_text], images=image)

                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                    
                    similarity = (image_embeds @ text_embeds.T).squeeze().item()
                    all_scores.append({"edited_filename": filename, "longclip_similarity_score": round(similarity, 4)})
                except Exception as e:
                    print(f"\n[警告] 处理文件 '{filename}' 时出错: {e}")

        if not all_scores:
            print("\n处理完成，但没有计算出任何有效分数。")
            return
            
        print(f"\n计算完成，总共评估了 {len(all_scores)} 张图片。")
        sorted_scores = sorted(all_scores, key=lambda x: x['edited_filename'])
        
        try:
            with open(CLIP_SCORES_FILE, 'w', encoding='utf-8') as f:
                json.dump(sorted_scores, f, indent=4, ensure_ascii=False)
            print(f"LongCLIP评估结果已成功保存至: '{CLIP_SCORES_FILE}'")
        except Exception as e:
            print(f"[错误] 保存结果文件失败: {e}")
        print("="*25, "Part 1 完成", "="*25)

    # 执行评分
    run_clip_scoring()
    
    # 返回平均分数以供参考
    try:
        with open(CLIP_SCORES_FILE, 'r', encoding='utf-8') as f:
            scores_data = json.load(f)
        scores = [item['longclip_similarity_score'] for item in scores_data]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"平均 LongCLIP 相似度分数: {avg_score:.4f}")
        return avg_score
    
    except Exception as e:
        print(f"[错误] 读取评分结果失败: {e}")
        return None

# ================== 调用示例 ==================
# 使用传入的模型名称调用函数
if __name__ == "__main__":

    evaluate_with_longclip("Qwen-Image-Edit-Plus")
