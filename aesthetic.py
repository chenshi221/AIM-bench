from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json
from warnings import filterwarnings
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm
from os.path import join
import clip
from PIL import Image, ImageFile
import csv
import torch.nn.functional as F

# 忽略警告
filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 支持的图片格式
SUPPORTED_FORMATS = {'.png'}

# MLP 模型定义（与训练时保持一致）
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    """归一化函数：将向量标准化为单位向量"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_image_files(directory):
    """获取目录下所有支持格式的图片文件"""
    image_files = []
    if os.path.isfile(directory):
        if os.path.splitext(directory)[1].lower() in SUPPORTED_FORMATS:
            return [directory]
        else:
            print(f"警告: {directory} 不是支持的图片格式，跳过")
            return []
    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_FORMATS:
                image_files.append(os.path.join(root, file))
    return sorted(image_files)


def parse_emotions_from_filename(filename):
    """
    从标准格式的文件名中解析原始情感和目标情感。
    - 原始图片: amusement_..._original.png -> original='amusement', target='amusement'
    - 编辑后图片: ..._sadness_instruction_1.png -> original='amusement', target='sadness'
    """
    parts = filename.split('_')
    if not parts:
        return 'unknown', 'unknown'

    original_emo = parts[0]
    target_emo = original_emo

    if "_instruction_" in filename:
        try:
            target_emo = parts[-3]
        except IndexError:
            target_emo = 'parse_error'
            
    return original_emo, target_emo


def predict_aesthetic_score(image_path, clip_model, preprocess, mlp_model, device):
    """预测单张图片的美学评分"""
    try:
        pil_image = Image.open(image_path).convert('RGB')
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = mlp_model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        return prediction.item()
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return None


def get_average_aesthetic_score(modelname):
    """封装的函数：接受 modelname 作为参数，处理其 edited_output 目录下的 PNG 图像，返回平均美学评分，并保存结果到 modelname/eval/aesthetic.json"""
    img_directory = os.path.join(modelname, "edited_output")
    
    print("=" * 70)
    print("美学评分与情感分析器 - 封装函数模式")
    print(f"模型名称: {modelname}")
    print("=" * 70)
    
    # 1. 加载 MLP 模型
    print("\n[1/4] 加载 MLP 模型...")
    model = MLP(768)
    if os.path.exists("./improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"):
        s = torch.load("./improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")
        model.load_state_dict(s)
    else:
        print("错误: MLP 模型文件未找到!")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    model.to(device)
    model.eval()
    
    # 2. 加载 CLIP 模型
    print("\n[2/4] 加载 CLIP 模型...")
    clip_model, preprocess = clip.load("./ViT-L-14.pt", device=device)
    
    # 3. 获取所有图片文件 (只处理 PNG)
    print(f"\n[3/4] 扫描目录: {img_directory}")
    image_files = get_image_files(img_directory)
    if not image_files:
        print("错误: 未找到任何支持的 PNG 图片文件！")
        return None
    print(f"找到 {len(image_files)} 张 PNG 图片")
    
    # 4. 批量处理图片
    print("\n[4/4] 开始处理图片...")
    results = []
    
    for img_path in tqdm.tqdm(image_files, desc="处理进度"):
        score = predict_aesthetic_score(img_path, clip_model, preprocess, model, device)
        if score is not None:
            image_name = os.path.basename(img_path)
            results.append({
                'name': image_name,
                'score': round(score, 4),
            })
    
    # 5. 计算平均分并保存结果
    if results:
        scores = [r['score'] for r in results]
        avg_score = np.mean(scores)
        print("\n" + "=" * 70)
        print("处理完成！")
        print("=" * 70)
        print(f"成功处理: {len(results)} 张")
        print(f"平均评分: {avg_score:.4f}")
        
        # 保存到 modelname/eval/aesthetic.json
        eval_dir = os.path.join(modelname, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        output_json = os.path.join(eval_dir, "aesthetic.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 详细结果已保存到: {output_json}")
        
        return avg_score
    else:
        print("❌ 没有成功处理任何图片")
        return None


if __name__ == "__main__":
    # 示例调用（当作为脚本运行时，可以修改以测试）
    model_names = ["Qwen-Image-Edit-Plus"]  # 示例，可以修改
    for model in model_names:
        avg = get_average_aesthetic_score(model)
        print(f"模型 {model} 的平均分: {avg}")
