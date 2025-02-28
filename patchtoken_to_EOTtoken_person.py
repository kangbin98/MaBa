import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPTokenizer
from model import build_model
import json

class SimilarityVisualizer:
    def __init__(self, model, device):
        """
        初始化可视化工具
        :param model: 加载好的MaBa模型
        :param device: 计算设备（'cuda' 或 'cpu'）
        """
        self.model = model
        self.device = device

        # 获取视觉模型的一些参数
        self.visual_model = model.base_model.visual
        self.patch_size = self.visual_model.patch_size
        self.stride_size = self.visual_model.stride_size
        self.input_resolution = self.visual_model.input_resolution
        self.num_x = (self.input_resolution[1] - self.patch_size) // self.stride_size + 1
        self.num_y = (self.input_resolution[0] - self.patch_size) // self.stride_size + 1

        # 图像预处理流程
        self.transform = Compose([
            Resize(self.input_resolution),
            CenterCrop(self.input_resolution),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def preprocess_text(self, text, tokenizer, context_length=77):
        """
        文本预处理（适配HuggingFace Tokenizer）
        :param text: 输入文本字符串
        :param tokenizer: CLIPTokenizer实例
        :return: 处理后的文本张量 [1, context_length]
        """
        # 将所有文本统一为 "person"
        text = "person"
        inputs = tokenizer(text, max_length=context_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids.to(self.device)

    def compute_similarity_map(self, image_tensor, text_tensor):
        """
        计算每个patch与文本之间的相似度，并生成相似度热力图
        :return: 相似度矩阵 (num_y, num_x)
        """
        with torch.no_grad():
            # 获取图像特征（[batch_size, num_patches + 1, feature_dim]）
            image_features = self.model.encode_image(image_tensor)  # 形状 [1, 197, 512]

            # 获取文本特征
            text_features = self.model.encode_text(text_tensor)  # [batch_size, feature_dim]

        # 确保提取每个patch的特征（忽略CLS token）
        patch_features = image_features[:, 1:, :]  # 形状 [batch_size, num_patches, feature_dim]

        # 计算余弦相似度
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算每个patch与文本的相似度
        similarity = (patch_features @ text_features.unsqueeze(-1)).squeeze(-1)

        # 返回相似度矩阵，并确保形状为 (num_y, num_x)
        similarity_map = similarity.squeeze().cpu().numpy().reshape(self.num_y, self.num_x)

        return similarity_map

    def generate_heatmap(self, similarity_map, original_image):
        """
        生成热力图叠加效果
        :return: 叠加后的热力图图像 (H, W, 3)
        """
        # 获取原图的尺寸
        img_width, img_height = original_image.size

        # 将相似度图调整到原图的大小
        heatmap = cv2.resize(similarity_map, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

        # 归一化处理
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)

        # 应用颜色映射
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 叠加热力图与原图
        overlay = cv2.addWeighted(np.array(original_image), 0.6, heatmap, 0.4, 0)

        return heatmap, overlay

    def visualize(self, image_path, text, tokenizer, output_dir):
        """
        完整可视化流程
        :param image_path: 图像路径
        :param text: 查询文本
        :param tokenizer: 文本tokenizer（CLIPTokenizer实例）
        :param output_dir: 输出目录
        """
        try:
            # 预处理输入
            raw_image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(raw_image).unsqueeze(0).to(self.device)
            text_tensor = self.preprocess_text(text, tokenizer)

            # 计算相似度
            similarity_map = self.compute_similarity_map(image_tensor, text_tensor)

            # 生成可视化
            heatmap, overlay = self.generate_heatmap(similarity_map, raw_image)

            # 保存结果
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(image_path))

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(raw_image)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Heatmap")
            plt.imshow(heatmap)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Overlay")
            plt.imshow(overlay)
            plt.axis('off')

            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return  # Skip this image if there's an error


def process_dataset_and_visualize(model, tokenizer, data_file, output_dir, device, image_root_dir):
    with open(data_file, "r") as f:
        data = json.load(f)

    visualizer = SimilarityVisualizer(model, device)

    for entry in data:
        captions = entry.get("captions", [])
        image_path = entry.get("file_path", "")

        # Construct the full image path
        full_image_path = os.path.join(image_root_dir, image_path)

        if not os.path.exists(full_image_path):  # Skip image if it doesn't exist
            print(f"Image not found: {full_image_path}")
            continue

        if captions:  # Check if captions exist
            caption = captions[0]  # Take the first caption
            visualizer.visualize(
                image_path=full_image_path,
                text=caption,
                tokenizer=tokenizer,
                output_dir=output_dir
            )
        else:
            print(f"No captions available for image: {image_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # 定义模型参数类
    class Args:
        pretrain_choice = 'ViT-B/16'
        img_size = (224, 224)
        stride_size = 16
        temperature = 0.07
        loss_names = 'itc+id+mlm'
        vocab_size = 49408
        cmt_depth = 2
        id_loss_weight = 1.0
        mlm_loss_weight = 1.0
        context_length = 77  # 添加 context_length
        output_dir = 'visualization_person_results'  # 确保与图像可视化代码一致
        data_file = 'data/CUHK-PEDES/reid_raw.json'  # 这里指定你的数据文件路径
        image_root_dir = '/home/kb/TBPR/MaBa/data/CUHK-PEDES/imgs/'  # 图像的根目录


    args = Args()
    # 加载模型
    model = build_model(args).to(device).eval()

    # 初始化HuggingFace的CLIP Tokenizer
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 处理数据集并可视化
    process_dataset_and_visualize(model, clip_tokenizer, args.data_file, args.output_dir, device, args.image_root_dir)
