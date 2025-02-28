from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model.MaBa import MaBa_model
import argparse
import logging
from utils.iotools import load_train_configs
import os.path as op
from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging
import os
from PIL import Image, ImageDraw, ImageFont
def ensure_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    # Ensure that q_pids and g_pids are 1D LongTensors
    if q_pids.dim() != 1:
        q_pids = q_pids.view(-1)
    if g_pids.dim() != 1:
        g_pids = g_pids.view(-1)
    q_pids = q_pids.long()
    g_pids = g_pids.long()

    # Debugging prints
    # print(f"q_pids shape: {q_pids.shape}, dtype: {q_pids.dtype}")
    # print(f"g_pids shape: {g_pids.shape}, dtype: {g_pids.dtype}")

    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # accelerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk

    # Ensure indices is a LongTensor
    indices = indices.long()

    # Debugging prints
    # print(f"indices shape: {indices.shape}, dtype: {indices.dtype}")

    # Check if indices are within range
    if (indices >= g_pids.size(0)).any():
        raise ValueError("Some indices are out of range for g_pids")

    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader  # gallery
        self.txt_loader = txt_loader  # query
        self.logger = logging.getLogger("MaBa.eval")
        self.visualize_info = []  # 存储用于可视化的信息

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # 处理文本（查询）
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            # 确保 pid 是 LongTensor 并且是 1D
            if isinstance(pid, torch.Tensor):
                pid = pid.view(-1).long()
            else:
                pid = torch.tensor([pid], dtype=torch.long)
            qids.append(pid)
            qfeats.append(text_feat)
            # print(f"Processed text PID: {pid}, caption shape: {caption.shape}")
        qids = torch.cat(qids, 0).view(-1).long()
        qfeats = torch.cat(qfeats, 0)

        # 处理图像（画廊）
        for idx, (pid, img) in enumerate(self.img_loader):
            img = img.to(device)
            with torch.no_grad():
                # 假设 encode_image 返回 (cls_feat, layer_tokens)
                img_feat, layer_tokens = model.encode_image(img) if hasattr(model, 'encode_image') else (
                model.encode_image(img), None)
            # 确保 pid 是 LongTensor 并且是 1D
            if isinstance(pid, torch.Tensor):
                pid = pid.view(-1).long()
            else:
                pid = torch.tensor([pid], dtype=torch.long)
            gids.append(pid)
            gfeats.append(img_feat)
            # print(f"Processed image PID: {pid}, image shape: {img.shape}")
            # 如果需要可视化，保存 layer_tokens
            if hasattr(model, 'visualize_similarities') and layer_tokens is not None:
                # 增加可视化图像的数量限制
                for i in range(img.size(0)):
                    if len(self.visualize_info) >= 10:
                        break
                    single_img = img[i].unsqueeze(0)  # [1, 3, H, W]
                    single_pid = pid[i]
                    single_layer_tokens = {k: v[i] for k, v in layer_tokens.items()}  # for each layer, get ith token
                    self.visualize_info.append((single_img, single_layer_tokens, single_pid))
                    print(f"Added image {i + 1} for visualization with PID: {single_pid.item()}")
        gids = torch.cat(gids, 0).view(-1).long()
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids

    def concatenate_visualizations(self, save_dir, output_filename="concatenated_visualizations.png", gap=10):

        images = [Image.open(os.path.join(save_dir, fname)) for fname in sorted(os.listdir(save_dir)) if fname.endswith('.png')]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths) + gap * (len(images) - 1)
        max_height = max(heights) + 50  # 增加底部空间以放置文本

        new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        x_offset = 0
        for idx, im in enumerate(images):
            new_img.paste(im, (x_offset, 0))
            draw = ImageDraw.Draw(new_img)
            text = f"Layer {idx+1}"
            draw.text((x_offset + im.width//2 - 30, max_height - 40), text, fill=(0, 0, 0))

            x_offset += im.width + gap  # 增加间隙

        new_img.save(os.path.join(save_dir, output_filename))
        print(f"Visualizations concatenated and saved to {os.path.join(save_dir, output_filename)}")

    def eval(self, model, i2t_metric=False, visualize=False, save_dir='data/visualization'):
        """
        评估模型，并可选择性地进行可视化。

        参数:
            model (nn.Module): IRRA 模型
            i2t_metric (bool): 是否计算图像到文本的检索指标
            visualize (bool): 是否进行相似度可视化
            save_dir (str): 保存热力图的目录
        """
        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
        gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

        similarity = qfeats @ gfeats.t()

        # 调用 rank 函数并捕获可能的错误
        try:
            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        except Exception as e:
            self.logger.error(f"Error in ranking: {e}")
            raise e

        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            try:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            except Exception as e:
                self.logger.error(f"Error in ranking (i2t): {e}")
                raise e
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        # 设置表格格式
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))

        if visualize and self.visualize_info:
            self.logger.info(f"Number of images to visualize: {len(self.visualize_info)}")
            ensure_dir(save_dir)
            self.logger.info(f"Generating visualizations and saving to {save_dir}...")
            for idx, (img, layer_tokens, pid) in enumerate(self.visualize_info):
                self.logger.info(f"Visualizing image {idx + 1} with PID: {pid.item()}")
                try:
                    model.visualize_similarities(img, device=img.device, save_dir=save_dir)
                    self.logger.info(f"Visualized similarities for PID: {pid.item()}")
                except Exception as e:
                    self.logger.error(f"Error in visualizing similarities for PID {pid.item()}: {e}")

            # 生成所有单独的热力图后拼接它们
            self.concatenate_visualizations(save_dir)

        else:
            self.logger.info("No visualizations to generate.")

        return t2i_cmc[0]


def do_inference(model, test_img_loader, test_txt_loader, visualize=False,
                 save_dir='data/visualization'):
    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval(), i2t_metric=False, visualize=visualize, save_dir=save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml', help="Path to the config file")
    parser.add_argument("--visualize", action='store_true', help="Enable similarity visualization")
    parser.add_argument("--save_dir", default='/home/kb/TBPR/IRRA-main/data/visualization',
                        help="Directory to save heatmaps")
    args = parser.parse_args()

    # 加载配置文件
    config = load_train_configs(args.config_file)

    # 合并命令行参数到配置中
    config.visualize = args.visualize
    config.save_dir = args.save_dir

    # 其他参数设置
    config.training = False
    logger = setup_logger('IRRA', save_dir=config.output_dir, if_train=config.training)
    logger.info(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建数据加载器
    test_img_loader, test_txt_loader, num_classes = build_dataloader(config)

    # 构建模型
    model = MaBa_model(config, num_classes=num_classes)

    # 加载模型权重
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(config.output_dir, 'best.pth'))

    # 将模型移动到设备
    model.to(device)

    # 进行推理
    do_inference(model, test_img_loader, test_txt_loader, visualize=config.visualize, save_dir=config.save_dir)
