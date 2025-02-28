import torch
import torch.nn.functional as F
from torch import nn
import nltk
from nltk.corpus import wordnet
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from utils.text_aug import replace_strategy, modify_strategy, mask_strategy
from utils.loss import intra_modal_separation_loss, intra_modal_contrastive_loss,NCITCLoss
from utils.cluster import dynamic_clustering

class IntraModalReasoning(nn.Module):
    def __init__(self, text_encoder, vision_encoder, vocab, pos_tags,
                 gamma=0.7, k=5, lambda_=0.3, alpha=0.5):
        super().__init__()
        # 共享参数
        self.text_encoder = text_encoder  # 输入: (B, L) 输出: (B, D)
        self.vision_encoder = vision_encoder  # 输入: (B, C, H, W) 输出: (B, D)
        self.alpha = alpha

        # 文本增强
        self.vocab = vocab
        self.pos_tags = pos_tags  # 预定义POS分类词典
        self.gamma = gamma  # 相似度阈值
        self.k = k  # 近邻数
        self.lambda_ = lambda_

        # TF-IDF初始化 (B, L) -> (B, V)
        self.tfidf = TfidfTransformer()
        self.idf_matrix = torch.ones(len(vocab))
        self.replace_strategy = replace_strategy
        self.modify_strategy = modify_strategy
        self.mask_strategy = mask_strategy
        self.ims_loss = intra_modal_separation_loss
        self.imc_loss = intra_modal_contrastive_loss
        self.NITC_loss = NCITCLoss

        # POS标签扩展字典
        self.pos_dict = {
            'NN': 'noun', 'NNS': 'noun', 'VB': 'verb', 'VBD': 'verb',
            'VBG': 'verb', 'VBN': 'verb', 'VBP': 'verb', 'VBZ': 'verb',
            'JJ': 'adjective', 'JJR': 'adjective', 'JJS': 'adjective'
        }


    def select_negatives(self, images, labels):
        with torch.no_grad():
            features = self.vision_encoder(images)

            sim_matrix = F.cosine_similarity(
                features.unsqueeze(1), features.unsqueeze(0), dim=-1
            )

            label_mask = (labels.unsqueeze(1) != labels.unsqueeze(0))  # (B, B)
            masked_sim = sim_matrix * label_mask.float()

            # 选择每个样本的top-N负样本
            _, top_indices = torch.topk(masked_sim, k=self.k, dim=1, largest=False)
            negatives = features[top_indices]  # (B, N, D)

        return negatives

    def forward(self, images, texts, labels):
        # 生成增强后的token序列
        replace_ids = self.replace_strategy(texts, texts)  # (B, L)
        mask_ids = self.mask_strategy(texts)  # (B, L)
        modify_ids = self.modify_strategy(texts, texts)  # (B, L)

        # 编码所有文本变体 (4个变体)
        text_features = []
        for variant_ids in [texts, replace_ids, mask_ids, modify_ids]:
            feat = self.text_encoder(variant_ids)  # (B, D)
            text_features.append(feat)
        text_features = torch.stack(text_features)  # (4, B, D)

        img_features = self.vision_encoder(images)  # (B, D)
        img_negatives = self.select_negatives(images, labels)  # (B, N, D)

        # 图像-文本对齐损失
        loss_ims = self.ims_loss(
            anchors=img_features,
            positives=text_features[0],  # 原始文本特征
            negatives=img_negatives
        )
        loss_imc = 0
        for i in range(1, 4):  # 遍历三个增强变体
            loss_imc += self.imc_loss(
                anchors=text_features[0],
                positives=text_features[i],
                negatives=text_features[[j for j in range(4) if j != 0 and j != i]]
            )
        loss_imc /= 3

        total_loss = loss_ims + 0.5 * loss_imc

        return {
            'total_loss': total_loss,
            'ims_loss': loss_ims,
            'imc_loss': loss_imc,
            'text_variants': text_features.permute(1, 0, 2),  # (B,4,D)
            'image_negatives': img_negatives
        }


class CrossModalRefinement(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.tau = nn.Parameter(torch.log(torch.tensor(0.07).exp()))  # exp约束

        self.vis_gate = nn.Sequential(
            nn.Linear(3 * dim, dim),  # 增强上下文信息
            nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Sigmoid()
        )

        self.vis_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.text_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)

    def forward(self, vis_features, text_features):
        """
        输入维度:
        vis_features: (B, N, D)  # [CLS] + N patches
        text_features: (B, M, D)  # [SOS] + ... + [EOT]

        输出维度:
        refined_vis: (B, N, D)
        refined_text: (B, M, D)
        """
        B, N, D = vis_features.shape
        M = text_features.shape[1]

        vis_refined, _ = self.vis_attn(
            query=vis_features,
            key=text_features,
            value=text_features
        )  # (B, N, D)

        text_refined, _ = self.text_attn(
            query=text_features,
            key=vis_features,
            value=vis_features
        )  # (B, M, D)

        vis_ctx = torch.cat([
            vis_features,
            vis_refined,
            text_refined.mean(dim=1, keepdim=True).expand(-1, N, -1)  # 文本全局上下文
        ], dim=-1)  # (B, N, 3D)
        g_v = self.vis_gate(vis_ctx)  # (B, N, D)
        final_vis = g_v * vis_features + (1 - g_v) * vis_refined

        text_ctx = torch.cat([
            text_features,
            text_refined,
            vis_refined.mean(dim=1, keepdim=True).expand(-1, M, -1)  # 视觉全局上下文
        ], dim=-1)  # (B, M, 3D)
        g_t = self.text_gate(text_ctx)  # (B, M, D)
        final_text = g_t * text_features + (1 - g_t) * text_refined

        return final_vis, final_text

class EnhancedCLIPWithRefinement(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.base = clip_model
        self.cmr = CrossModalRefinement(dim=clip_model.text_projection.shape[1])

        # 复用CLIP原有投影层
        self.vis_proj = nn.Identity()
        self.text_proj = nn.Identity()

    def _get_eot_position(self, text_tokens):
        """ 动态检测EOT位置 """
        eot_id = self.base.token_embedding.weight.shape[0] - 1  # 假设EOT是最后一个token
        return (text_tokens == eot_id).int().argmax(dim=-1)

    def forward(self, images, texts):
        vis_features = self.base.visual(images)  # (B, N, D)
        text_features = self.base.encode_text(texts)  # (B, M, D)

        refined_vis, refined_text = self.cmr(vis_features, text_features)

        vis_global = refined_vis[:, 0, :]  # CLS token
        eot_pos = self._get_eot_position(texts)
        text_global = refined_text[torch.arange(texts.size(0)), eot_pos]

        vis_global = self.vis_proj(vis_global) @ self.base.visual.proj.T
        text_global = self.text_proj(text_global) @ self.base.text_projection.T

        loss = self.NCITCLoss(vis_global, text_global)

        return {
            'image_features': vis_global,
            'text_features': text_global,
            'loss': loss
        }


class DiscriminativeClueCorrection(nn.Module):
    def __init__(self, dim=512, num_clusters=3, beta=5, num_heads=8):
        super().__init__()
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.tau_p = nn.Parameter(torch.tensor(0.1).log())  # exp约束
        self.tau_n = nn.Parameter(torch.tensor(0.05).log())
        self.memory_size = 1024
        self.register_buffer('text_memory', torch.randn(self.memory_size, dim))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * dim, 1),
            nn.Sigmoid()
        )



    def forward(self, vis_global, text_features, labels=None):
        B, M, D = text_features.shape

        similarities = F.cosine_similarity(
            vis_global.unsqueeze(1), text_features, dim=-1
        )  # (B, M)

        cluster_means, cluster_stds = dynamic_clustering(similarities)  # (B,K)

        cluster_scores = cluster_means / (cluster_stds + 1e-6)  # (B,K)
        # 排除最高和最低得分簇
        sorted_indices = torch.argsort(cluster_scores, dim=1, descending=True)
        valid_mask = ~torch.isin(sorted_indices,
                                 torch.tensor([0, self.num_clusters - 1], device=cluster_scores.device))
        selected = sorted_indices[valid_mask].view(B, -1)[:, 0]  # 取剩余最高分簇

        corrected, _ = self.cross_attn(
            query=vis_global.unsqueeze(1),
            key=text_features,
            value=text_features
        )  # (B,1,D) → (B,D)
        corrected = corrected.squeeze(1)

        tau_p = self.tau_p.exp()  # 确保>0
        tau_n = self.tau_n.exp()

        pos_sim = F.cosine_similarity(vis_global, corrected, dim=-1)  # (B,)

        with torch.no_grad():
            self.text_memory[self.ptr: self.ptr + B] = corrected.detach()
            self.ptr = (self.ptr + B) % self.memory_size

            all_negs = torch.cat([
                corrected.roll(shifts=1, dims=0),  # batch内负样本
                self.text_memory
            ], dim=0)  # (B + memory_size, D)

        neg_sims = F.cosine_similarity(
            vis_global.unsqueeze(1),  # (B,1,D)
            all_negs.unsqueeze(0),  # (1,N,D)
            dim=-1
        )  # (B, N)

        neg_mask = torch.ones_like(neg_sims, dtype=torch.bool)
        if labels is not None:
            label_mask = (labels.unsqueeze(1) != labels.view(-1, 1))
            neg_mask = label_mask & (
                        torch.arange(neg_sims.size(1), device=device) != torch.arange(B, device=device).unsqueeze(1))

        hard_negs = torch.topk(
            neg_sims.masked_fill(~neg_mask, -1e4),
            k=self.beta,
            dim=1
        ).values  # (B, β)

        # 对比损失计算
        pos_term = torch.exp(pos_sim / tau_p)
        neg_terms = torch.exp(hard_negs / tau_n).sum(dim=1)
        loss = -torch.log(pos_term / (pos_term + neg_terms + 1e-8)).mean()

        return {
            'loss': loss,
            'corrected_features': corrected,
            'cluster_scores': cluster_scores
        }

    def inference(self, vis_global, text_global, corrected):
        # 上下文增强融合
        context = torch.cat([
            vis_global,
            text_global,
            corrected
        ], dim=-1)  # (B,3D)
        gamma = self.fusion_gate(context)  # (B,1)

        # 相似度融合
        base_sim = F.cosine_similarity(vis_global, text_global, dim=-1)
        corrected_sim = F.cosine_similarity(vis_global, corrected, dim=-1)
        fused_sim = gamma.squeeze() * base_sim + (1 - gamma.squeeze()) * corrected_sim

        return fused_sim  # (B,)