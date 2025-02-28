from utils.loss import compute_id,compute_itc
from .baseline import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from model import CrossModalRefinement,IntraModalReasoning,DiscriminativeClueCorrection


class MaBa(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        # 初始化CLIP基础模型
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg['embed_dim']

        self.intra_reasoning = IntraModalReasoning(
            text_encoder=self.base_model.text_encoder,
            vision_encoder=self.base_model.visual,
            vocab=load_vocab(),  # 加载词汇表
            pos_tags=load_pos_tags()
        )

        self.cross_refinement = CrossModalRefinement(dim=self.embed_dim)
        self.dcc_correction = DiscriminativeClueCorrection(dim=self.embed_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(2*self.embed_dim, 1),
            nn.Sigmoid()
        )

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.classifier.weight, std=0.01)
            nn.init.zeros_(self.classifier.bias)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def _set_task(self):
        loss_names = self.args.loss_names.split('+')
        self.current_task = [l.strip() for l in loss_names]

    def _get_visual_features(self, images):
        """ 统一视觉特征提取流程 """
        # (B, 1+N, D) → [CLS] + patch tokens
        all_vis = self.base_model.visual(images, return_all=True)  
        return {
            'global': all_vis[:, 0],    # (B,D)
            'local': all_vis[:, 1:]     # (B,N,D)
        }

    def _get_text_features(self, text_ids):
        """ 统一文本特征提取流程 """
        # (B, 1+M) → [SOS] + tokens + [EOT]
        all_text = self.base_model.text_encoder(text_ids, return_all=True)  
        return {
            'global': all_text[:, -1],  # (B,D) 取EOT
            'local': all_text[:, :-1]   # (B,M,D)
        }

    def forward(self, batch):
        images = batch['images']
        text_ids = batch['caption_ids']

        with torch.no_grad():
            vis_base = self._get_visual_features(images)
            txt_base = self._get_text_features(text_ids)

        intra_outputs = self.intra_reasoning(
            images=images,
            texts=text_ids,
            labels=batch.get('pids', None)
        )

        refined_vis, refined_txt = self.cross_refinement(
            vis_features=torch.cat([
                vis_base['global'].unsqueeze(1),  # (B,1,D)
                intra_outputs['image_variants']   # (B,4,D)
            ], dim=1),  # (B,5,D)
            text_features=torch.cat([
                intra_outputs['text_variants'],   # (B,4,D)
                txt_base['global'].unsqueeze(1)   # (B,1,D)
            ], dim=1)  # (B,5,D)
        )
        
        # 鉴别线索修正
        dcc_outputs = self.dcc_correction(
            vis_global=refined_vis[:, 0],  # (B,D)
            text_features=refined_txt      # (B,5,D)
        )
        
        # 动态特征融合
        gamma = self.fusion_gate(
            torch.cat([refined_vis[:,0], dcc_outputs['corrected']], dim=1)
        )  # (B,1)
        final_vis = gamma * refined_vis[:,0] + (1-gamma) * dcc_outputs['corrected']
        final_txt = refined_txt[:, -1]  # 最终文本特征取EOT

        #损失计算
        losses = {}
        logit_scale = self.logit_scale.exp()

        if 'itc' in self.current_task:
            losses['itc_loss'] = compute_itc(
                final_vis, final_txt, logit_scale
            )

        if 'id' in self.current_task:
            img_logits = self.classifier(final_vis)
            txt_logits = self.classifier(final_txt)
            losses['id_loss'] = compute_id(
                img_logits, txt_logits, batch['pids']
            )
        
        # 模块特定损失
        losses.update({
            'intra_loss': intra_outputs['total_loss'],
            'dcc_loss': dcc_outputs['loss']
        })
        
        return {
            'losses': losses,
            'logit_scale': logit_scale,
            'image_features': final_vis,
            'text_features': final_txt
        }

def Maba_model(args, num_classes):
    model = MaBa(args, num_classes)
    convert_weights(model)  # 转换混合精度训练参数
    return model