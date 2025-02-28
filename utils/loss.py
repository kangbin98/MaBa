import torch
import torch.nn.functional as F
from torch import nn



def intra_modal_contrastive_loss(self, anchors, positives, negatives):
    pos_sim = F.cosine_similarity(anchors, positives, dim=-1)  # (B,)
    neg_sim = F.cosine_similarity(
    anchors.unsqueeze(1), negatives, dim=-1
            ).mean(dim=1)  # (B,)
            
    return -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))).mean()    

def intra_modal_separation_loss(self, anchors, positives, negatives):
    pos_sim = F.cosine_similarity(anchors, positives, dim=-1)  # (B,)
    neg_sim = F.cosine_similarity(
        anchors.unsqueeze(1), negatives, dim=-1
    )  # (B, N)

    # 最难负样本
    hardest_neg = neg_sim.max(dim=1)[0]  # (B,)
    losses = F.relu(self.alpha + hardest_neg - pos_sim)
    return losses.mean()


class NCITCLoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/tau)))  
        
    def forward(self, vis_global, text_global, labels=None):
        logit_scale = self.logit_scale.exp()
        
        # 标准化特征
        vis_global = F.normalize(vis_global, p=2, dim=-1)
        text_global = F.normalize(text_global, p=2, dim=-1)
        
        # 计算相似度
        sim_v2t = logit_scale * vis_global @ text_global.T  # (B, B)
        sim_t2v = sim_v2t.T
        
        # 构建目标矩阵
        if labels is not None:
            # 有监督模式
            targets = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        else:
            # 无监督模式 (对角线为正样本)
            targets = torch.eye(sim_v2t.size(0), device=sim_v2t.device)
        
        # 对称交叉熵损失
        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * targets, dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1) * targets, dim=1).mean()
        
        return (loss_v2t + loss_t2v) / 2


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss