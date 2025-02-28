import os.path as op
import random
import time
from datasets import build_dataloader
from utils.logger import setup_logger
from model import Maba_model
from utils.comm import get_rank, synchronize
from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import argparse


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

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
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger("MaBa.eval")  # 修改日志标识

    def _compute_embedding(self, model):
        model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []

        # 文本特征提取
        for pid, caption in self.txt_loader:
            inputs = {
                'images': torch.zeros(1, 3, 224, 224).to(device),  # 虚拟图像输入
                'caption_ids': caption.to(device)
            }
            with torch.no_grad():
                outputs = model(inputs)
            qids.append(pid.view(-1))
            qfeats.append(outputs['text_features'])

        # 图像特征提取
        for pid, img in self.img_loader:
            inputs = {
                'images': img.to(device),
                'caption_ids': torch.zeros(1, 77).long().to(device)  # 虚拟文本输入
            }
            with torch.no_grad():
                outputs = model(inputs)
            gids.append(pid.view(-1))
            gfeats.append(outputs['image_features'])

        return (
            torch.cat(qfeats),
            torch.cat(gfeats),
            torch.cat(qids),
            torch.cat(gids)
        )

    def eval(self, model, i2t_metric=False):
        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1)
        gfeats = F.normalize(gfeats, p=2, dim=1)
        similarity = qfeats @ gfeats.t()

        # 结果表格格式调整
        table = PrettyTable(["Task", "R1", "R5", "R10", "mAP", "mINP"])
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity, qids, gids, get_mAP=True)
        table.add_row(['T2I', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity.t(), gids, qids, get_mAP=True)
            table.add_row(['I2T', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        self.logger.info('\n' + str(table))
        return t2i_cmc[0]


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def get_args():
    parser = argparse.ArgumentParser(description="MaBa Training Args")

    parser.add_argument("--pretrain_choice", default='ViT-B/16', help="CLIP预训练模型选择")
    parser.add_argument("--temperature", type=float, default=0.02)

    parser.add_argument("--name", default="maba_base", help="实验名称")
    parser.add_argument("--output_dir", default="maba_logs")
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--pretrain_choice", default='ViT-B/16')  # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02,
                        help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true',
                        help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+id+mlm',
                        help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")

    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)

    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args
    return parser.parse_args()


def build_model(args, num_classes):
    model = Maba_model(args, num_classes)

    # 冻结CLIP主干参数
    for name, param in model.base_model.named_parameters():
        if 'visual.proj' not in name and 'text_projection' not in name:
            param.requires_grad = False

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M, Trainable: {trainable_params / 1e6:.2f}M")

    return model


def main():
    args = get_args()
    set_seed(1 + get_rank())

    # 初始化分布式训练
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        synchronize()

    # 日志系统初始化
    args.output_dir = op.join(args.output_dir, f"{args.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    logger = setup_logger('MaBa', args.output_dir, distributed_rank=get_rank())  # 修改日志名称
    logger.info(f"Running with {os.environ.get('WORLD_SIZE', 1)} GPUs")

    # 数据加载
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)

    # 模型构建
    model = build_model(args, num_classes).cuda()

    # 优化器配置
    optimizer = torch.optim.AdamW([
        {'params': model.intra_reasoning.parameters(), 'lr': args.lr},
        {'params': model.cross_refinement.parameters(), 'lr': args.lr * 2},
        {'params': model.dcc_correction.parameters(), 'lr': args.lr * 3},
        {'params': model.fusion_gate.parameters()},
        {'params': model.base_model.parameters(), 'lr': args.lr * 0.1},
        {'params': [model.logit_scale]}
    ], weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 分布式训练包装
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )

    # 训练循环
    best_r1 = 0.0
    for epoch in range(1, args.num_epoch + 1):
        model.train()
        total_loss = 0.0

        # 动态损失权重
        intra_weight = max(0.5 * (0.9 ** (epoch // 5)), 0.1)
        dcc_weight = min(0.3 * (1.2 ** (epoch // 5)), 0.6)

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # 数据准备
            inputs = {
                'images': batch['images'].cuda(),
                'caption_ids': batch['caption_ids'].cuda(),
                'pids': batch['pids'].cuda()
            }

            # 前向计算
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(inputs)
                loss = (
                        outputs['losses']['intra_loss'] * intra_weight +
                        outputs['losses']['dcc_loss'] * dcc_weight +
                        outputs['losses'].get('itc_loss', 0) * 0.1 +
                        outputs['losses'].get('id_loss', 0) * 0.1
                )

            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # 日志记录
            if batch_idx % args.log_period == 0:
                logger.info(
                    f"Epoch[{epoch}/{args.num_epoch}] Iter[{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        # 验证和保存
        if epoch % args.eval_period == 0:
            r1 = Evaluator(val_img_loader, val_txt_loader).eval(model)
            if r1 > best_r1:
                best_r1 = r1
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, op.join(args.output_dir, f"best_model.pth"))
                logger.info(f"New best model saved at epoch {epoch} with R1: {r1:.2f}")

            logger.info(f"Epoch {epoch} Evaluation:")
            logger.info(f"Current R1: {r1:.2f} | Best R1: {best_r1:.2f}")

        scheduler.step()


if __name__ == '__main__':
    main()