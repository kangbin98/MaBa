import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def trainer(start_epoch, args, model, train_loader, evaluator, optimizer,
            scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    # 修改日志标识为MaBa
    logger = logging.getLogger("MaBa.train")
    logger.info('Start MaBa Training')

    # 调整监控指标
    meters = {
        "total_loss": AverageMeter(),
        "intra_loss": AverageMeter(),
        "dcc_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "logit_scale": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0

    # 训练循环
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        # 动态损失权重
        intra_weight = max(0.5 * (0.9  ** (epoch // 5)), 0.1)
        dcc_weight = min(0.3 * (1.2  ** (epoch // 5)), 0.6)

        for n_iter, batch in enumerate(train_loader):
            # 数据格式转换
            inputs = {
                'images': batch['images'].to(device),
                'caption_ids': batch['caption_ids'].to(device),
                'pids': batch['pids'].to(device)
            }

            # 前向计算
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(inputs)

                # 加权损失计算
                total_loss = (
                        outputs['losses']['intra_loss'] * intra_weight +
                        outputs['losses']['dcc_loss'] * dcc_weight +
                        outputs['losses'].get('itc_loss', 0) * 0.1 +
                        outputs['losses'].get('id_loss', 0) * 0.1
                )

            # 记录指标
            batch_size = inputs['images'].size(0)
            meters['total_loss'].update(total_loss.item(), batch_size)
            meters['intra_loss'].update(outputs['losses']['intra_loss'].item(), batch_size)
            meters['dcc_loss'].update(outputs['losses']['dcc_loss'].item(), batch_size)
            meters['itc_loss'].update(outputs['losses'].get('itc_loss', 0), batch_size)
            meters['id_loss'].update(outputs['losses'].get('id_loss', 0), batch_size)
            meters['logit_scale'].update(outputs['logit_scale'].item(), 1)

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            synchronize()

            # 日志记录
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iter[{n_iter + 1}/{len(train_loader)}]"
                info_str += " | "
                info_str += f"Loss: {meters['total_loss'].avg:.3f}"
                info_str += f" (Intra: {meters['intra_loss'].avg:.3f}"
                info_str += f" DCC: {meters['dcc_loss'].avg:.3f}"
                info_str += f" ITC: {meters['itc_loss'].avg:.3f}"
                info_str += f" ID: {meters['id_loss'].avg:.3f})"
                info_str += f" | Scale: {meters['logit_scale'].avg:.2f}"
                info_str += f" | LR: {scheduler.get_lr()[0]:.1e}"
                logger.info(info_str)

        # TensorBoard记录
        tb_writer.add_scalar('LR', scheduler.get_lr()[0], epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        # 学习率更新
        scheduler.step()

        # 耗时统计
        if get_rank() == 0:
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch} Completed | Time: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")
            logger.info(f"Speed: {len(train_loader) / epoch_time:.1f} batches/s")

        # 模型验证
        if epoch % eval_period == 0 and get_rank() == 0:
            logger.info(f"Validation Start @ Epoch {epoch}")
            eval_model = model.module if args.distributed else model
            top1 = evaluator.eval(eval_model)

            # 保存最佳模型
            if top1 > best_top1:
                best_top1 = top1
                arguments["epoch"] = epoch
                checkpointer.save("best_model", ** arguments)
                logger.info(f"New Best R1: {best_top1:.2f}%")

            # 释放显存
            torch.cuda.empty_cache()

    # 训练结束
    if get_rank() == 0:
        logger.info(f"Training Completed | Best R1: {best_top1:.2f}%")
        tb_writer.close()


def inference(model, test_img_loader, test_txt_loader):
    # 修改测试日志标识
    logger = logging.getLogger("MaBa.test")
    logger.info("Start MaBa Inference")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
    logger.info(f"Final Test R1: {top1:.2f}%")
    return top1