# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch
import time
from torchvision import transforms
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from losses import DistillationLoss
import utils
import wandb
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
import sys


def train_one_epoch(
    model: torch.nn.Module,
    criterion: DistillationLoss,
    teacher_model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    lr_scheduler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    args=None,
):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    no_mixup_drw_flag = True
    accum_iter = args.accum_iter

    for data_iter_step, (samples_student, targets) in enumerate(
        metric_logger.log_every(
            iterable=data_loader, print_freq=print_freq, header=header
        )
    ):

        if args.multi_crop:
            samples_student_global = torch.cat(samples_student[:2], dim=0)
            samples_student_local = torch.cat(samples_student[2:], dim=0)
            targets = torch.cat([targets] * len(samples_student), dim=0)
            # samples_student = [im.cuda(non_blocking=True) for im in samples_student]
            #     # samples_student = torch.cat(samples_student, dim=0)
            samples_student_global = samples_student_global.to(device)
            samples_student_local = samples_student_local.to(device)
        # else:
        # samples_student = [im.cuda(non_blocking=True) for im in samples_student]
        # samples_student = samples_student.to(device, non_blocking=True)
        # outputs = model(samples_student)
        # exit()
            
        targets = targets.to(device, non_blocking=True)
        drw = args.epochs + 1 if args.drw is None else args.drw
        # when not doing drw, if we want to do mixup, to keep the epoch<drw condition true inside 'elif' we set drw = 1200+1


        # Code modified for DeiT-LT
        if accum_iter > 1 and (
            data_iter_step % accum_iter == 0 or data_iter_step == (len(data_loader) - 1)
        ):
            lr_scheduler.step(epoch + data_iter_step / len(data_loader))
        if mixup_fn is not None and epoch < drw:  # do mixup before starting drw only
            if args.student_transform == 0:
                # --> Mixup for local and global crops
                if args.multi_crop:
                    samples_student_global, targets_student_global = mixup_fn(
                        samples_student_global, targets[: 2 * args.batch_size]
                    )  # mixing student and teacher both
                    samples_student_local, targets_student_local = mixup_fn(
                        samples_student_local, targets[2 * args.batch_size :]
                    )  # mixing student and teacher both
                    targets_student = torch.cat(
                        [targets_student_global, targets_student_local], dim=0
                    )

                # --> Normal mixup and cutmix
                else:
                    samples_student = samples_student.to(device)
                    targets = targets.to(device)
                    samples_student, targets_student = mixup_fn(
                        samples_student, targets
                    )  # mixing student and teacher both
            else:
                targets_student = targets.to(device)

        else:  # in drw stage
            if not args.no_mixup_drw:
                if args.student_transform == 0 and mixup_fn is not None:
                    # --> Mixup for local and global crops
                    if args.multi_crop:
                        samples_student_global, targets_student_global = mixup_fn(
                            samples_student_global, targets[: 2 * args.batch_size]
                        )  # mixing student and teacher both
                        samples_student_local, targets_student_local = mixup_fn(
                            samples_student_local, targets[2 * args.batch_size: ]
                        )  # mixing student and teacher both
                        targets_student = torch.cat(
                            [targets_student_global, targets_student_local], dim=0
                        )

                    # --> Normal mixup and cutmix
                    else:
                        samples_student, targets_student = mixup_fn(
                            samples_student, targets
                        )  # mixing student and teacher both
                #     samples_student, targets_student = mixup_fn(
                #         samples_student, targets
                #     )
                # else:
                #     targets_student = targets

            else:
                if no_mixup_drw_flag:
                    print("In no mixup drw phase")
                    no_mixup_drw_flag = False
                samples_student = samples_student.to(device)
                targets_student = targets.to(device)

        if args.bce_loss:
            if epoch >= drw:
                targets_student = torch.nn.functional.one_hot(
                    targets_student.to(torch.int64), num_classes=args.nb_classes
                ).cuda()
            else:
                targets_student = targets_student.gt(0.0).type(targets_student.dtype)

        # --> setting teacher samples (pass only global crop)
        if args.input_size != args.teacher_size and not args.no_distillation:
            if args.multi_crop:
                samples_teacher = transforms.Compose(
                    [transforms.Resize(args.teacher_size, interpolation=3)]
                )(samples_student_global)
            else:
                samples_teacher = transforms.Compose(
                    [transforms.Resize(args.teacher_size, interpolation=3)]
                )(samples_student)
        # --> Normal execution
        else:
            if args.multi_crop:
                samples_teacher = [samples_student_global, samples_student_local]
            else:
                samples_teacher = samples_student

        with torch.cuda.amp.autocast():
            # --> Multi-crop forward pass
            if args.multi_crop and not args.no_distillation:
                out_student_local = model(samples_student_local)
                out_student_global = model(samples_student_global)
                x_local, x_dist_local, sim_12_local, adl_local = out_student_local
                x_global, x_dist_global, sim_12_global, adl_global = out_student_global

                sim_12 = torch.cat([sim_12_global, sim_12_local], dim=0)
                x = torch.cat([x_global, x_local], dim=0)
                x_dist = torch.cat([x_dist_global, x_dist_local], dim=0)
                if args.adl:
                    adl = torch.cat([adl_global, adl_local], dim=0)
                else:
                    adl = adl_local
                outputs_student = (x, x_dist, sim_12, adl)

            # --> No distillation forward pass with multi-crop
            elif args.multi_crop and args.no_distillation:
                out_student_local = model(samples_student_local)
                out_student_global = model(samples_student_global)
                outputs_student = torch.cat(
                    [out_student_global, out_student_local], dim=0
                )

            # Normal forward pass
            else:
                outputs_student = model(samples_student)
                _, _, sim_12, adl = outputs_student

            if not args.no_distillation:
                sim_12 = torch.mean(sim_12)
                loss, cls_loss, dst_loss = criterion(
                    samples_teacher, outputs_student, targets_student
                )

                # * ADL + base loss
                if args.adl:
                    loss += adl

                loss_value = loss.item()
                cls_loss_value = cls_loss.item()
                dst_loss_value = dst_loss.item()
                sim_12_value = sim_12.item()
            else:
                loss = criterion(outputs_student, targets_student)
                loss_value = loss.item()
                cls_loss_value = 0
                dst_loss_value = 0
                sim_12_value = 0

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if accum_iter > 1:
            loss /= accum_iter

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )

        # Code modified for DeiT-LT
        if accum_iter > 1:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=(data_iter_step + 1) % accum_iter == 0
                or data_iter_step == (len(data_loader) - 1),
                create_graph=is_second_order,
            )
        else:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )

        if accum_iter == 1:
            optimizer.zero_grad()
        elif (data_iter_step + 1) % accum_iter == 0 or data_iter_step == (
            len(data_loader) - 1
        ):
            optimizer.zero_grad()

        torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(sim_12=sim_12_value)
        metric_logger.update(cls_loss=cls_loss_value)
        metric_logger.update(dst_loss=dst_loss_value)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return train_stats


def accuracy(output, target, args, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        # print(pred.size())

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    all_preds_cls = []
    all_preds_dist = []
    all_preds_avg = []
    all_targets = []

    for obj in metric_logger.log_every(
        iterable=data_loader, print_freq=100, header=header
    ):
        samples_student = obj[0].to(device, non_blocking=True)
        targets_student = obj[1].to(device, non_blocking=True)

        batch_size = targets_student.shape[0]
        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(samples_student)

            # ! [CHANGE] Split the outputs even for no distillation
            if args.no_distillation:
                output_cls = outputs
                output_dist = outputs
            else:
                output_cls, output_dist = outputs
                # output_cls = outputs
                # output_dist = outputs
        # print(samples_student[0])
        # print(targets_student)
        # sys.exit(1)

        # targets_student = torch.Tensor(np.array(args.reverse_class_map)[targets_student.detach().cpu()]).to(device, non_blocking = True).to(torch.int)
        output_avg = (output_cls + output_dist) / 2
        acc1_cls, acc5_cls = accuracy(output_cls, targets_student, args, topk=(1, 5))
        acc1_dist, acc5_dist = accuracy(output_dist, targets_student, args, topk=(1, 5))
        acc1_avg, acc5_avg = accuracy(output_avg, targets_student, args, topk=(1, 5))

        _, pred_cls = torch.max(output_cls, 1)  # (256, N)
        all_preds_cls.extend(pred_cls.cpu().numpy())

        _, pred_dist = torch.max(output_dist, 1)
        all_preds_dist.extend(pred_dist.cpu().numpy())

        _, pred_avg = torch.max(output_avg, 1)
        all_preds_avg.extend(pred_avg.cpu().numpy())

        all_targets.extend(targets_student.cpu().numpy())

        batch_size = samples_student.shape[0]

        # import pdb;pdb.set_trace()

        metric_logger.meters["acc1_cls"].update(acc1_cls.item(), n=batch_size)
        metric_logger.meters["acc5_cls"].update(acc5_cls.item(), n=batch_size)

        metric_logger.meters["acc1_dist"].update(acc1_dist.item(), n=batch_size)
        metric_logger.meters["acc5_dist"].update(acc5_dist.item(), n=batch_size)

        metric_logger.meters["acc1_avg"].update(acc1_avg.item(), n=batch_size)
        metric_logger.meters["acc5_avg"].update(acc5_avg.item(), n=batch_size)

    # gather the stats from all processes
        
    # Code modified for DeiT-LT
    cf_avg = confusion_matrix(
        all_targets, all_preds_avg, labels=range(args.nb_classes)
    ).astype(float)
    cf_cls = confusion_matrix(
        all_targets, all_preds_cls, labels=range(args.nb_classes)
    ).astype(float)
    cf_dist = confusion_matrix(
        all_targets, all_preds_dist, labels=range(args.nb_classes)
    ).astype(float)

    cls_count_avg = cf_avg.sum(axis=1)
    cls_count_cls = cf_cls.sum(axis=1)
    cls_count_dist = cf_dist.sum(axis=1)

    cls_hit_avg = np.diag(cf_avg)
    cls_hit_cls = np.diag(cf_cls)
    cls_hit_dist = np.diag(cf_dist)

    cls_acc_avg = cls_hit_avg * 100.0 / cls_count_avg
    cls_acc_cls = cls_hit_cls * 100.0 / cls_count_cls
    cls_acc_dist = cls_hit_dist * 100.0 / cls_count_dist

    head_acc_avg = np.mean(cls_acc_avg[: args.categories[0]])
    med_acc_avg = np.mean(cls_acc_avg[args.categories[0] : args.categories[1]])
    tail_acc_avg = np.mean(cls_acc_avg[args.categories[1] :])

    head_acc_cls = np.mean(cls_acc_cls[: args.categories[0]])
    med_acc_cls = np.mean(cls_acc_cls[args.categories[0] : args.categories[1]])
    tail_acc_cls = np.mean(cls_acc_cls[args.categories[1] :])

    head_acc_dist = np.mean(cls_acc_dist[: args.categories[0]])
    med_acc_dist = np.mean(cls_acc_dist[args.categories[0] : args.categories[1]])
    tail_acc_dist = np.mean(cls_acc_dist[args.categories[1] :])

    metric_logger.meters["head_acc_avg"].update(head_acc_avg, n=1)
    metric_logger.meters["med_acc_avg"].update(med_acc_avg, n=1)
    metric_logger.meters["tail_acc_avg"].update(tail_acc_avg, n=1)

    metric_logger.meters["head_acc_cls"].update(head_acc_cls, n=1)
    metric_logger.meters["med_acc_cls"].update(med_acc_cls, n=1)
    metric_logger.meters["tail_acc_cls"].update(tail_acc_cls, n=1)

    metric_logger.meters["head_acc_dist"].update(head_acc_dist, n=1)
    metric_logger.meters["med_acc_dist"].update(med_acc_dist, n=1)
    metric_logger.meters["tail_acc_dist"].update(tail_acc_dist, n=1)

    metric_logger.synchronize_between_processes()

    print("\nCURRENT NUMBERS ----->")
    print("Overall / Head / Med / Tail")
    print(
        "AVERAGE: ",
        round(metric_logger.acc1_avg.global_avg, 3),
        " / ",
        round(metric_logger.head_acc_avg.global_avg, 3),
        " / ",
        round(metric_logger.med_acc_avg.global_avg, 3),
        " / ",
        round(metric_logger.tail_acc_avg.global_avg, 3),
    )
    print(
        "CLS    : ",
        round(metric_logger.acc1_cls.global_avg, 3),
        " / ",
        round(metric_logger.head_acc_cls.global_avg, 3),
        " / ",
        round(metric_logger.med_acc_cls.global_avg, 3),
        " / ",
        round(metric_logger.tail_acc_cls.global_avg, 3),
    )
    print(
        "DIST   : ",
        round(metric_logger.acc1_dist.global_avg, 3),
        " / ",
        round(metric_logger.head_acc_dist.global_avg, 3),
        " / ",
        round(metric_logger.med_acc_dist.global_avg, 3),
        " / ",
        round(metric_logger.tail_acc_dist.global_avg, 3),
    )
    print("\n\n")

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return test_stats