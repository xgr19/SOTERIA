# _*_ coding: utf-8 _*_
import copy
from soteria.utils import *
from soteria.metric import *
import soteria.dynn as dynn
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
import json
import random



def validate(
    run_manager,
    val_settings,
    epoch=0,
):
    dynamic_net = run_manager.net
    dynamic_net.eval()
    losses_of_subnets, top1_of_subnets, top5_of_subnets, Obj_of_subnets = [], [], [], []
    valid_log = ""
    object_type = run_manager.object_type
    for setting in val_settings:
        dynamic_net.set_active(setting)
        loss, (top1, top5) = run_manager.validate(
            epoch=epoch, net=dynamic_net
        )
        losses_of_subnets.append(loss)
        top1_of_subnets.append(top1)
        top5_of_subnets.append(top5)
        static_net = copy.deepcopy(dynamic_net)
        static_net.get_active_module()
        obj = run_manager.profile(static_net)
        Obj_of_subnets.append(obj)
        run_manager.write_log(
            "-" * 30 + " Validate %s:%.2f" % (object_type, obj) + "-" * 30, "train", should_print=False
        )
        run_manager.write_log(json.dumps(setting), "train", should_print=False)
        valid_log += "%s:%.2f (%.3f), " % (object_type, obj, top1)
    run_manager.callback_val(top1_of_subnets, Obj_of_subnets, epoch)

    return (
        list_mean(losses_of_subnets),
        list_mean(top1_of_subnets),
        list_mean(top5_of_subnets),
        multi_acc(top1_of_subnets, Obj_of_subnets),
        valid_log,
    )

def train_one_epoch_like_alphanet(run_manager, epoch, train_settings, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.net
    cfg = run_manager.cfg
    soft_criterion = dynn.AdaptiveLossSoft(-1.0, 1.0, 5.0)

    # switch to train mode
    dynamic_net.train()

    train_loader = run_manager.train_dataloader

    nBatch = run_manager.nBatch

    data_time = AverageMeter()
    losses = AverageMeter()
    metric_dict = run_manager.get_metric_dict()
    num_sample = run_manager.num_sample_train_per_epoch
    Acc = []
    with tqdm(
            total=nBatch,
            desc="Train Epoch #{}".format(epoch + 1),
            disable=False,
    ) as t:
        end = time.time()
        for i, (x, y_true) in enumerate(train_loader):
            data_time.update(time.time() - end)
            if epoch < warmup_epochs:
                new_lr = run_manager.warmup_adjust_learning_rate(
                    run_manager.optimizer,
                    warmup_epochs * nBatch,
                    nBatch,
                    epoch,
                    i,
                    warmup_lr,
                )
            else:
                new_lr = run_manager.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )
                # wandb.log(dict(lr=new_lr), step=epoch)

            x, y_true = x.cuda(), y_true.cuda()
            target = y_true
            # clean gradients
            dynamic_net.zero_grad()

            # step 1 sample the largest network
            dynamic_net.set_active(run_manager.max_net_setting)
            output = dynamic_net(x)
            loss = run_manager.train_criterion(output, target)
            loss.backward()

            with torch.no_grad():
                soft_logits = output.clone().detach()

            loss_of_subnets = []
            # compute output
            for subnet_setting in train_settings:
                dynamic_net.set_active(subnet_setting)
                output = dynamic_net(x)
                loss = soft_criterion(output, soft_logits)

                loss_type = "alphanet-loss:%f" % (loss.item())

                # measure accuracy and record loss
                loss_of_subnets.append(loss)
                run_manager.update_metric(metric_dict, output, target)
                loss.backward()
                Acc.append(run_manager.Accuracy(output, target)[0].cpu().numpy())

            run_manager.optimizer.step()
            losses.update(list_mean(loss_of_subnets), x.size(0))
            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "lr": new_lr,
                    "loss_type": loss_type,
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()
    run_manager.callback_train(Acc, train_settings, epoch=epoch)
    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)


def train_one_epoch(run_manager, epoch, train_settings, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.net
    cfg = run_manager.cfg

    # switch to train mode
    dynamic_net.train()

    train_loader = run_manager.train_dataloader

    nBatch = run_manager.nBatch

    data_time = AverageMeter()
    losses = AverageMeter()
    metric_dict = run_manager.get_metric_dict()
    num_sample = run_manager.num_sample_train_per_epoch
    Acc = []
    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=False,
    ) as t:
        end = time.time()
        for i, (x, y_true) in enumerate(train_loader):
            data_time.update(time.time() - end)
            if epoch < warmup_epochs:
                new_lr = run_manager.warmup_adjust_learning_rate(
                    run_manager.optimizer,
                    warmup_epochs * nBatch,
                    nBatch,
                    epoch,
                    i,
                    warmup_lr,
                )
            else:
                new_lr = run_manager.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )

            x, y_true = x.cuda(), y_true.cuda()
            target = y_true

            # soft target
            if cfg.distilation.kd_ratio > 0:
                cfg.teacher_model.train()
                with torch.no_grad():
                    soft_logits = cfg.teacher_model(x).detach()
                    soft_label = F.softmax(soft_logits, dim=1)

            # clean gradients
            dynamic_net.zero_grad()

            loss_of_subnets = []
            # compute output
            for _ in range(cfg.runtime.dynamic_batch_size):
                # set random seed before sampling
                idx = (i * cfg.runtime.dynamic_batch_size + _) % num_sample
                subnet_setting = train_settings[idx]
                dynamic_net.set_active(subnet_setting)
                output = dynamic_net(x)
                if cfg.distilation.kd_ratio == 0:
                    loss = run_manager.train_criterion(output, y_true)
                    loss_type = "ce"
                else:
                    if cfg.distilation.kd_type == "ce":
                        kd_loss = dynn.cross_entropy_loss_with_soft_target(
                            output, soft_label
                        )
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = cfg.distilation.kd_ratio * kd_loss + run_manager.train_criterion(
                        output, y_true
                    )
                    loss_type = "%.1fkd-%s & ce" % (cfg.distilation.kd_ratio, cfg.distilation.kd_type)

                # measure accuracy and record loss
                loss_of_subnets.append(loss)
                run_manager.update_metric(metric_dict, output, target)
                loss.backward()
                Acc.append(run_manager.Accuracy(output, target)[0].cpu().numpy())

            run_manager.optimizer.step()
            losses.update(list_mean(loss_of_subnets), x.size(0))
            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "lr": new_lr,
                    "loss_type": loss_type,
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()
    run_manager.callback_train(Acc, train_settings, epoch=epoch)
    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)


def train(run_manager, validate_func=None):
    if validate_func is None:
        validate_func = validate

    cfg = run_manager.cfg
    no_improveed_epochs = 0
    max_no_improved_epochs = run_manager.cfg.get("max_no_improved_epochs", 5)
    for epoch in range(
        run_manager.start_epoch, run_manager.n_epochs + cfg.runtime.warmup_epochs
    ):
        random.seed(epoch)
        train_settings, val_settings = run_manager.get_settings(epoch=epoch)
        train_loss, (train_top1, train_top5) = train_one_epoch(
            run_manager, epoch, train_settings, cfg.runtime.warmup_epochs, cfg.runtime.warmup_lr
        )


        if (epoch + 1) % cfg.runtime.validation_frequency == 0:
            val_loss, val_acc, val_acc5, val_metric, _val_log = validate_func(
                run_manager, val_settings, epoch=epoch
            )
            # wandb.log(dict(val_acc=val_acc, val_metric=val_metric), step=epoch)
            # best_acc
            # is_best = val_acc > run_manager.best_acc
            is_best = val_metric > run_manager.best_acc
            # run_manager.best_acc = max(run_manager.best_acc, val_acc)
            run_manager.best_acc = max(run_manager.best_acc, val_metric)

            val_log = (
                "Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} metric={4:.3f} ({5:.3f})".format(
                    epoch + 1 - cfg.runtime.warmup_epochs,
                    run_manager.n_epochs,
                    val_loss,
                    val_acc,
                    val_metric,
                    run_manager.best_acc,
                )
            )
            val_log += ", Train top-1 {top1:.3f}, Train loss {loss:.3f}\t".format(
                top1=train_top1, loss=train_loss
            )
            val_log += _val_log
            run_manager.write_log(val_log, "valid", should_print=False)

            if not is_best:
                no_improveed_epochs += 1
            else:
                no_improveed_epochs = 0

            if no_improveed_epochs >= max_no_improved_epochs:
                run_manager.write_log(
                    "Early stopping at epoch %d" % epoch
                )
                no_improveed_epochs = 0
                if run_manager.cfg.get("finetune", False):
                    run_manager.finetune = True
                    run_manager.cfg.finetune = False
                    best_path = os.path.join(run_manager.save_path, "model_best.pth.tar")
                    load_models(run_manager, run_manager.net, model_path=best_path)
                else:
                    break

            run_manager.save_model(
                {
                    "epoch": epoch,
                    "best_acc": run_manager.best_acc,
                    "optimizer": run_manager.optimizer.state_dict(),
                    "state_dict": run_manager.net.state_dict(),
                },
                is_best=is_best,
                epoch=epoch
            )
    # wandb.finish()

def load_models(run_manager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location="cpu")["state_dict"]
    dynamic_net.load_state_dict(init)
    run_manager.write_log("Loaded init from %s" % model_path, "valid")






