# _*_ coding: utf-8 _*_
from soteria.utils import *
import soteria.dynn as dynn
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from soteria.metric import *
import json
import time
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
import random
torch.manual_seed(2022)
torch.cuda.manual_seed(2022)      
torch.cuda.manual_seed_all(2022)   
random.seed(2022)
np.random.seed(2022)
torch.backends.cudnn.deterministic = True


class Manager():
    def __init__(self, net, train_dataset, test_dataset, cfg):
        # ----------Init----------#
        run_cfg = cfg.runtime
        self.path = cfg.work_path
        self.net = net
        self.n_epochs = run_cfg.n_epochs
        self.start_epoch = 0
        self.best_acc = 0
        self.cfg = cfg
        # optimizer
        self.init_lr = run_cfg.init_lr
        self.lr_schedule_type = run_cfg.lr_schedule_type
        self.opt_type = run_cfg.opt_type
        self.opt_param = {
            "momentum": run_cfg.momentum,
            "nesterov": not run_cfg.no_nesterov}
        self.weight_decay = float(run_cfg.weight_decay)
        self.no_decay_keys = run_cfg.no_decay_keys

        os.makedirs(self.path, exist_ok=True)

        # ----------target device----------#
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        # ----------initial model----------#
        init_models(self.net, run_cfg.model_init)
        self.save_config()

        # ----------criterion----------#
        if run_cfg.label_smoothing > 0:
            self.train_criterion = (
                lambda pred, target: dynn.cross_entropy_with_label_smoothing(
                    pred, target, run_cfg.label_smoothing))
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # ----------optimizer----------#
        if self.no_decay_keys:
            keys = self.no_decay_keys.split("#")
            net_params = [
                self.net.get_parameters(
                    keys, mode="exclude"
                ),  # parameters with weight decay
                self.net.get_parameters(
                    keys, mode="include"
                ),  # parameters without weight decay
            ]
        else:
            try:
                net_params = self.net.weight_parameters()
            except Exception:
                net_params = []
                for param in self.net.parameters():
                    if param.requires_grad:
                        net_params.append(param)
        self.optimizer = self.build_optimizer(net_params)

        # ----------dataloader----------#
        self.train_dataloader = self.build_dataloader(train_dataset,
                                                      run_cfg.batch_size,
                                                      run_cfg.num_workers)
        self.test_dataloader = self.build_dataloader(test_dataset,
                                                     run_cfg.batch_size,
                                                     run_cfg.num_workers)
        self.nBatch = len(self.train_dataloader)
        self.best_settings = None

    """dataloader"""

    @staticmethod
    def build_dataloader(dataset, batch_size, num_workers):
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=True)
        return loader

    """optimizer"""

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """adjust learning of a given optimizer and return the new learning rate"""
        new_lr = calc_learning_rate(
            epoch, self.init_lr, self.n_epochs, batch, nBatch, self.lr_schedule_type
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(
            self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0
    ):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def build_optimizer(self, net_params):
        return build_optimizer(
            net_params,
            self.opt_type,
            self.opt_param,
            self.init_lr,
            self.weight_decay,
            self.no_decay_keys,
        )

    """save and load model"""

    def save_model(self,
                   checkpoint=None,
                   is_best=False,
                   model_name=None,
                   epoch=0):
        if checkpoint is None:
            checkpoint = {"state_dict": self.net.state_dict()}

        if model_name is None:
            model_name = "checkpoint.pth.tar"

        latest_fname = os.path.join(self.save_path, "latest.txt")
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, "w") as fout:
            fout.write(model_path + "\n")
        torch.save(checkpoint, model_path)

        if is_best:
            if self.__dict__.get("val_settings", False):
                self.best_settings = self.__dict__.get("val_settings", False)
            best_log_fname = os.path.join(self.save_path, "best_log.txt")
            with open(best_log_fname, "w") as fout:
                fout.write("%d-train.pkl" % epoch + "\n" + "%d-val.pkl" % epoch)
            best_path = os.path.join(self.save_path, "model_best.pth.tar")
            torch.save({"state_dict": checkpoint["state_dict"]}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline()
                if model_fname[-1] == "\n":
                    model_fname = model_fname[:-1]
      
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = "%s/checkpoint.pth.tar" % self.save_path
                with open(latest_fname, "w") as fout:
                    fout.write(model_fname + "\n")
            print("=> loading checkpoint '{}'".format(model_fname))
            checkpoint = torch.load(model_fname, map_location="cpu")
        except Exception:
            print("fail to load checkpoint from %s" % self.save_path)
            return {}

        self.net.load_state_dict(checkpoint["state_dict"])
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            self.best_acc = checkpoint["best_acc"]
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print("=> loaded checkpoint '{}'".format(model_fname))
        return checkpoint

    """Metric"""

    @staticmethod
    def get_metric_names():
        return "top1", "top5"

    @staticmethod
    def get_metric_vals(metric_dict, return_dict=False):
        if return_dict:
            return {key: metric_dict[key].avg for key in metric_dict}
        else:
            return [metric_dict[key].avg for key in metric_dict]

    @staticmethod
    def get_metric_dict():
        return {
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }

    @staticmethod
    def Accuracy(output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        return acc1, acc5

    @staticmethod
    def update_metric(metric_dict, output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        metric_dict["top1"].update(acc1[0].item(), output.size(0))
        metric_dict["top5"].update(acc5[0].item(), output.size(0))

    """log operation"""

    def save_config(self, extra_run_config=None, extra_net_config=None):
        import json
        """dump run_config and net_config to the model_folder"""
        run_save_path = os.path.join(self.path, "run.config")
        if not os.path.isfile(run_save_path):
            run_config = self.cfg
            if extra_run_config is not None:
                run_config = Merge(run_config, extra_run_config)
            json.dump(run_config, open(run_save_path, "w"), indent=4)
            print("Run configs dump to %s" % run_save_path)

        try:
            net_save_path = os.path.join(self.path, "net.config")
            net_config = self.net.config
            if extra_net_config is not None:
                net_config = Merge(net_config, extra_run_config)
            json.dump(net_config, open(net_save_path, "w"), indent=4)
            print("Network configs dump to %s" % net_save_path)
        except Exception:
            print("%s do not support net config" % type(self.net))

    def write_log(self, log_str, prefix="valid", should_print=True, mode="a"):
        write_log(self.logs_path, log_str, prefix, should_print, mode)

    """others"""

    @property
    def logs_path(self):
        if self.__dict__.get("_logs_path", None) is None:
            logs_path = os.path.join(self.path, "logs")
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__["_logs_path"] = logs_path
        return self.__dict__["_logs_path"]

    @property
    def save_path(self):
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.path, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    def validate(
            self,
            epoch=0,
            net=None,
            no_logs=False,
            train_mode=False,
            cls_report=False
    ):
        if net is None:
            net = self.net

        if train_mode:
            net.train()
        else:
            net.eval()

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()
        data_loader = self.test_dataloader
        nBatch = len(data_loader)
        if cls_report:
            output_set = []
            gt_set = []

        with torch.no_grad():
            with tqdm(
                    total=nBatch,
                    desc="Validate Epoch #{}".format(epoch + 1),
                    disable=no_logs,
            ) as t:
                for i, (x, y_true) in enumerate(data_loader):
                    x, y_true = x.to(self.device), y_true.to(self.device)
                  
                    output = net(x)
                    if cls_report:
                        pred_ = torch.argmax(output, dim=-1).cpu().numpy()
                        output_set.append(pred_)
                        gt_set.append(y_true.cpu().numpy())
                    loss = self.test_criterion(output, y_true)
                   
                    self.update_metric(metric_dict, output, y_true)

                    losses.update(loss.item(), x.size(0))
                    t.set_postfix(
                        {
                            "loss": losses.avg,
                            **self.get_metric_vals(metric_dict, return_dict=True),
                        }
                    )
                    t.update(1)
        if cls_report:
            report = classification_report(np.concatenate(gt_set, axis=0), np.concatenate(output_set, axis=0), output_dict=True)
            return losses.avg, self.get_metric_vals(metric_dict), report

        return losses.avg, self.get_metric_vals(metric_dict)

    def train_one_epoch(self, epoch, warmup_epochs=0, warmup_lr=0):
        self.net.train()

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()
        data_time = AverageMeter()
        train_loader = self.train_dataloader
        nBatch = self.nBatch

        with tqdm(
                total=nBatch,
                desc="Train Epoch #{}".format(epoch + 1),
        ) as t:
            end = time.time()
            for i, (x, y_true) in enumerate(train_loader):
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.warmup_adjust_learning_rate(
                        self.optimizer,
                        warmup_epochs * nBatch,
                        nBatch,
                        epoch,
                        i,
                        warmup_lr,
                    )
                else:
                    new_lr = self.adjust_learning_rate(
                        self.optimizer, epoch - warmup_epochs, i, nBatch
                    )

                x, y_true = x.to(self.device), y_true.to(self.device)
                target = y_true
               
                output = self.net(x)
                loss = self.train_criterion(output, y_true)
                
                self.net.zero_grad()  
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.net.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()

                losses.update(loss.item(), x.size(0))
                self.update_metric(metric_dict, output, target)

                t.set_postfix(
                    {
                        "loss": losses.avg,
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        "lr": new_lr,
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()
        return losses.avg, self.get_metric_vals(metric_dict)

    def train(self, warmup_epoch=0, warmup_lr=0):
        for epoch in range(self.start_epoch, self.n_epochs + warmup_epoch):
            train_loss, (train_top1, train_top5) = self.train_one_epoch(
                epoch, warmup_epoch, warmup_lr
            )
            # wandb.log(dict(train_loss=train_loss, train_top1=train_top1), step=epoch)

            if (epoch + 1) % self.cfg.runtime.validation_frequency == 0:
                loss, (top1, top5) = self.validate(epoch)
                is_best = top1 > self.best_acc
                self.best_acc = max(self.best_acc, top1)
                val_log = "Valid [{0}/{1}]\tloss {2:.3f}\t{5} {3:.3f} ({4:.3f})".format(
                    epoch + 1 - warmup_epoch,
                    self.n_epochs,
                    loss,
                    top1,
                    self.best_acc,
                    self.get_metric_names()[0],
                )
                val_log += "\t{2} {0:.3f}\tTrain {1} {top1:.3f}\tloss {train_loss:.3f}\t".format(
                    top5,
                    *self.get_metric_names(),
                    top1=train_top1,
                    train_loss=train_loss
                )
                self.write_log(val_log, prefix="valid", should_print=False)
                # wandb.log(dict(val_loss=loss, val_top1=top1), step=epoch)
            else:
                is_best = False

            self.save_model(
                {
                    "epoch": epoch,
                    "best_acc": self.best_acc,
                    "optimizer": self.optimizer.state_dict(),
                    "state_dict": self.net.state_dict(),
                },
                is_best=is_best,
            )
        # wandb.finish()

