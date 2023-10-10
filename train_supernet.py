# _*_ coding: utf-8 _*_
from soteria.manager import Manager, RdManager, NsgaManager
from soteria.manager import load_models, train
from soteria.model import Dynamic_MobileNet1d
import yaml
from easydict import EasyDict as edict
from botdataset import BotDataset
import os
import torch
import argparse
import random
import numpy as np
import datetime


seed = 2022
torch.manual_seed(seed)           
torch.cuda.manual_seed(seed)      
torch.cuda.manual_seed_all(seed)  
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, help='config')
parser.add_argument('-gpu', type=str, help="cuda id")
parser.add_argument('--ptr', type=str, help="pretrained model", default="")
parser.add_argument('--tea_ptr', type=str, help="teacher model", default="")
parser.add_argument('--data_path', type=str, help="path of the dataset", default="")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
with open(args.c, 'r') as file:
    cfg = yaml.safe_load(file)
cfg = edict(cfg)


if len(args.ptr) >= 1:
    cfg.checkpoint_path = os.path.join(args.ptr, "checkpoint/model_best.pth.tar")
    if len(args.tea_ptr) >= 1:
        cfg.distilation.teacher_model_path = os.path.join(args.tea_ptr, "checkpoint/model_best.pth.tar")
    best_log_fname = os.path.join(args.ptr, "checkpoint/best_log.txt")
    with open(best_log_fname, "r") as f:
        record_path = os.path.join(args.ptr, "record")
        cfg.load_train_settings = os.path.join(record_path, f.readline().replace("\n", ''))
        cfg.load_val_record = os.path.join(record_path, f.readline().replace("\n", ''))
        print("load_train_settings:", cfg.load_train_settings)
        print("load_val_record:", cfg.load_val_record)


train_dataset = BotDataset(args.data_path, key="train")
test_dataset = BotDataset(args.data_path, key="valid")



model = Dynamic_MobileNet1d(cfg)

task = cfg.Task
if task == "full":
    manager = Manager(model, train_dataset, test_dataset, cfg)
    manager.write_log("training supernet")
    time_begin = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    manager.write_log("Training begin at {}".format(time_begin))
    manager.train(cfg.runtime.warmup_epochs, cfg.runtime.warmup_lr)

else:
    manager = NsgaManager(model, train_dataset, test_dataset, cfg)
    manager.write_log("training supernet-{}".format(task))
    if cfg.distilation.kd_ratio > 0:
        cfg.teacher_model = Dynamic_MobileNet1d(cfg)
        cfg.teacher_model.cuda()
        manager.write_log("load teacher model from %s" % cfg.distilation.teacher_model_path)
        load_models(
            manager, cfg.teacher_model, model_path=cfg.distilation.teacher_model_path
        )
        manager.cfg = cfg
    load_models(manager, manager.net, model_path=cfg.checkpoint_path)
    time_begin = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    manager.write_log("Training begin at {}".format(time_begin))
    train(manager)

time_end = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
manager.write_log("Training end at {}".format(time_end))