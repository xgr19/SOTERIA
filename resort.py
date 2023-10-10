import pickle
import numpy as np
import copy
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import sys
import torch
from loguru import logger
from soteria.manager import RdManager
from soteria.manager import load_models
from soteria.model import Dynamic_MobileNet1d
from soteria.utils import measure_model
from tabulate import tabulate
import yaml
from easydict import EasyDict as edict
from botdataset import BotDataset
import os
import pandas as pd
import json

def convert2mnn(net, name):
    net = net.eval().cuda()
    dummy_input = torch.randn([1, 23], device='cuda')
    input_names = ["input"]
    output_names = ["output"]
    onnx_name = name + ".onnx"
    mnn_name = name + ".mnn"

    try:
        torch.onnx.export(net, dummy_input, onnx_name, verbose=False, input_names=input_names,
                          output_names=output_names)
    except:
        print("%s maybe is a invalid ops!" % (onnx_name))
        return 0

    print("Export %s to %s" % (name, onnx_name))
    cmd = "MNNConvert -f ONNX --modelFile {} --MNNModel {} --bizCode biz".format(onnx_name, mnn_name)
    os.system(cmd)
    if not os.path.isfile(mnn_name):
        raise ValueError("Fail to convert %s to %s" % (onnx_name, mnn_name))
    os.remove(onnx_name)

    print("Convert %s to %s" % (onnx_name, mnn_name))

nsga_path = "nsga_output"
config_path = "NSGA/NSGAD1234K357E346W8.yaml"
data_path = "dataset.pkl"
key = "ops_list/meta_latency.pkl"


with open(config_path, 'r') as file:
    cfg = yaml.safe_load(file)
cfg = edict(cfg)
cfg.work_path = "Temp"

train_dataset = BotDataset(data_path, key="train")
test_dataset = BotDataset(data_path, key="valid")

model = Dynamic_MobileNet1d(cfg)
manager = RdManager(model, train_dataset, test_dataset, cfg)

ignore_name = []
fname_list = [n for n in os.listdir(nsga_path) if n not in ignore_name]
logger.info("Read models from: ")
print(fname_list)

target_p = "/checkpoint/best_log.txt"
target_checkpoint = "/checkpoint/model_best.pth.tar"

data = []
all_mnn_path = []

sample = torch.ones([1,23]).cuda()
for n in fname_list:
    fp = os.path.join(nsga_path, n)
    load_models(manager, manager.net, model_path=fp + target_checkpoint)
    if n == "OFA_D4K7E6W64_10x":
        continue
    with open(fp + target_p, 'r') as f:
        train = f.readline().replace("\n", '')
        val = f.readline().replace("\n", '')
    with open(fp + "/record/" + val, "rb") as f:
        df = pickle.load(f)
    with open(fp + "/record/" + train, "rb") as f:
        settings = pickle.load(f)[1]

    for i, setting in enumerate(settings):
        manager.net.set_active(setting)
        manager.net(sample)
        # loss, [acc1, acc5], report = manager.validate(cls_report=True)
        net = copy.deepcopy(manager.net)
        net.get_active_module()
        tar_value = measure_model(net.cuda(), [23], key=key)
        df[i, 1] = tar_value
        path = "./models/soteria_acc1_%f_latency_%f" % (100-df[i, 0], tar_value)
        all_mnn_path.append(path)
        convert2mnn(net, path)
    data.append(df)
    
data = np.vstack(data)
fronts = NonDominatedSorting().do(data)[0]
best_models = data[fronts]
best_models[:, 0] = 100 - best_models[:, 0]
report = np.hstack([np.array(["./models/soteria_acc1_%f_latency_%f" % (x[0], x[1]) for x in best_models]).reshape(-1, 1), best_models])
headers = ["Model name", "Acc", "Latency"]
print(tabulate(report, headers=headers))

