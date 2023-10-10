# _*_ coding: utf-8 _*_
import copy
import pickle

import numpy as np
from .parser import encode, decode
from .RandomManager import RdManager
import torch
from soteria.utils import get_net_device, measure_model
from pymoo.core.problem import Problem
from soteria.predictors import fit_acc_predictor
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_algorithm, get_crossover, get_mutation
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

seed = 2021
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  
random.seed(seed)
np.random.seed(seed)


class NsgaManager(RdManager):
    def __init__(self, net, train_dataset, test_dataset, cfg):
        super(NsgaManager, self).__init__(net, train_dataset, test_dataset, cfg)
        self.search_space = OFASearchSpace(self.max_net_setting, cfg.search_space)
        self.xl = np.array(cfg.xl)
        self.xu = np.array(cfg.xu)
        self.val_settings = []
        self.write_log("using NSGA-Manager")
        self.min_survival_sample = cfg.get("min_survival_sample", 8)
        self.finetune = False
        self.val_settings = []
        if self.cfg.get("load_train_settings", False):
            with open(self.cfg.load_train_settings, "rb") as f:
                self.val_settings = pickle.load(f)[1]
            with open(self.cfg.load_val_record, "rb") as f:
                self.val_record = pickle.load(f)

    def profile(self, static_net):
        return measure_model(static_net,
                             self.cfg.input_shape,
                             key=self.object_type,
                             use_emb=self.cfg.get("use_emb", False),
                             )

    def warm_up(self):
        val_settings = self.search_space.initialize(self.num_sample_val_per_epoch)
        self.val_settings += val_settings
        uniques = np.unique([self.search_space.setting2vec(s) for s in self.val_settings], axis=0)
        self.val_settings = [self.search_space.vec2settings(u) for u in uniques]
        num_settings = len(self.val_settings)
        self.val_record = np.zeros((num_settings, 2))
        for i, setting in enumerate(self.val_settings):
            self.net.set_active(setting)
            loss, (top1, top5) = self.validate(no_logs=False)
            m = copy.deepcopy(self.net)
            m.get_active_module()
            self.val_record[i, 0] = 100 - top1
            self.val_record[i, 1] = self.profile(m)
        self.write_log("[warming up] initialize settings %d" % num_settings)

    def get_settings(self, epoch=0):
        if self.finetune:
            self.val_settings = self.best_settings
            num_settings = len(self.val_settings)
            self.val_record = np.zeros((num_settings, 2))
            self.train_record = np.zeros(num_settings)
            self.num_sample_train_per_epoch = num_settings
            self.num_sample_val_per_epoch = num_settings
            self.write_log("fine tune settings: %d" % num_settings)
            return self.val_settings, self.val_settings

        if epoch == 0:
            self.warm_up()

        df = {"settings": self.val_settings, "err": self.val_record[:, 0]}
        inputs = np.array([self.search_space.setting2vec(x) for x in df["settings"]])
        targets = np.array([y for y in df["err"]])
        predictor, _ = fit_acc_predictor(inputs, targets)
        fronts = NonDominatedSorting().do(self.val_record, only_non_dominated_front=False)
        front = fronts[0]
        nd_X = inputs[front]
        problem = AuxiliarySingleLevelProblem(
            self.search_space,
            predictor,
            self.xu,
            self.xl,
            supernet=self.net,
            input_shape=self.cfg.input_shape,
            dtype=self.cfg.get("dtype", "torch.float32"),
            test_program=self.profile)
       
        method = get_algorithm(
            "nsga2", pop_size=40, sampling=nd_X,  
            crossover=get_crossover("int_two_point", prob=0.9),
            mutation=get_mutation("int_pm", eta=1.0),
            eliminate_duplicates=True)
        
        res = minimize(problem, method, termination=('n_gen', 10), save_history=True, verbose=True)
     
        uniques = np.unique([self.search_space.mask_settings(x) for x in res.X], axis=0)
      
        new_front_settings = np.array([self.search_space.vec2settings(x) for x in uniques])
       
        previous_settings = np.array(df["settings"])
        
        new_front_settings = new_front_settings[np.logical_not([x in previous_settings for x in new_front_settings])]
        train_settings = new_front_settings.tolist()

        idx = 0
        stage = np.min([2, len(fronts)])
        while idx < stage:
            if len(train_settings) >= self.min_survival_sample:
                break
            train_settings += previous_settings[fronts[idx]].tolist()
            idx += 1

        if len(train_settings) < self.min_survival_sample:
            train_settings += self.search_space.initialize(2 + self.min_survival_sample - len(train_settings))[2:]
        np.random.shuffle(train_settings)
        self.val_settings = train_settings

    
        num_settings = len(self.val_settings)
        self.val_record = np.zeros((num_settings, 2))
        self.train_record = np.zeros(num_settings)
        self.num_sample_train_per_epoch = num_settings
        self.num_sample_val_per_epoch = num_settings
        self.write_log("num of proposed settings: %d" % num_settings)
        return train_settings, self.val_settings

    def initialize(self):
        self.val_record = np.zeros((self.num_sample_val_per_epoch, 2))
        self.train_record = np.zeros(self.num_sample_train_per_epoch)
        train_settings = self.search_space.initialize(self.num_sample_train_per_epoch)
        self.val_settings = train_settings[:self.num_sample_val_per_epoch]
        return train_settings, self.val_settings


class OFASearchSpace:
    def __init__(self, setting, search_space):
        self.num_blocks = 2  
        self.kernel_size = [3, 5, 7]  
        self.exp_ratio = [3, 4, 6]  
        self.depth = [1, 2, 3, 4] 
        self.embed_dim = [8, 16, 32, 64]
        self.setting = setting
        self.search_space = search_space

    def mask_settings(self, x):
        setting = self.vec2settings(x)
        return self.setting2vec(setting)

    def archi2setting(self, archi):
        setting = copy.deepcopy(self.setting)
        category = decode(setting)
        category["active_kernel_size"] = []
        category["active_expand_ratio"] = []

        idx = 0
        for dep in archi["d"]:
            end = idx + dep
            category["active_kernel_size"] += self.extend(archi["ks"][idx:end])
            category["active_expand_ratio"] += self.extend(archi["e"][idx:end])
            idx += dep

        category["active_kernel_size"] += [archi["ks"][-1]]
        category["active_expand_ratio"] += [archi["e"][-1]]
        category["active_stem0_depth"] = [archi["d"][0]]
        category["active_stem1_depth"] = [archi["d"][1]]
        category["active_embed_dim"] = archi["w"]
        return encode(category, setting)

    def sample(self, n_samples=1, nb=None, ks=None, e=None, d=None, w=None):
        """ randomly sample a architecture"""
        nb = self.num_blocks if nb is None else nb
        ks = self.search_space.active_kernel_size if ks is None else ks
        e = self.search_space.active_expand_ratio if e is None else e
        d = self.search_space.active_stem0_depth if d is None else d
        w = self.search_space.active_embed_dim if w is None else w

        data = []
        for n in range(n_samples):
            width = np.random.choice(w, 1, replace=True).tolist()
            # first sample layers
            depth = np.random.choice(d, nb, replace=True).tolist()
            # then sample kernel size, expansion rate and resolution
            kernel_size = np.random.choice(ks, size=int(np.sum(depth)) + 1, replace=True).tolist()
            exp_ratio = np.random.choice(e, size=int(np.sum(depth)) + 1, replace=True).tolist()
            data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, "w": width})
        return data

    def initialize(self, n_doe):
        ks = self.search_space.active_kernel_size
        er = self.search_space.active_expand_ratio
        dep = self.search_space.active_stem0_depth
        w = self.search_space.active_embed_dim
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = [
            self.sample(1, ks=[min(ks)], e=[min(er)],
                        d=[min(dep)], w=[min(w)])[0],

            self.sample(1, ks=[max(ks)], e=[max(er)],
                        d=[max(dep)], w=[min(w)])[0]
        ]
        data.extend(self.sample(n_samples=n_doe - 2))
        result = []
        for d in data:
            result.append(self.archi2setting(d))
        return result

    @staticmethod
    def extend(x, L=4, content=3):
        while len(x) < L:
            x.append(content)
        return x

    def pad_zero(self, x, depth):
        # pad zeros to make bit-string of equal length
        new_x, counter = [], 0
        for d in depth:
            for _ in range(d):
                new_x.append(x[counter])
                counter += 1
            if d < max(self.depth):
                new_x += [0] * (max(self.depth) - d)
        return new_x

    def encode(self, config):
        # encode config ({'ks': , 'd': , etc}) to integer bit-string [1, 0, 2, 1, ...]
        x = []
        depth = [np.argwhere(_x == np.array(self.depth))[0, 0] for _x in config['d']]
        kernel_size = [np.argwhere(_x == np.array(self.kernel_size))[0, 0] for _x in config['ks'][:-1]]
        exp_ratio = [np.argwhere(_x == np.array(self.exp_ratio))[0, 0] for _x in config['e'][:-1]]
        kernel_size = self.pad_zero(kernel_size, config['d'])
        exp_ratio = self.pad_zero(exp_ratio, config['d'])
        width = [np.argwhere(config['w'][-1] == np.array(self.embed_dim))[0, 0]]

        for i in range(len(depth)):
            x = x + [depth[i]] + kernel_size[i * max(self.depth):i * max(self.depth) + max(self.depth)] \
                + exp_ratio[i * max(self.depth):i * max(self.depth) + max(self.depth)]
        return x + [np.argwhere(config['ks'][-1] == np.array(self.kernel_size))[0, 0]] + [
            np.argwhere(config['e'][-1] == np.array(self.exp_ratio))[0, 0]] + width

    def decode(self, x):
        depth, kernel_size, exp_rate, width = [], [], [], []
        for i in range(0, len(x) - 3, 9):
            depth.append(self.depth[x[i]])
            kernel_size.extend(np.array(self.kernel_size)[x[i + 1:i + 1 + self.depth[x[i]]]].tolist())
            exp_rate.extend(np.array(self.exp_ratio)[x[i + 5:i + 5 + self.depth[x[i]]]].tolist())

        kernel_size += [self.kernel_size[x[-3]]]
        exp_rate += [self.exp_ratio[x[-2]]]
        width += [self.embed_dim[x[-1]]]
        return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 'w': width}

    @staticmethod
    def setting2archi(setting):
        category = decode(setting)
        archi = {"ks": category["active_kernel_size"], "e": category["active_expand_ratio"],
                 "d": category["active_stem0_depth"] + category["active_stem1_depth"],
                 "w": category["active_embed_dim"]}
        return archi

    def setting2vec(self, setting):
        archi = self.setting2archi(setting)
        return self.encode(archi)

    def vec2settings(self, vec):
        archi = self.decode(vec)
        return self.archi2setting(archi)


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self,
                 search_space,
                 predictor,
                 xu,
                 xl,
                 supernet=None,
                 test_program=None,
                 input_shape=[1, 256],
                 dtype="torch.float32"):
        super().__init__(n_var=21, n_obj=2, n_constr=0, type_var=np.int64)

        self.ss = search_space
        self.predictor = predictor
        self.engine = supernet
        self.test_program = test_program
        self.input_shape = input_shape
        self.device = get_net_device(self.engine)
        self.xu = xu
        self.xl = xl
        self.dtype = dtype

    def warm_up(self):
        x = torch.ones([1] + self.input_shape).to(self.device)
        _ = self.engine(x)

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)

        top1_err = self.predictor.predict(x)[:, 0]  

        for i, (_x, err) in enumerate(zip(x, top1_err)):
            setting = self.ss.vec2settings(_x)
            self.engine.set_active(setting)
            self.warm_up()
            static_net = copy.deepcopy(self.engine)
            static_net.get_active_module()
            f[i, 1] = self.test_program(static_net)
            f[i, 0] = err
        out["F"] = f
