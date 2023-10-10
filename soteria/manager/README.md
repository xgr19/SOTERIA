Manager系列
---
### 1.NSGAManager
依然采取渐进式训练的方式，随机采样网络，使用差分进化的方式完成交叉变异，使用非支配排序，得到front，
现在的问题是1）好的预测器 2）front过少，并且他们交叉变异后是否能产生更好的子代。
### 1.RandomManager
在指定的范围，随机采样训练，相比于Manager增加以下功能：
* 提供训练与验证的子网配置，接收子网的acc与[param/flops/latency]
* 保存子网的配置与acc,[param/flops/latency]

|新增函数|解释|
|---|---|
|random_sample|随机采样子网配置|
|profile|测试静态网络的特定属性[param/flops/latency]|
|get_settings|获取一个epoch的训练配置与测试配置|
|callback|记录对应epoch的验证配置的准确率与资源占用|
|record_path|每epoch保存配置结果的位置|

### 2.Manager
支持训练单个网络

|Data|__init__中指定的参数|
|------|----|
|build_dataloader|None|

|优化器类|__init__中指定的参数|
|------|----|
|adjust_learning_rate|init_lr, n_epochs, lr_schedule_type|
|warmup_adjust_learning_rate|init_lr|
|build_optimizer|opt_type, opt_param, init_lr, weight_decay, no_decay_keys|

|模型保存与加载|__init__中指定的参数|
|---|---|
|save_model|net|
|load_model|start_epoch, best_acc, optimizer|

|log保存|__init__中指定的参数|
|------|------------------|
|save_config|cfg, net|
|write_log|None|

|Others|__init__中指定的参数|
|---------|--------------|
|logs_path|path|
|save_path|path|

|Metric|__init__中指定的参数|
|------|-----------------|
|get_metric_names|None|
|get_metric_vals|None|
|get_metric_dict|None|
|Accuracy|None|
|update_metric|None|




