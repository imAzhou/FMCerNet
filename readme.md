# FMCerNet

FMCerNet 是一个面向宫颈细胞学/病理图像的 patch 级和 slide 级训练项目。当前主要入口包括：

- `main4PatchNet.py`：PatchNet/SlideNet 训练入口。
- `tools/test_PatchNet.py`：加载训练好的 checkpoint 做验证/测试。
- `tools/vis_bbox_pred.py`：保存预测可视化结果。

## 项目结构

```text
configs/
  dataset/        数据集配置
  model/          backbone、classifier 和 patch 模型配置
  slide/          slide 级 MIL 配置
fmcernet/
  datasets/       数据加载
  nets/           PatchNet、SlideNet、backbone、classifier、MIL 模块
  utils/          训练、评估、可视化、WSI 相关工具
tools/
  test_PatchNet.py    验证/测试 checkpoint
  vis_bbox_pred.py    保存预测可视化图
checkpoints/      backbone 预训练权重
```

## 安装

建议先创建独立 Python 环境，再安装依赖和本项目。

```bash
cd /shared_storage/xzly/codes/FMCerNet

# 可选：创建 conda 环境
# conda create -n fmcernet python=3.10 -y
# conda activate fmcernet

pip install -r requirements.txt
pip install -e .
```

当前 `requirements.txt` 主要固定了这些版本：

```text
torch==2.7.1+cu128
torchvision==0.22.1
xformers==0.0.31.post1
transformers==4.56.2
timm==1.0.19
mmcv==2.2.0
mmdet==3.3.0
mmpretrain==1.2.0
scikit-learn==1.6.1
nystrom-attention==0.0.14
```

注意：

- backbone 配置默认会从 `checkpoints/` 下读取预训练权重，训练前需要确认对应文件存在。

## 训练流程

训练时需要传入 3 个配置文件：

1. 数据集配置，例如 `configs/dataset/l_cerscan_ws800.py`
2. 模型配置，例如 `configs/model/wscernet.py`
3. 训练策略配置，例如 `configs/strategy_patch.py`

单卡训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12340 main4PatchNet.py \
  configs/dataset/l_cerscan_ws800.py \
  configs/model/wscernet.py \
  configs/strategy_patch.py \
  --record_save_dir work_dir/debug
```

多卡训练示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12345 main4PatchNet.py \
  configs/dataset/l_cerscan_ws800.py \
  configs/model/wscernet.py \
  configs/strategy_patch.py \
  --record_save_dir work_dir/mlc/ours/wscernet
```

训练输出会保存到 `--record_save_dir` 下。最优模型默认保存为：

```text
<record_save_dir>/<timestamp>/checkpoints/best.pth
```

如果要换 backbone 或 classifier，编辑对应的 `configs/model/*.py`，例如：

```python
backbone_type = 'fusionnet'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]
taskhead_model = 'wscer_mlc'
```

如果要从已有模型继续训练或初始化，在 `configs/strategy_patch.py` 中设置：

```python
load_from = 'path/to/checkpoint.pth'
```

## 推理/测试流程

### 验证集评估

使用 `test_PatchNet.py` 加载训练时保存的 `config.py` 和 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12347 test_PatchNet.py \
  work_dir/mlc/ours/wscernet/config.py \
  work_dir/mlc/ours/wscernet/checkpoints/best.pth \
  work_dir/mlc/ours/wscernet \
  --save_result
```

可选参数：

- `--val_json <path>`：覆盖配置里的 `cfg.val_datasets['ann_file']`
- `--save_result`：保存预测结果到 `<save_dir>/pred_result.pkl`


## 可选 Backbone

backbone 名称来自 `fmcernet/nets/get_backbone.py` 的 `allowed_backbone_type`，具体参数在 `configs/model/*backbone_cfg.py` 中维护。

| 名称 | 配置分组 | 默认 checkpoint |
| --- | --- | --- |
| `resnet` | common | `checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth` |
| `convnext` | common | `checkpoints/dinov3-convnext-base-pretrain-lvd1689m` |
| `vit` | common | `checkpoints/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth` |
| `dinov2` | common | `checkpoints/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth` |
| `dinov3` | common | `checkpoints/dinov3-vitl16-pretrain-lvd1689m` |
| `smartccs` | cytopathology | `checkpoints/CCS_vitl_100M.pth` |
| `fusionnet` | cytopathology | `checkpoints/CCS_vitl_100M.pth` |
| `cytofm` | cytopathology | `checkpoints/cytofm_weights.pth` |
| `unicas` | cytopathology | `checkpoints/UniCAS.pth` |
| `ctranspath` | histopathology | `checkpoints/ctranspath.pth` |
| `uni` | histopathology | `checkpoints/uni.bin` |
| `uni2-h` | histopathology | `checkpoints/uni2-h.bin` |
| `virchow` | histopathology | `checkpoints/virchow.safetensors` |
| `virchow2` | histopathology | `checkpoints/virchow2.safetensors` |
| `gpfm` | histopathology | `checkpoints/GPFM.pth` |
| `genbio-pathfm` | histopathology | `checkpoints/genbio-pathfm.pth` |

## 可选 Classifier

classifier 名称来自 `fmcernet/nets/get_classifier.py` 的 `allowed_classifier_type`，在模型配置中通过 `taskhead_model` 选择。

| 名称 | 对应配置/示例 |
| --- | --- |
| `binary_linear` | `configs/model/binary_linear.py` |
| `mc_linear` | `configs/model/mc_linear.py` |
| `mlc_linear` | `configs/model/mlc_linear.py` |
| `chief` | `configs/model/chief.py` |
| `ml_decoder` | `configs/model/ml_decoder.py` |
| `query2label` | `configs/model/query2label.py` |
| `mlc_nc` | `configs/model/mlc_nc.py` |
| `wscer_mlc` | `configs/model/wscernet.py` |

当前默认的 `configs/model/wscernet.py` 使用：

```python
backbone_type = 'fusionnet'
taskhead_model = 'wscer_mlc'
```

## 常用配置文件

- 数据集配置：`configs/dataset/*.py`
- 模型配置：`configs/model/*.py`
- 训练策略：`configs/strategy_patch.py`
- slide 级配置：`configs/slide/*.py`

