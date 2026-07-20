# FMCerNet

FMCerNet is a research codebase for patch-level and slide-level cervical
cytology/pathology image classification. It provides configurable image
backbones, classification heads, multiple-instance learning (MIL) models,
distributed training, evaluation, whole-slide inference, and data-preparation
utilities.

## Repository layout

```text
configs/
  dataset/            Patch-level dataset configurations
  model/              Backbone and classification-head configurations
  slide/              Slide-level datasets, MIL models, and strategies
  strategy_patch.py   Patch-level optimization and scheduling
fmcernet/
  datasets/           Active patch and slide data loaders
  nets/
    backbone/         Image backbone implementations
    classifier/       Patch-level classification heads
    MIL/              Slide-level MIL implementations
  utils/              Distributed training, metrics, loss, and WSI utilities
tools/
  main4PatchNet.py    PatchNet and SlideNet training entry point
  test_PatchNet.py    Checkpoint evaluation entry point
  process_WSI/        Whole-slide inference and feature extraction
  data_process/       Dataset preparation utilities
scripts/
  analysis/           Result analysis utilities
tests/                Unit tests
checkpoints/          Local pretrained weights (not included in the repository)
data_resource/        Local images and slide features (not included)
work_dir/             Local training outputs (not included)
```

## Installation

All project scripts are expected to run in the `sam2` Conda environment.
Python 3.10 is recommended.

```bash
git clone <repository-url>
cd FMCerNet

conda create -n sam2 python=3.10 -y
conda activate sam2

pip install -r requirements.txt
pip install -e .
```

The requirements pin CUDA-enabled PyTorch and related packages. Ensure that the
installed CUDA toolkit, GPU driver, and PyTorch build are compatible.

Pretrained weights are loaded from project-relative paths under `checkpoints/`.
Download or copy the required weights before constructing the corresponding
backbone.

## Data configuration

The active dataset types are:

| `dataset_type` | Loader | Purpose |
| --- | --- | --- |
| `multicls` | `mmpretrain.datasets.MultiLabelDataset` | Patch-level multi-label classification |
| `slide` | `fmcernet.datasets.SlideDataset` | Slide-level MIL over saved tile features |

Patch-level configurations are stored in `configs/dataset/`. They define the
image root, annotation JSON files, augmentation pipeline, input size, and batch
size. The default C2FHead example uses:

```text
data_resource/LCerScan/WS800/
  images/
  annofiles/
    multilabel_puretrain.json
    multilabel_val.json
```

Slide-level loading expects CSV metadata and one feature tensor per patient:

```text
data_resource/LCerScan/
  train.csv
  val.csv
  WS800/
    slide_feat_ours/
      <patientId>.pt
```

The slide CSV must contain `patientId` and `slide_clsname` columns.

Supported slide feature formats are:

- `pn_only`
- `pn_posprob`
- `pos_only_top1`
- `all_prob_weighted`
- `raw_pn_pos_tokens`

## Patch-level training

Training merges three configuration files in this order:

1. Dataset configuration
2. Model configuration
3. Training strategy

### Single GPU

```bash
conda activate sam2

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_port=12340 \
  tools/main4PatchNet.py \
  configs/dataset/l_cerscan_ws800.py \
  configs/model/c2fhead.py \
  configs/strategy_patch.py \
  --record_save_dir work_dir/patch/c2fhead
```

### Multiple GPUs

```bash
conda activate sam2

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=12345 \
  tools/main4PatchNet.py \
  configs/dataset/l_cerscan_ws800.py \
  configs/model/c2fhead.py \
  configs/strategy_patch.py \
  --record_save_dir work_dir/patch/c2fhead
```

Each run creates a timestamped directory:

```text
work_dir/patch/c2fhead/<timestamp>/
  config.py
  result.log
  checkpoints/
    best.pth
```

To initialize from an existing model, set `load_from` in
`configs/strategy_patch.py`.

## Evaluation

Evaluate the consolidated configuration and checkpoint saved by a training run:

```bash
conda activate sam2

RUN_DIR=work_dir/patch/c2fhead/<timestamp>

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_port=12347 \
  tools/test_PatchNet.py \
  "${RUN_DIR}/config.py" \
  "${RUN_DIR}/checkpoints/best.pth" \
  "${RUN_DIR}" \
  --save_result
```

Optional arguments:

- `--val_json <path>` overrides `cfg.val_datasets["ann_file"]`.
- `--save_result` writes predictions to `<save_dir>/pred_result.pkl`.

## Available backbones

Backbone names are defined by `allowed_backbone_type` in
`fmcernet/nets/get_backbone.py`. Configuration values are grouped under
`configs/model/`.

| Group | List |
| --- | --- |
| Common | `resnet`, `convnext`, `vit`, `dinov2`, `dinov3` |
| Histopathology | `ctranspath`, `uni2-h`, `virchow2`, `gpfm`, `genbio-pathfm` |
| Cytopathology | `smartccs`, `cytofm`, `unicas`, `lfreqvit` |

Select a backbone in a model configuration:

```python
backbone_type = "lfreqvit"
backbone_cfg = _base_.backbone_cfgdict[backbone_type]
```

LFreqViT currently requires 1024 x 1024 inputs and supports `dtcwt`,
`dwt_haar`, and `fft_radial` frequency operators.

## Available classification heads

Classification-head names are defined by `allowed_classifier_type` in
`fmcernet/nets/get_classifier.py` and selected through `taskhead_model`.

| Name | Example configuration |
| --- | --- |
| `binary_linear` | `configs/model/binary_linear.py` |
| `mc_linear` | `configs/model/mc_linear.py` |
| `mlc_linear` | `configs/model/mlc_linear.py` |
| `chief` | `configs/model/chief.py` |
| `ml_decoder` | `configs/model/ml_decoder.py` |
| `query2label` | `configs/model/query2label.py` |
| `mlc_nc` | `configs/model/mlc_nc.py` |
| `c2fhead` | `configs/model/c2fhead.py` |

The default C2FHead configuration uses:

```python
backbone_type = "lfreqvit"
taskhead_model = "c2fhead"
```

## Slide-level MIL training

Available MIL models are:

- `ABMIL`
- `TransMIL`
- `DSMIL`
- `RRTMIL`
- `CAMIL`

The default slide example combines RRTMIL with `pn_posprob` features:

```bash
conda activate sam2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  --master_port=12345 \
  tools/main4PatchNet.py \
  configs/slide/0_dataset_cfg.py \
  configs/slide/rrtmil.py \
  configs/slide/1_strategy_slide.py \
  --record_save_dir work_dir/slide/rrtmil_pn_posprob
```

Here, `configs/slide/rrtmil.py` uses `in_dim = 517`, while
`configs/slide/1_strategy_slide.py` uses `format_type = "pn_posprob"`.
Each tile is represented by its negative/positive feature vector followed by
five positive-class probabilities.

## Whole-slide feature extraction

`tools/process_WSI/extract_all_patch.py` performs distributed patch extraction
and slide-feature generation. FMCerNet does not bundle a vendor-specific WSI
backend. Implement `fmcernet.utils.WSIReader` in your own package and select it
with `FMCERNET_WSI_READER` using `package.module:ReaderClass` syntax.

```bash
conda activate sam2

FMCERNET_WSI_READER=your_package.reader:YourWSIReader \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  --master_port=12341 \
  tools/process_WSI/extract_all_patch.py
```

The custom reader must expose pyramid dimensions and downsample factors, return
RGB `uint8` NumPy arrays from `read_region`, and release resources in `close`.
The slide metadata CSV used by this tool must provide `wsi_path`,
`slide_clsname`, and `patientId` columns. Before running, review the
project-relative dataset, checkpoint, and output paths near the top of the
script.

## Tests

Run the unit tests in the `sam2` environment:

```bash
conda activate sam2
python -m unittest discover -s tests -p "test_*.py"
```
