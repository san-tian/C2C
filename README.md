<div align="center">
  <img src="resource/logo.png" alt="Cache-to-Cache Logo" width="100"/>
  
  <h1>Cache-to-Cache</h1>
  <h3>Direct Semantic Communication Between Large Language Models</h3>

  <p>
    <a href="https://fuvty.github.io/C2C_Project_Page/">🌐 <b>Project Page</b></a> •
    <a href="https://arxiv.org/abs/2510.03215">📑 <b>Paper</b></a> •
    <a href="https://huggingface.co/nics-efc/C2C_Fuser">🤗 <b>HuggingFace</b></a> •
    <a href="https://huggingface.co/spaces/nics-efc/C2C_demo">🚀 <b>Live Demo</b></a>
  </p>

</div>


**原仓库采用门控将base model的KVCache与teacher model经过projector的KVCache拼接。本分支仅保留teacher model经过projector的KVCache，去除了base model的KVCache，从而对projector的效果进行消融。** 具体实现[请看这里](#my-anchor)。


## Demo

## Environment Setup

Create a new environment:

```bash
conda create -n rosetta python=3.10
conda activate rosetta
```

Install the package:

```bash
pip install -e .
```

For training and evaluation, install additional dependencies:

```bash
pip install -e ".[training,evaluation]"
```

## How to

### Train C2C Projectors

Prepare a training configuration file in `recipe/train_recipe/`. Specify the base model, teacher model, projector type and parameters, training hyperparameters, dataset, and output directory. See `recipe/train_recipe/C2C_0.6+0.5.json` for a complete example.

Run training:

```bash
# Single GPU
python script/train/SFT_train.py --config recipe/train_recipe/C2C_0.6+0.5.json

# Multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_0.6+0.5.json
```

**Run PURE C2C training:**
<a id="my-anchor"></a>
```
# example 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 script/train/SFT_train.py \
--config recipe/train_recipe/C2C_0.6+0.5_pure.json

# example 2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 script/train/SFT_train.py \
--config recipe/train_recipe/C2C_base_2.5_teacher_3.json.json

```


During training, only the C2C projector parameters are updated while both source and target models remain frozen.

### Evaluate C2C

Prepare an evaluation configuration file in `recipe/eval_recipe/`. Specify the model configuration with base model, teacher model, checkpoint directory, generation config, evaluation dataset, and output directory. See `recipe/eval_recipe/unified_eval.yaml` for a complete example.

Run evaluation:

```bash
python script/evaluation/unified_evaluator.py --config recipe/eval_recipe/unified_eval.yaml
```

