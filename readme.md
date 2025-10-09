<div align="center">
<h2><center>👉 Ctrl-World: A Controllable Generative World Model for Robot Manipulation </h2>

[Yanjiang Guo*](https://robert-gyj.github.io), [Lucy Xiaoyang Shi*](https://cospui.github.io),  [Jianyu Chen](http://people.iiis.tsinghua.edu.cn/~jychen/), [Chelsea Finn](https://causallu.com)

 \*Equal contribution; Stanford University, Tsinghua University


<a href='https://arxiv.org/abs/2412.14803'><img src='https://img.shields.io/badge/ArXiv-2412.14803-red'></a> 
<a href='https://sites.google.com/view/ctrl-world'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>

This repo is the official PyTorch implementation for  [**Ctrl-World**](https://sites.google.com/view/ctrl-world) paper.

**TL; DR:** Ctrl-World is an action-conditioned world model compatible with modern VLA policies and enables policy-in-the-loop rollouts entirely in imagination, which can be used to evaluate and improve the instruction following ability of VLA. 

<p>
    <img src="synthetic_traj/gallery/ctrl_world.jpg" alt="wild-data" width="100%" />
</p>
<!-- synthetic_traj/gallery/ctrl_world.jpg -->



##  Content

**1. Generate synthetic trajectory via replay the actions recorded in DROID datasets.** 

**2. Generate synthetic trajectory via interaction with advanced VLA model $\pi_{0.5}$.**

**3. Whole training pipeline of Ctrl-World on DROID dataset.**

## Installation 🛠️


```bash
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt

#  If you want to use ctrl-world to interact with $\pi_{0.5}$ model, fowllowing the pi official repo to install the pi model dependencies. Otherwise you can skip it.
# (from https://github.com/Physical-Intelligence/openpi/tree/main)
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```


## CheckPoint and Dataset 📷


| Ckpt name     | Training type | Size |
|---------------|------------------|---------|
| [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)  | CLIP text and image encoder    |  ~600M   |
| [svd](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)  | Pretrained SVD video diffusion model   | ~8G    |
| [Ctrl-World](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) |   Ctrl-World model trained on DROID dataset  | ~8G   |
| [DROID Dataset](https://huggingface.co/datasets/cadene/droid_1.0.1) |   Opensourced DROID dataset, ~95k traj, 564 scene    |  ~370G  |


<!-- **📊 Replay opensourced trajectory:** If you want to replay 

**📊 Replicate results on calvin abc:** If you want to replicate results on calvin abc, download the svd-robot-calvin model.

**📊  Train VPP in cunstom environments**: If you want to run VPP algorithm on your own robot, download the svd-robot model and follow instructions in the training section. -->



## Ctrl-World Inference 📊
### 📊 (1) Replay the recorded trajectories within world model.
**Task Description:** We start from an initial observation sampled from the recorded trajectories and then generate long trajectories by replaying the recorded actions. At each interaction step, a 1-second action chunk is provided to the world model, and the interaction is repeated multiple times to produce the full rollout. 

We provide a very small subset of DROID dataset in `dataset_example/droid_subset`. After download the ckpt in section 1, you can directly run the following command to replay some long trajectories:


```bash
CUDA_VISIBLE_DEVICES=0 python rollout_replay_traj.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt}
```
The rollout configuration can be found in `config.py`.
If you want to replay more trajectories, you need to download and process the original DROID datasets following the instructions in training section.


### 📊 (2) Interact with $\pi_{0.5}$ model within world model

**Task Description:** We take some snapshot from a new DROID setup and perform policy-in-the-loop rollouts inside world model. Both $\pi_{0.5}$ and Ctrl-World need to zero-shot transferr to new setups.

On our new droid setup, we tried tasks including `task_types = ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','stack']`. We provide some snapshots in `dataset_example/droid_new_setup`.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python rollout_interact_pi.py --task_type pickplace --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt} --pi_ckpt ${path to ctrl-world ckpt} --task_type ${pickplace}
```
Or you can set all parameters in `config.py` and directly run `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python rollout_interact_pi.py`. Since the official $\pi_{0.5}$ are in jax, We need to set XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 to avoild jax occupy too much space on GPU.



## Training Ctrl-World 📊

In this section, we provide detailed instructions on how to train Ctrl-World on DROID dataset. If you want to train with custun datasets, you can also follow this instructions with neccesary modifications.


### 🛸 (0) Training requirements
Our experiments are run on one node with 8 A100/H100 cards.

### 🛸 (1) Prepare dataset
(1) Since the video diffusion model are run in latent space of image encoder, we first extract the latent sapce of the video to improve training efficiency. After download the [huggingface DROID datasets](https://huggingface.co/datasets/cadene/droid_1.0.1), you can run the following command to extract latent in parrallel:
```bash
accelerate launch dataset_example/extract_latent.py --droid_hf_path ${path to droid} --droid_output_path dataset_example/droid --svd_path ${path to svd}
```
The processed data will be saved at `dataset_example/droid`. The structure of this dataset should be same as `dataset_example/droid_subset`, we already included some trajectories in it.


(2) After extract the video latent, we can prepare dataset meta information, which create a json file include all items and calculate the normalization of states and actions, which are required during training.
```bash
python dataset_meta_info/create_meta_info.py --droid_output_path ${path to processed droid data} --dataset_name droid
```

### 🛸 (2) Launch training
After prepare the datasets, you can launch training. You can first test the environment with a small subset of droid we provided in the repo:
```bash
WANDB_MODE=offline accelerate launch --main_process_port 29501 train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset
```
Then you can launch the training process with whole datasets:
```bash
accelerate launch --main_process_port 29501 train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid
```

## Acknowledgement

Video prediction policy is developed from the opensourced video foundation model [Stable-Video-Diffusion](https://github.com/Stability-AI/generative-models). We thank the authors for their efforts!


## Bibtex 
If you find our work helpful, please leave us a star and cite our paper. Thank you!
```
@article{hu2024video,
  title={Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations},
  author={Hu, Yucheng and Guo, Yanjiang and Wang, Pengchao and Chen, Xiaoyu and Wang, Yen-Jen and Zhang, Jianke and Sreenath, Koushil and Lu, Chaochao and Chen, Jianyu},
  journal={arXiv preprint arXiv:2412.14803},
  year={2024}
}