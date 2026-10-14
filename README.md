## [Project Website](https://nus-core.github.io/assets/standalone/SIGN/index.html)
[![arXiv](https://img.shields.io/badge/arXiv-2310.07473-b31b1b.svg)](https://arxiv.org/pdf/2508.12394)
[![Watch Demo](https://img.shields.io/badge/Watch-Demo-red?logo=youtube&logoColor=white)](https://nus-core.github.io/assets/standalone/SIGN/index.html)

## 1. Install
### 1.1 Install habitat-lab 
```bash
cd SIGN/

# create conda env
conda create -n sign python=3.8
conda activate sign

# install habitat-sim
conda install habitat-sim=0.2.2 withbullet headless -c conda-forge -c aihabitat

# install pytorch
pip3 install torch torchvision torchaudio

# clone habitat-lab code
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout 1f7cfbdd3debc825f1f2fd4b9e1a8d6d4bc9bfc7
pip install -e habitat-lab 
pip install -e habitat-baselines
```
### 1.2 Install other requirements
```bash
cd ..
pip install -r requirements.txt
```

## 2. Download Dataset
Dataset preparation refers to https://github.com/XinyuSun/FGPrompt

<pre>
data
├── datasets
│   └── imagenav
│       ├── gibson
│       │   ├── v1
│       │       ├── train
│       │       └── val
│       ├── hm3d
│       │   ├── v1
│       │       ├── val_easy
│       │       ├── val_hard
│       │       └── val_medium
│       └── mp3d
│           ├── v1
│               ├── test_easy
│               ├── test_hard
│               └── test_medium
└── scene_datasets
    ├── gibson
    │   ├── xx.glb
    │   └── xx.navmesh
    ├── hm3d
    │   ├── scene_id
    │       ├── xx.glb
    │       └── xx.navmesh
    └── mp3d
        ├── scene_id
            ├── xx.glb
            └── xx.navmesh


</pre>

## 3. Run ImageNav Training
```bash
python run_sim.py --overwrite \
--exp-config exp_config/ppo_imagenav_gibson.yaml,policy,reward,sensors \
--run-type train --model-dir results/imagenav/training_log
```

## 4. Run Evaluation
```bash
python run_sim.py \
--exp-config exp_config/ppo_imagenav_gibson.yaml,policy,reward,sensors,eval \
--run-type eval --model-dir ckpt \
habitat_baselines.eval_ckpt_path_dir ckpt/nav_model.pth
```

## 5. Cross-domain Evaluation
difficulty= [val_easy, val_hard, val_medium]
```bash
python run_sim.py \
--exp-config exp_config/ppo_imagenav_hm3d.yaml,policy,reward,sensors,eval \
--run-type eval --model-dir ckpt \
habitat_baselines.eval_ckpt_path_dir ckpt/nav_model.pth \
habitat_baselines.eval.split val_easy
```
difficulty= [test_easy, test_hard, test_medium]
```bash
python run_sim.py \
--exp-config exp_config/ppo_imagenav_mp3d.yaml,policy,reward,sensors,eval \
--run-type eval --model-dir ckpt \
habitat_baselines.eval_ckpt_path_dir ckpt/nav_model.pth \
habitat_baselines.eval.split test_easy
```

## 6. Citation
```
@article{yan2025sign,
  title={SIGN: Safety-Aware Image-Goal Navigation for Autonomous Drones via Reinforcement Learning},
  author={Yan, Zichen and Huang, Rui and He, Lei and Guo, Shao and Zhao, Lin},
  journal={IEEE Robotics and Automation Letters},
  volume={11},
  number={2},
  pages={1962--1969},
  year={2025},
  publisher={IEEE}
}
```
