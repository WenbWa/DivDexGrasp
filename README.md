# <p align="center"> UniGraspTransformer: Simplified Policy Distillation for Scalable Dexterous Robotic Grasping </p>

### <p align="center"> Microsoft Research Asia </p>

### <p align="center">[ArXiv](https://arxiv.org/abs/2412.02699) | [Website](https://dexhand.github.io/UniGraspTransformer/)

<p align="center">
  <img width="100%" src="results/overview.png"/>
</p>
    We introduce UniGraspTransformer, a universal Transformer-based network for dexterous robotic grasping that simplifies training while enhancing scalability and performance. Unlike prior methods such as UniDexGrasp++, which require complex, multi-step training pipelines, UniGraspTransformer follows a streamlined process: first, dedicated policy networks are trained for individual objects using reinforcement learning to generate successful grasp trajectories; then, these trajectories are distilled into a single, universal network. Our approach enables UniGraspTransformer to scale effectively, incorporating up to 12 self-attention blocks for handling thousands of objects with diverse poses. Additionally, it generalizes well to both idealized and real-world inputs, evaluated in state-based and vision-based settings. Notably, UniGraspTransformer generates a broader range of grasping poses for objects in various shapes and orientations, resulting in more diverse grasp strategies. Experimental results demonstrate significant improvements over state-of-the-art, UniDexGrasp++, across various object categories, achieving success rate gains of 3.5%, 7.7%, and 10.1% on seen objects, unseen objects within seen categories, and completely unseen objects, respectively, in the vision-based setting.
</p>

# Released
- [x] Code for training and testing.
- [x] Pre-trained models for isaacgym3.
- [ ] Pre-trained models for isaacgym4.

# Get Started
## Folder Structure:
```
PROJECT_BASE
    └── Logs
        └── Results
            └── results_train
            └── results_distill
            └── results_trajectory
    └── Assets
        └── datasetv4.1_posedata.npy
        └── meshdatav3_pc_feat
        └── meshdatav3_scaled
        └── meshdatav3_init
        └── textures
        └── mjcf
        └── urdf
    └── isaacgym3
    └── isaacgym4
    └── UniGraspTransformer
        └── results
        └── dexgrasp
```

## Install Environment:
Create conda env:
```
conda create -n dexgrasp python=3.8
conda activate dexgrasp
```

Install isaacgym3 as used in our paper:
```
cd PROJECT_BASE/isaacgym3/isaacgym3/python
pip install -e .
```

Install UniGraspTransformer:
```
cd PROJECT_BASE/UniGraspTransformer
pip install -e .
```

Install pytorch_kinematics:
```
cd PROJECT_BASE/UniGraspTransformer/pytorch_kinematics
pip install -e .
```

Install pytorch3d:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

# Train from scratch
## Step1: Train and Test Dedicated Policy:
```
cd PROJECT_BASE/UniGraspTransformer/dexgrasp/
```

Train&Test dedicated policy for single $nline=0 object in $Object_File=train_set_results.yaml:
```
python run_online.py --task=StateBasedGrasp --algo=ppo --seed=0 --rl_device cuda:0 \
    --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --headless \
    --object_scale_file train_set_results.yaml --start_line 0 --end_line 1
python run_online.py --task=StateBasedGrasp --algo=ppo --seed=0 --rl_device cuda:0 \
    --num_envs 1000 --max_iterations 10000 --config dedicated_policy.yaml --headless --test --test_iteration 1 \
    --object_scale_file train_set_results.yaml --start_line 0 --end_line 1
```

## Step2: Generate Trajectory Dataset:
Generate trajectories for single $nline=0 object in $Object_File=train_set_results.yaml:
```
python run_online.py --task StateBasedGrasp --algo ppo --seed 0 --rl_device cuda:0 \
    --num_envs 100 --max_iterations 10000 --config dedicated_policy.yaml --headless --test --test_iteration 10 \
    --object_scale_file train_set_results.yaml --start_line 0 --end_line 1 --save --save_train
```

## Step3: Train and Test Universal Policies:
Repeat step1 and step2 for $nline objects, like from 0 to 9, and train universal policies:
```
python run_offline.py --start 0 --finish 9 --config universal_policy_state_based.yaml --object train_set_results.yaml --device cuda:0
python run_offline.py --start 0 --finish 9 --config universal_policy_vision_based.yaml --object train_set_results.yaml --device cuda:0
```
Test state-based universal policy on $nline=0 object.
```
python run_online.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:0 \
--num_envs 100 --max_iterations 10000 --config universal_policy_state_based.yaml --headless --test --test_iteration 1 \
--model_dir distill_0000_0009 --object_scale_file train_set_results.yaml --start_line 0 --end_line 1
```
Test vision-based universal policy on $nline=0 object.
```
python run_online.py --task StateBasedGrasp --algo dagger_value --seed 0 --rl_device cuda:0 \
--num_envs 100 --max_iterations 10000 --config universal_policy_vision_based.yaml --headless --test --test_iteration 1 \
--model_dir distill_0000_0009 --object_scale_file train_set_results.yaml --start_line 0 --end_line 1
```

# Test pre-trained models (IsaacGym3)
## Test State-based Universal Policy:

## Test Vision-based Universal Policy:
