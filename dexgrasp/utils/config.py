# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import yaml
import glob

from isaacgym import gymapi
from isaacgym import gymutil

import torch
import random
import numpy as np
from utils.general_utils import *


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception("Unrecognized task!")

def warn_algorithm_name():
    raise Exception("Unrecognized algorithm!\nAlgorithm should be one of: [ppo, happo, hatrpo, mappo]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def retrieve_cfg(args, use_rlg_config=False):
    if args.task == "ShadowHandGrasp":
        return os.path.join(args.logdir, "shadow_hand_grasp/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_grasp.yaml"
    elif args.task == "ShadowHandRandomLoadVision":
        return os.path.join(args.logdir, "shadow_hand_random_load_vision/{}/{}".format(args.algo,args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_random_load_vision.yaml"
    elif args.task == "StateBasedGrasp":
        return os.path.join(args.logdir, "state_based_grasp/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/state_based_grasp.yaml"
    elif args.task == "InspireBasedGrasp":
        return os.path.join(args.logdir, "state_based_grasp/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/inspire_based_grasp.yaml"
    else:
        warn_task_name()



def load_cfg(args, use_rlg_config=False):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task
    cfg["wandb"] = args.wandb
    cfg["headless"] = args.headless
    # render camera images
    cfg["render_each_view"] = args.render_each_view
    cfg["render_hyper_view"] = args.render_hyper_view
    cfg["render_point_clouds"] = args.render_point_clouds
    cfg["sample_point_clouds"] = args.sample_point_clouds
    # test_iteration
    cfg['test'] = args.test
    cfg['config'] = args.config
    cfg['container'] = args.container
    cfg["test_epoch"] = args.test_epoch
    cfg["test_iteration"] = args.test_iteration
    # object_scale_dict
    cfg["object_scale_file"] = args.object_scale_file
    cfg["object_scale_list"] = args.object_scale_list
    cfg["start_line"], cfg["end_line"], cfg["group"] = args.start_line, args.end_line, args.group
    cfg["shuffle_dict"], cfg["shuffle_env"] = args.shuffle_dict, args.shuffle_env

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = args.logdir
    if use_rlg_config:

        # Set deterministic mode
        if args.torch_deterministic:
            cfg_train["params"]["torch_deterministic"] = True

        exp_name = cfg_train["params"]["config"]['name']

        if args.experiment != 'Base':
            if args.metadata:
                exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

                if cfg["task"]["randomize"]:
                    exp_name += "_DR"
            else:
                exp_name = args.experiment

        # Override config name
        cfg_train["params"]["config"]['name'] = exp_name

        if args.resume > 0:
            cfg_train["params"]["load_checkpoint"] = True

        if args.checkpoint != "Base":
            cfg_train["params"]["load_path"] = args.checkpoint

        # Set maximum number of training iterations (epochs)
        if args.max_iterations > 0:
            cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

        cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

        seed = cfg_train["params"].get("seed", -1)
        if args.seed is not None:
            seed = args.seed
        cfg["seed"] = seed
        cfg_train["params"]["seed"] = seed

        cfg["args"] = args
    else:

        # Set deterministic mode
        if args.torch_deterministic:
            cfg_train["torch_deterministic"] = True

        # Override seed if passed on the command line
        if args.seed is not None:
            cfg_train["seed"] = args.seed

        log_id = args.logdir
        if args.experiment != 'Base':
            if args.metadata:
                log_id = args.logdir + "_{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])
                if cfg["task"]["randomize"]:
                    log_id += "_DR"
            else:
                log_id = args.logdir + "_{}".format(args.experiment)

        logdir = os.path.realpath(log_id)
        # os.makedirs(logdir, exist_ok=True)
    
    ocd_tag = args.ocd_tag

    if args.backbone_type != "":
        cfg_train["policy"]["backbone_type"] = args.backbone_type
    else:
        cfg_train["policy"]["backbone_type"] = None
        
    
    cfg_train["policy"]["freeze_backbone"] = args.freeze_backbone
        
    if len(ocd_tag) > 0:
        ocd_group_file = 'eval_results/object_code_dict_groups.yaml'
        with open(ocd_group_file, 'r') as f:
            ocd_groups = yaml.safe_load(f)
        ocd = {}
        for tag in ocd_groups['run_tags'][ocd_tag]:
            ocd.update(ocd_groups['ocd_groups'][tag])
        cfg['env']['object_code_dict'] = ocd
    
    cfg['logdir'] = logdir
    cfg['algo'] = args.algo

    # modify PPO params
    if args.algo == 'ppo':
        # double_update_step
        if args.config['Modes']['double_update_step'] or args.config['Modes']['double_update_half_iteration_step']:
            cfg_train['learn']['nsteps'] = 16
    # modify DaggerValue params
    elif args.algo == 'dagger_value':
        # double_update_step
        if args.config['Distills']['double_update_step'] or args.config['Distills']['double_update_half_iteration_step']:
            cfg_train['learn']['nsteps'] = 16
    
    # num_observation
    if 'num_observation' in args.config['Weights']: cfg["env"]["numObservations"] = args.config['Weights']['num_observation']
    return cfg, cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False, use_rlg_config=False):
    custom_parameters = [
        {"name": "--default", "action": "store_true", "default": False,
            "help": "Default UniDexGrasp++"},
        {"name": "--save", "action": "store_true", "default": False,
            "help": "Save observation-action data"},
        {"name": "--save_train", "action": "store_true", "default": False,
            "help": "Save observation-action train data"},
        {"name": "--save_render", "action": "store_true", "default": False,
            "help": "Save observation-action train data with rendered point clouds"},
        {"name": "--init", "action": "store_true", "default": False,
            "help": "Save init object state"},
        {"name": "--wandb", "action": "store_true", "default": False,
            "help": "Init wandb or not"},
        {"name": "--container", "action": "store_true", "default": False,
            "help": "Run in container"},
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--config", "type": str, "default": "dexgrasp/train.yaml",
            "help": "Training and Testing Config"},
        {"name": "--test_iteration", "type": int, "default": 10,
            "help": "Number of times to run test for envs"},
        {"name": "--test_epoch", "type": int, "default": 0,
            "help": "Test trained epochs"},
        {"name": "--object_scale_file", "type": str, "default": "Default",
            "help": "Path to the saved object_scale yaml"},
        {"name": "--object_scale_list", "type": str, "default": "",
            "help": "Line list within object_scale_file to be processed"},
        {"name": "--start_line", "type": int, "default": None,
            "help": "Start line in object_scale yaml"},
        {"name": "--end_line", "type": int, "default": None,
            "help": "End line in object_scale yaml"},
        {"name": "--group", "type": int, "default": None,
            "help": "Group number in object_scale yaml"},
        {"name": "--block", "type": int, "default": 0,
            "help": "Block number in distill entire yaml"},
        {"name": "--is_expert", "action": "store_true", "default": False,
            "help": "Expert mode in distill entire yaml"},
        {"name": "--shuffle_dict", "action": "store_true", "default": False,
            "help": "Shuffle load object_scale or not"},
        {"name": "--shuffle_env", "action": "store_true", "default": False,
            "help": "Shuffle load env object or not"},
        {"name": "--vision", "action": "store_true", "default": False,
            "help": "Use vision backbone"},
        {"name": "--freeze_backbone", "action": "store_true", "default": False,
            "help": "freeze_backbone during training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--render_each_view", "action": "store_true", "default": False,
            "help": "Render each view camera"},
        {"name": "--render_hyper_view", "action": "store_true", "default": False,
            "help": "Render hyper view camera"},
        {"name": "--render_point_clouds", "action": "store_true", "default": False,
            "help": "Render object point clouds for visualization"},
        {"name": "--sample_point_clouds", "action": "store_true", "default": False,
            "help": "Sample point clouds from depth images"},
        {"name": "--horovod", "action": "store_true", "default": False,
            "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library"},
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg_train", "type": str,
            "default": "Base"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--steps_num", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--algo", "type": str, "default": "happo",
            "help": "Choose an algorithm"},
        {"name": "--model_dir", "type": str, "default": "",
            "help": "Choose a model dir"},
        {"name": "--expert_model_dir", "type": str, "default": "",
            "help": "Choose a expert model dir"},
        {"name": "--datatype", "type": str, "default": "random",
            "help": "Choose an ffline datatype"},
         {'name': '--ocd_tag', 'type': str, 'default': ''},
         {'name': '--backbone_type', 'type': str, 'default': ''}
        ]

    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # allignment with examples
    # args.device_id = args.compute_device_id
    args.device_id = int(args.rl_device.split(':')[-1])
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'
    # set test, train, play mode
    if args.test: args.play, args.train = args.test, False
    elif args.play: args.train = False
    else: args.train = True 
    
    # locate log, train:cfg/args.algo/config.yaml, env:cfg/args.task
    logdir, cfg_train, cfg_env = retrieve_cfg(args, use_rlg_config)
    
    # load config settings
    args.config = load_yaml(os.path.realpath('cfg/train/{}'.format(args.config)))
    # save observation-action-value-success trajectory
    args.config['Save'] = args.save
    args.config['Save_Train'] = args.save_train
    args.config['Save_Render'] = args.save_render
    # save init object state trajectory
    args.config['Init'] = args.init
    # TODO: locate Logs_Dir for locate and cluster
    Logs_Dir = LOG_DIR if os.path.exists(LOG_DIR) else '/mnt/blob/Desktop/Logs'
    # Logs_Dir = '../../Logs' if os.path.exists(os.path.realpath('../../Logs')) else '/mnt/blob/Desktop/Logs'
    if args.container: Logs_Dir = '../../Container/Desktop/Logs'
    # locate Save_Base: Logs/config['Infos']['save_name']
    args.config['Save_Base'] = os.path.realpath('{}/{}'.format(Logs_Dir, args.config['Infos']['save_name']))
    # locate logdir for ppo train: Logs/save_name/results_train/nline_seed0
    if args.algo == 'ppo': 
        args.logdir = os.path.realpath('{}/results_train/{:04d}'.format(args.config['Save_Base'], args.start_line))
        # locate logdir for ppo test: Logs/save_name/results_test_*/nline_seed0
        if args.object_scale_file == 'test_set_seen_cat_results.yaml':
            args.logdir = os.path.realpath('{}/results_test_seen/{:04d}'.format(args.config['Save_Base'], args.start_line))
        if args.object_scale_file == 'test_set_unseen_cat_results.yaml':
            args.logdir = os.path.realpath('{}/results_test_unseen/{:04d}'.format(args.config['Save_Base'], args.start_line))
        
    # locate logdir for dagger_value train
    elif args.algo == 'dagger_value':
        # train with random lines: Logs/save_name/results_distill/random/save_name/distill_start_end_seed0
        if args.group is None: args.logdir = os.path.realpath('{}/results_distill/random/{}/distill_{:04d}_{:04d}'.format(args.config['Save_Base'], args.config['Distills']['save_name'], args.start_line, args.end_line-1))
        # train with group lines: Logs/save_name/results_distill/group/save_name/distill_group_group_seed0
        else: args.logdir = os.path.realpath('{}/results_distill/group/{}/distill_group_{:04d}'.format(args.config['Save_Base'], args.config['Distills']['save_name'], args.group))
        # test seen and unseen set objects: Logs/save_name/results_distill/random/save_name/results_test_*_seed0
        if args.object_scale_file == 'test_set_seen_cat_results.yaml':
            args.logdir = '{}/results_test_seen'.format(os.path.dirname(args.logdir))
        if args.object_scale_file == 'test_set_unseen_cat_results.yaml':
            args.logdir = '{}/results_test_unseen'.format(os.path.dirname(args.logdir))
        # distill entire objects with blocks
        args.config['Distills']['block'] = args.block
        args.config['Distills']['is_expert'] = args.is_expert

    # use custom parameters if provided by user
    if use_rlg_config == False:
        if args.horovod:  print("Distributed multi-gpu training with Horovod is not supported by rl-pytorch. Use rl_games for distributed training.")
        if args.steps_num != -1: print("Setting number of simulation steps per iteration from command line is not supported by rl-pytorch.")
        if args.minibatch_size != -1: print("Setting minibatch size from command line is not supported by rl-pytorch.")
        if args.checkpoint != "Base": raise ValueError("--checkpoint is not supported by rl-pytorch. Please use --resume <iteration number>")
    if args.logdir == "logs/": args.logdir = logdir
    if args.cfg_train == "Base": args.cfg_train = cfg_train
    if args.cfg_env == "Base": args.cfg_env = cfg_env

    # format object_scale_list
    args.object_scale_list = None if args.object_scale_list == "" else [int(string) for string in args.object_scale_list.split(' ')]
    # decide train_flag
    train_flag = True
    if args.object_scale_list is not None:
        for line_index in range(args.start_line, args.end_line):
            if line_index not in args.object_scale_list: train_flag = False

    # ppo mode: train or test with model_dir
    if args.algo == 'ppo':
        # double_iteration_step
        if args.config['Modes']['double_iteration_step']: args.max_iterations *= 2
        # double_update_step with half_iteration_step
        if args.config['Modes']['double_update_half_iteration_step']: args.max_iterations *= 0.5

        # locate ppo model_dir: Logs/save_name/results_train/nline_seed0/model_10000.pt
        model_dir = os.path.realpath('{}_seed0/model_{}.pt'.format(args.logdir, args.max_iterations))
        # locate ppo model_dir: Logs/save_name/results_test_*/model_best.pt
        if args.object_scale_file == 'test_set_seen_cat_results.yaml':
            model_dir = os.path.realpath('{}/results_test_seen/model_best.pt'.format(args.config['Save_Base']))
        if args.object_scale_file == 'test_set_unseen_cat_results.yaml':
            model_dir = os.path.realpath('{}/results_test_unseen/model_best.pt'.format(args.config['Save_Base']))
        
        # train mode: skip training already trained objects
        if not args.test and os.path.exists(model_dir):
            train_flag = False
            print('======== Find Existing Trained Model! ========')

        # test mode: test the last existing model
        if args.test and args.model_dir == "":
            args.model_dir = model_dir

    # dagger_value mode: train or test with model_dir
    elif args.algo == 'dagger_value':
        # double_iteration_step
        if args.config['Distills']['double_iteration_step']: args.max_iterations *= 2
        # double_update_step with half_iteration_step
        if args.config['Distills']['double_update_half_iteration_step']: args.max_iterations *= 0.5

        # locate dagger_value model_dir: Logs/save_name/results_distill/random/save_name/distill_start_end_seed0/model_10000.pt
        model_dir = os.path.realpath('{}_seed0/model_{}.pt'.format(args.logdir, args.max_iterations))
        # locate dagger_value model_dir: Logs/save_name/results_distill/random/save_name/model_best.pt
        if args.object_scale_file == 'test_set_seen_cat_results.yaml':
            model_dir = '{}/model_best.pt'.format(os.path.dirname(args.logdir))
        if args.object_scale_file == 'test_set_unseen_cat_results.yaml':
            model_dir = '{}/model_best.pt'.format(os.path.dirname(args.logdir))
        
        # train mode: skip training already trained objects
        if not args.test and (os.path.exists(model_dir) or os.path.exists(os.path.realpath('{}_seed0/model_best.pt'.format(args.logdir)))):
            train_flag = False
            print('======== Find Existing Trained Model! ========')
        
        # test mode with indicated mode_dir: distill_0000_0000
        if args.test:
            # locate distill_folder: Logs/save_name/results_distill/random_or_group/save_name/model_dir_seed0
            distill_folder = args.logdir.replace(args.logdir.split('/')[-1], '{}_seed0'.format(args.model_dir))
            # update model_dir: Logs/save_name/results_distill/random_or_group/save_name/model_dir_seed0
            args.model_dir = os.path.join('{}/model_{}.pt'.format(distill_folder, args.max_iterations))
            # update model_dir as model_best.pt for offline mlp distillation
            if os.path.exists('{}/model_best.pt'.format(distill_folder)): args.model_dir = '{}/model_best.pt'.format(distill_folder)
            if args.test_epoch != 0 and os.path.exists('{}/model_{}.pt'.format(distill_folder, args.test_epoch)): args.model_dir = '{}/model_{}.pt'.format(distill_folder, args.test_epoch)
            # update random logdir: Logs/save_name/results_distill/random/save_name/model_dir_seed0/nline_seed0
            if args.group is None: args.logdir = os.path.join('{}/{:04d}'.format(distill_folder, args.start_line))
            # update group logdir: Logs/save_name/results_distill/group/save_name/model_dir_seed0/nline_seed0
            else:
                # load object_scale_groups
                object_scale_groups = load_yaml(os.path.realpath('../results/configs/{}'.format(args.object_scale_file)))
                if args.start_line >= len(object_scale_groups[args.group]['object_line']): return args, False
                args.logdir = os.path.join('{}/{:04d}'.format(distill_folder, object_scale_groups[args.group]['object_line'][args.start_line]))
            
            # test seen and unseen set objects: Logs/save_name/results_distill/random/save_name/model_best.pt
            if args.object_scale_file == 'test_set_seen_cat_results.yaml':
                args.logdir = os.path.realpath('{}/results_distill/random/{}/results_test_seen_seed0/{:04d}'.format(args.config['Save_Base'], args.config['Distills']['save_name'], args.start_line))
                args.model_dir = '{}/model_best.pt'.format(os.path.dirname(os.path.dirname(args.logdir)))
            if args.object_scale_file == 'test_set_unseen_cat_results.yaml':
                args.logdir = os.path.realpath('{}/results_distill/random/{}/results_test_unseen_seed0/{:04d}'.format(args.config['Save_Base'], args.config['Distills']['save_name'], args.start_line))
                args.model_dir = '{}/model_best.pt'.format(os.path.dirname(os.path.dirname(args.logdir)))

    # train, test default UniDexGrasp++
    if args.default: args.model_dir = '{}/dexgrasp/example_model/state_based_model.pt'.format(BASE_DIR)
    return args, train_flag
