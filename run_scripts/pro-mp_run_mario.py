from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.point_envs.point_env_2d_corner import MetaPointEnvCorner
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.meta_trainer import Trainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.policies.conv import MAMLGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder

import numpy as np
import tensorflow as tf
import os
import json
import argparse
import time

# Import for mario
from railrl.torch.metac.gcg.make_env import make_env

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):
    set_seed(config['seed'])

    
    baseline =  globals()[config['baseline']]() #instantiate baseline
    env = make_env(config['env_id'], config)
    # import pdb; pdb.set_trace()# env = globals()[config['env']]() # instantiate env
    # env = normalize(env) # apply normalize wrapper to env

    print("MARIO obs shape", env.observation_space.shape)
    policy = MAMLGaussianMLPPolicy(
            'conv',
            obs_dim=int(np.prod(env.observation_space.shape)),
            action_dim=int(np.prod(env.action_space.shape)),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )

    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )

    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
    )

    algo = ProMP(
        policy=policy,
        inner_lr=config['inner_lr'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        learning_rate=config['learning_rate'],
        num_ppo_steps=config['num_promp_steps'],
        clip_eps=config['clip_eps'],
        target_inner_step=config['target_inner_step'],
        init_inner_kl_penalty=config['init_inner_kl_penalty'],
        adaptive_inner_kl_penalty=config['adaptive_inner_kl_penalty'],
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
    )

    trainer.train()

if __name__=="__main__":
    idx = int(time.time())

    parser = argparse.ArgumentParser(description='ProMP: Proximal Meta-Policy Search')
    parser.add_argument('--config_file', type=str, default='', help='json file with run specifications')
    parser.add_argument('--dump_path', type=str, default=meta_policy_search_path + '/data/pro-mp/run_%d' % idx)

    args = parser.parse_args()


    if args.config_file: # load configuration from json file
        with open(args.config_file, 'r') as f:
            config = json.load(f)

    else: # use default config

        config = {
            'seed': 1,
            'baseline': 'LinearFeatureBaseline',
            'env_id': 'mariomultilevel',

            # sampler config
            'rollouts_per_meta_task': 2,
            'max_path_length': 10,
            'parallel': True,

            # sample processor config
            'discount': 0.99,
            'gae_lambda': 1,
            'normalize_adv': True,

            # policy config
            'hidden_sizes': (64, 64),
            'learn_std': True, # whether to learn the standard deviation of the gaussian policy

            # ProMP config
            'inner_lr': 0.1, # adaptation step size
            'learning_rate': 1e-3, # meta-policy gradient step size
            'num_promp_steps': 5, # number of ProMp steps without re-sampling
            'clip_eps': 0.3, # clipping range
            'target_inner_step': 0.01,
            'init_inner_kl_penalty': 5e-4,
            'adaptive_inner_kl_penalty': False, # whether to use an adaptive or fixed KL-penalty coefficient
            'n_itr': 1001, # number of overall training iterations
            'meta_batch_size': 40, # number of sampled meta-tasks per iterations
            'num_inner_grad_steps': 1, # number of inner / adaptation gradient steps
             
            # Mario config
            "env_kwargs" : {
                "screen_size": 20,
                "grayscale_obs": False,
                "frame_skip": 1,
                "lifelong": False,
                "max_lives": 1,
                "scramble_action_freq": 0,
                "frame_stack": 1,
                "action_stack": 0,
                "default_level": 0,
                "shuffle_env_actions": True,
                "shuffle_envs": False,
                "singletask": True
            },

            "algo_kwargs":{
                "batch_size":8,
                "adapt_batch_size": 64,
                "meta_batch_size":26,
                "test_size": 6,
                "mpc_horizon":5,
                "window_len": 200,
                "min_num_steps_before_training": 1000,
                "min_num_steps_before_adapting": 7,
                "num_expl_steps_per_train_loop": 100,
                "max_path_length":1000,
                "eval_freq": 10,
                "outer_update_steps":20,
                "inner_update_steps":4,
                "adapt_freq": 1,
                "num_adapt_steps": 5,
                "num_epochs":10000,
                "inner_lr":1e-3,
                "inner_opt_name": "SGD",
                "adapt_opt_name": "SGD",
                "adapt_inner_lr": 1e-3,
                "debug":False,
                "use_consecutive_batch": False,
                "reset_meta_model": True,
                "adapt_same_batch": False,
                "train_same_batch": True,
                "shuffle_actions": False,
                "explore_if_stuck": False,
                "shuffle_env_actions": False,
                "adapt_from_replay": False,
                "test_buffer_size": 550,
                "save_buffer": True
            },

            "trainer_kwargs":{
                "learning_rate":1e-4,
                "discount":0.99,
                "data_type": "uint8",
                "opt_name": "Adam",
                "optimizer_kwargs": {
                    "weight_decay": 0
                },
                "bayesian": False
            },

            "controller_kwargs": {
                "num_simulated_paths":500,
                "cem_steps":3
            },

            "reward_predictor_kwargs":{
                "reward_type":"categorical",
                "num_bins":41
            },
            "replay_buffer_kwargs":{
                "max_replay_buffer_size":20000
            },
            "adaptive_replay_buffer_kwargs":{
                "max_replay_buffer_size":10
            },
            "extra_args": {
                "prior_sigma_1": 0.001,
                "prior_pi": 1.0,
                "posterior_rho_init": -6
            },
            "model_kwargs": {
            	"data_type": "uint8",
                "reward_scale": 10.0,
                "bayesian": False,
                "conv_norm_type": "layer"
            },
            "log_comet": True,
            "debug": False,
            "use_gpu": True,
        }

    # configure logger
    logger.configure(dir=args.dump_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')

    # dump run configuration before starting training
    json.dump(config, open(args.dump_path + '/params.json', 'w'), cls=ClassEncoder)

    # start the actual algorithm
    main(config)
