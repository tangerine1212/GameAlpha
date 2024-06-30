{
    'env': 'NlHoldemEnvWithOpponent',
    #'default_policy_class': 'TrinalPpoPolicy',
    'framework': 'torch',
    'sample_batch_size': 50,
    'train_batch_size': 1000,
    'num_workers': 10,
    "_enable_learner_api":False,
    "_enable_rl_module_api":False,
    "disable_env_checking":True,
    "max_episode_steps":25,
    'num_cpus_per_learner_worker': 1,
    "exploration_config": {"type": "StochasticSampling"},
    "log_lever":"error",
    "reuse_actors":True,
    #'num_envs_per_worker': 1,
    #'broadcast_interval': 5,
    #'max_sample_requests_in_flight_per_worker': 1,
    #'num_data_loader_buffers': 4,
    'num_gpus': 2,
    'gamma': 1,
    'entropy_coeff': 1e-1,
    'lr': 3e-4,
    'model':{
       'custom_model': 'NlHoldemNet',
       'max_seq_len': 20,
    },
    "env_config":{
        'custom_options': {
            'weight':"default",
            "cut":[[0,12],[13,25],[26,38],[39,51],[52,53],[53,54]],
            'epsilon': 0.15,
            'tracker_n': 1000,
            'conut_bb_rather_than_winrate': 2,
            'use_history': True,
            'use_cardnum': True,
            'history_len': 20,
        },
    }
}