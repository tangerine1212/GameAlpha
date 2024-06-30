import ray
import gymnasium as gym
import logging
import argparse
import rlcard
import random
from pathlib import Path
from matplotlib import pyplot as plt
#from utils import ProgressBar,ma_sample,get_winrate_and_weight,register_restore_weight_trainer
#from custom_model import CustomFullyConnectedNetwork,KerasBatchNormModel,BatchNormModel,OriginalNetwork
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import pickle
import os
import pandas as pd

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.train import report
from ray import tune

from agi.nl_holdem_env import NlHoldemEnvWithOpponent
from agi.nl_holdem_net import NlHoldemNet
from agi.nl_holdem_lg_net import NlHoldemLgNet
from agi.TrinalPolicy import TrinalPpoPolicy
from agi.league import League
from util import get_winrate_and_weight


import ray
import ray.rllib
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

import warnings
warnings.filterwarnings("ignore")

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate




parser = argparse.ArgumentParser()
parser.add_argument('--conf',  type=str)
parser.add_argument('--gap',  type=int, default=500)
parser.add_argument('--sp',  type=float, default=0.0)
parser.add_argument('--exg_oppo_prob',  type=float, default=0.01)
parser.add_argument('--upwin',  type=float, default=1)
parser.add_argument('--kbest',  type=int, default=5)
parser.add_argument('--league_tracker_n',  type=float, default=1000)
parser.add_argument('--last_num',  type=int, default=100000)
parser.add_argument('--rwd_update_ratio',  type=float, default=1.0)
parser.add_argument('--restore',  type=str,default=None)
parser.add_argument('--output_dir',  type=str,default="league/history_agents")
parser.add_argument('--mode', type=str, default="local")
parser.add_argument('--experiment_name', default='run_trial_1', type=str)  # please change a new name
args = parser.parse_args()

if args.mode == "local":
    ray.init()
else:
    raise RuntimeError("unknown mode: {}".format(args.mode))





league = League.remote(
    n=args.league_tracker_n,
    last_num=args.last_num,
    kbest=args.kbest,
    output_dir=args.output_dir,
)





class TrinalCallsback(DefaultCallbacks):
    @static_vars(league=league)
    def on_episode_end(self,*,worker,base_env,policies,episode,env_index,**kwargs,):
        envs = base_env
        default_policy = policies["default_policy"]
        for env in envs.vector_env.envs:
            if env.is_done:
                # 1. 更新结果到league
                rewards = episode.agent_rewards
                chips = env.env.get_perfect_information()["chips"]
                for i in range(len(chips)):
                    if i == env.our_pid:
                        lowest = float(-chips[i])/2
                    else:
                        highest = float(chips[i])/2
                rewards[("agent0", episode.policy_for("agent0"))] = np.clip(rewards[("agent0", episode.policy_for("agent0"))], lowest, highest)
                last_reward = np.clip(env.last_reward, lowest, highest)
                pid = env.oppo_name
                
                if np.random.random() < args.rwd_update_ratio:
                    if pid == "self":
                        ray.get(TrinalCallsback.on_episode_end.league.update_result.remote(None,last_reward,selfplay=True))
                    else:
                        ray.get(TrinalCallsback.on_episode_end.league.update_result.remote(pid,last_reward,selfplay=False))

                # 2. 更新对手权重
                
                # 以0.2的概率self play
                if np.random.random() < args.exg_oppo_prob:
                    if np.random.random() < args.sp:
                        p_weights = default_policy.get_weights()
                        weight = {}
                        for k,v in p_weights.items():
                            k = k.replace("default_policy","oppo_policy")
                            weight[k] = v
                        env.oppo_name = "self"
                        env.oppo_policy.set_weights(weight)
                    else:
                        pid,weight = ray.get(league.select_opponent.remote())
                        env.oppo_name = pid
                        env.oppo_policy.set_weights(weight)
    @static_vars(league=league)
    def on_episode_start(self,*,worker,base_env,policies,episode,env_index,**kwargs,) -> None:
        envs = base_env
        default_policy = policies["default_policy"]
        
        # 如果league 没有第一个权重，那么使用当前policy中的权重当作第一个
        if not ray.get(TrinalCallsback.on_episode_start.league.initized.remote()):
            p_weights = default_policy.get_weights()
            weight = {}
            for k,v in p_weights.items():
                k = k.replace("default_policy","oppo_policy")
                weight[k] = v
            ray.get(TrinalCallsback.on_episode_start.league.initize_if_possible.remote(weight))
        for env in envs.vector_env.envs:
            if env.oppo_name is None:
                pid,weight = ray.get(TrinalCallsback.on_episode_start.league.select_opponent.remote())
                env.oppo_name = pid
                env.oppo_policy.set_weights(weight)

    @static_vars(league=league,count=0)
    def on_train_result(self,*,algorithm,result,**kwargs,):
        winrates_pd = ray.get(TrinalCallsback.on_train_result.league.get_statics_table.remote())
        winrates_pd.to_csv("winrates.csv",header=False,index=False)
        
        table_t = winrates_pd.T
        table_t["mbb/h"] = np.asarray(table_t["winrate"] / 2.0 * 1000.0,np.int32)
        result['winrates'] = table_t.T
        TrinalCallsback.on_train_result.count += 1
        
        gap = args.gap
        if ray.get(TrinalCallsback.on_train_result.league.winrate_all_match.remote(args.upwin))  \
            or TrinalCallsback.on_train_result.count % gap == gap - 1:
            policy = algorithm.get_policy("default_policy")
            p_weights = policy.get_weights()
            weight = {}
            for k,v in p_weights.items():
                k = k.replace("default_policy","oppo_policy")
                weight[k] = v 
            ray.get(TrinalCallsback.on_train_result.league.add_weight.remote(weight))
            if not os.path.exists("weights"):
                os.makedirs("weights")
            with open('output_weight.pkl','wb') as whdl:
                pickle.dump(weight,whdl)
            with open('weights/output_weight_{}.pkl'.format(TrinalCallsback.on_train_result.count),'wb') as whdl:
                pickle.dump(weight,whdl)
    
conf = eval(open("confs//nl_holdem.py").read().strip())
ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
ModelCatalog.register_custom_model('NlHoldemLgNet', NlHoldemLgNet)
register_env("NlHoldemEnvWithOpponent", lambda config: NlHoldemEnvWithOpponent(
        conf
))

def get_train(weight):
    if weight is None:
        pweight = None
    else:
        pweight = {}
        for k,v in weight.items():
            k = k.replace("oppo_policy","default_policy")
            pweight[k] = v
            
    def train_fn_load(config):
        PPOconf = PPOConfig("PPO")
        print("conf created")
        # print(PPOconf.to_dict())
        #config["default_policy_class"] = PolicySpec(TrinalPpoPolicy)
        PPOconf.update_from_dict(config_dict=config)
        algo = PPOconf.build(env="NlHoldemEnvWithOpponent")
        print("LOAD: after init, before load")

        if pweight is not None:
            algo.workers.local_worker().get_policy().set_weights(pweight)
            algo.workers.sync_weights()
        print("LOAD: before train, after load")
        while True:
            result = algo.train()
            print(result['info']['learner']["default_policy"]['learner_stats'])
        algo.stop()

    return train_fn_load
        
if args.restore is not None:
    get_winrate_and_weight(args.restore,league)
    pid = ray.get(league.get_latest_policy_id.remote())
    print("latest pid: {}".format(pid))
    weight = ray.get(league.get_weight.remote(pid))
    #register_restore_weight_trainer(weight)
    train_func = get_train(weight)
else:
    train_func = get_train(None)

tune_config = {
    'max_sample_requests_in_flight_per_worker': 1,
    'num_data_loader_buffers': 4,
    'callbacks':TrinalCallsback,
}
conf["default_policy_class"] = PolicySpec(TrinalPpoPolicy,None,None,None)
conf["multiagent"] = {"policies":{"default_policy":PolicySpec(TrinalPpoPolicy,None,None,None)}}

tune_config.update(conf)
tune.run(
    train_func,
    config=tune_config,
    stop={
        'timesteps_total': 10000000000,
    },
    # resources_per_trial=ImpalaTrainer.default_resource_request,
    # resources_per_trial=algori.default_resource_request(tune_config),
    local_dir=str(Path("./log/").absolute()),
    resources_per_trial=tune.PlacementGroupFactory(
    [{'GPU': 4.0, 'CPU': 2.0}]+[{'CPU':8}]*10
    )
)
