import gymnasium as gym
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor, Preprocessor
from ray.rllib.env import BaseEnv
torch, nn = try_import_torch()
import copy
from io import StringIO 
import sys
import rlcard
from rlcard.utils import set_seed
import random

color2ind = dict(zip("CDHS",[0,1,2,3]))
rank2ind = dict(zip("23456789TJQKA",[0,1,2,3,4,5,6,7,8,9,10,11,12]))

class NlHoldemEnvWrapper():
    def __init__(self,policy_config,weights=None):
        self.policy_config = policy_config
        seed = random.randint(0,1000000)
        self.env = rlcard.make(
            'no-limit-holdem',
            config={
                'seed': seed,
            }
        )
        set_seed(seed)
        self.action_num = 5
        
        
        space = {
                'card_info': gym.spaces.Box(low=-1024, high=1024, shape=(6,4,13)),
                'action_info': gym.spaces.Box(low=-256, high=256, shape=(25, 4,self.action_num)),
                'extra_info': gym.spaces.Box(low=-256, high=256, shape=(4,)),
                'legal_moves': gym.spaces.Box(low=-1,high=1,shape=(5,)),
            }
        
        self.observation_space = gym.spaces.Dict(space)
        self.action_space = gym.spaces.Discrete(self.action_num)
        
    @property
    def unwrapped(self):
        return None

    def _get_observation(self,obs):
        card_info = np.zeros([6,4,13], np.uint8)
        action_info = np.zeros([25,4,self.action_num],np.uint8) # 25 channel
        extra_info = np.zeros([4],np.uint8) # 25 channel
        legal_actions_info = np.zeros([self.action_num],np.uint8) # 25 channel
        
        hold_card = obs[0]["raw_obs"]["hand"]
        public_card = obs[0]["raw_obs"]["public_cards"]
        current_legal_actions = [i.value for i in obs[0]["raw_obs"]["legal_actions"]]
        
        for ind in current_legal_actions:
            legal_actions_info[ind] = 1
        
        flop_card = public_card[:3]
        turn_card = public_card[3:4]
        river_card = public_card[4:5]
        
        for one_card in hold_card:
            card_info[0][color2ind[one_card[0]]][rank2ind[one_card[1]]] = 1
            
        for one_card in flop_card:
            card_info[1][color2ind[one_card[0]]][rank2ind[one_card[1]]] = 1
            
        for one_card in turn_card:
            card_info[2][color2ind[one_card[0]]][rank2ind[one_card[1]]] = 1
            
        for one_card in river_card:
            card_info[3][color2ind[one_card[0]]][rank2ind[one_card[1]]] = 1
            
        for one_card in public_card:
            card_info[4][color2ind[one_card[0]]][rank2ind[one_card[1]]] = 1
            
        for one_card in public_card + hold_card:
            card_info[5][color2ind[one_card[0]]][rank2ind[one_card[1]]] = 1
            
        
        for ind_round,one_history in enumerate(self.history):
            for ind_h,(player_id,action_id,legal_actions) in enumerate(one_history[:6]):
                action_info[ind_round * 6 + ind_h,player_id,action_id] = 1
                action_info[ind_round * 6 + ind_h,2,action_id] = 1
                
                for la_ind in legal_actions:
                    action_info[ind_round * 6 + ind_h,3,la_ind] = 1
                    
        action_info[:,:,-1] = self.my_agent()
        
        extra_info[0] = obs[0]["raw_obs"]["stakes"][0]
        extra_info[1] = obs[0]["raw_obs"]["stakes"][1]
        extra_info[2] = obs[0]["raw_obs"]["all_chips"][0]
        extra_info[3] = obs[0]["raw_obs"]["all_chips"][1]
        return {
            "card_info": card_info,
            "action_info": action_info,
            "legal_moves": legal_actions_info,
            "extra_info": extra_info,
        }
    
    def _log_action(self,action_ind):
        self.history[
            self.last_obs[0]["raw_obs"]["stage"].value
        ].append([
            self.last_obs[0]["raw_obs"]["current_player"],
            action_ind,
            [x.value for x in self.last_obs[0]["raw_obs"]["legal_actions"]]
        ])
    
    def my_agent(self):
        return self.env.get_player_id()
    
    def convert(self,reward):
        return float(reward)
        
    def step(self, action):
        self._log_action(action)
        obs = self.env.step(action)
        self.last_obs = obs
        obs = self._get_observation(obs)
        
        done = False
        reward = [0,0]
        info = {}
        if self.env.game.is_over():
            done = True
            reward = list(self.env.get_payoffs())
            
        return obs,reward,done,info

    def reset(self,*,seed=None,options=None):
        self.history = [[],[],[],[]]
        obs = self.env.reset()
        self.last_obs = obs
        return self._get_observation(obs)
    
    def legal_moves(self):
        pass
    

class NlHoldemEnvWithOpponent(NlHoldemEnvWrapper, gym.Env):
    count = 0
    def __init__(self,policy_config,weights=None,opponent="nn"):
        super(NlHoldemEnvWithOpponent, self).__init__(policy_config,weights)
        self.opponent = opponent
        self.rwd_ratio = policy_config["env_config"]["custom_options"].get("rwd_ratio",1)
        self.is_done = False
        if self.opponent == "nn":
            self.oppo_name = None
            self.oppo_preprocessor = ModelCatalog.get_preprocessor_for_space(self.observation_space, policy_config.get("model"))
            policy_config["num_gpus"] = 0
            policy_config["exploration"] = None
            policy_config["_disable_initialize_loss_from_dummy_batch"] = True
            policy_config["exploration_config"] = {"type":"StochasticSampling"}
            self.oppo_policy = PPOTorchPolicy(
                observation_space=self.oppo_preprocessor.observation_space,
                action_space=self.action_space,
                config=policy_config
            )
            if weights is not None:
                import pickle
                with open(weights,'rb') as fhdl:
                    weights = pickle.load(fhdl)
                self.oppo_policy.set_weights(weights)
        
    def _opponent_step(self,obs):
        if self.opponent == "random":
            rwd = [0,0]
            done = False
            info = {}
            while self.my_agent() != self.our_pid:
                legal_moves = obs["legal_moves"]
                action_ind = np.random.choice(np.where(legal_moves)[0])
                obs,rwd,done,info = super(NlHoldemEnvWithOpponent, self).step(action_ind)
                if done:
                    break
            return obs,rwd,done,info
        elif self.opponent == "nn":
            rwd = [0,0]
            done = False
            info = {}
            while self.my_agent() != self.our_pid:
                observation = self.oppo_preprocessor.transform(obs)
                action_ind = self.oppo_policy.compute_actions([observation])[0][0]
                obs,rwd,done,info = super(NlHoldemEnvWithOpponent, self).step(action_ind)
                if done:
                    break
            return obs,rwd,done,info
        else:
            raise        
        
    def reset(self,*,seed=None,options=None):
        self.last_reward = 0
        self.is_done = False
        self.our_pid = random.randint(0,1)
        
        
        obs = super(NlHoldemEnvWithOpponent, self).reset()
        while True:
            obs,rwd,done,info = self._opponent_step(obs)
            if not done:
                return obs, info
            else:
                obs = super(NlHoldemEnvWithOpponent, self).reset()
            
    def step(self,action):
        NlHoldemEnvWithOpponent.count += 1
        #print(f"step applied, counts to reset:{NlHoldemEnvWithOpponent.count}")
        obs,reward,done,info = super(NlHoldemEnvWithOpponent, self).step(action)
        reward = [i * self.rwd_ratio for i in reward]
        #print(f"step_legal_moves:{obs['legal_moves']}")
        if done:
            self.is_done = True
            self.last_reward = reward[self.our_pid]
            return obs,reward[self.our_pid],done,False,info
        else:
            obs,reward,done,info = self._opponent_step(obs)
            reward = [i * self.rwd_ratio for i in reward]
            if done:
                self.is_done = True
                self.last_reward = reward[self.our_pid]
            return obs,reward[self.our_pid],done,False,info
        