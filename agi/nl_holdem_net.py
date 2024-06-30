import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN



class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,padding = 1, stride=[1,1]) -> None:
        super(BasicBlock, self).__init__()
        padding = []
        for i in stride:
            if i == 1:
                padding.append(1)
            else:
                padding.append((3,7))
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding[0],bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding[1],bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), 
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NlHoldemNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,name): 
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,name)
        nn.Module.__init__(self)
        self.in_channels = 16
        self.card_conv1 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16)
        )
        # card_conv2
        self.card_conv2 = self._make_layer(BasicBlock,32,[[1,1],[1,1]],[[1,1],[1,1]])
        # card_conv3
        self.card_conv3 = self._make_layer(BasicBlock,64,[[(3,7),1], [1,1]],[[2,1],[1,1]])
        
        self.in_channels = 16
        self.action_conv1 = nn.Sequential(
            nn.Conv2d(25,16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16)
        )
        # action_conv2
        self.action_conv2 = self._make_layer(BasicBlock,32,[[1,1], [1,1]],[[1,1],[1,1]])
        # action_conv3
        self.action_conv3 = self._make_layer(BasicBlock,64,[[1,1], [1,1]],[[1,1],[1,1]])

        self.fc1 = nn.Sequential(nn.Linear(64*13*4+64*4*5+16, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.action_fc = nn.Linear(64, 5)
        self.value_fc = nn.Linear(64, 1)
        self.extra_fc = nn.Sequential(nn.Linear(4, 16), nn.ReLU())
        self.theta2 = 0
        self.theta3 = 0
        
    def _make_layer(self, block, out_channels, padding, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, padding, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
            
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        card_info = obs["card_info"].float()
        action_info = obs["action_info"].float()
        extra_info = obs["extra_info"].float()
        
        card_info = self.card_conv1(card_info)
        card_info = self.card_conv2(card_info)
        card_info = self.card_conv3(card_info)
        card_info = torch.flatten(card_info, start_dim=-3, end_dim=-1)
        
        action_info = self.action_conv1(action_info)
        action_info = self.action_conv2(action_info)
        action_info = self.action_conv3(action_info)
        action_info = torch.flatten(action_info, start_dim=-3, end_dim=-1)
        
        extra_info = torch.squeeze(self.extra_fc(extra_info), 1)
        
        feature_fuse = torch.concat((card_info, action_info, extra_info), -1)
        feature_fuse = self.fc1(feature_fuse)
        feature_fuse = self.fc2(feature_fuse)
        feature_fuse = self.fc3(feature_fuse)
        
        action = self.action_fc(feature_fuse)
        self._value_out = self.value_fc(feature_fuse)
        action_mask = obs["legal_moves"].float()
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        #assert False
        return action+inf_mask, state
    def value_function(self): 
        return torch.reshape(self._value_out, [-1])


