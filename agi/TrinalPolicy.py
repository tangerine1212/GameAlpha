from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.policy.view_requirement import ViewRequirement
from typing import List, Type, Union
torch, nn = try_import_torch()


class TrinalPpoPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.theta1 = 1.5
        super().__init__(observation_space, action_space, config)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid
        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        # prev_action_dist = dist_class(
        #     train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        # )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        # if self.config["kl_coeff"] > 0.0:
        #     action_kl = prev_action_dist.kl(curr_action_dist)
        #     mean_kl_loss = reduce_mean_valid(action_kl)
        #     # TODO smorad: should we do anything besides warn? Could discard KL term
        #     # for this update
        #     warn_if_infinite_kl_divergence(self, mean_kl_loss)
        # else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(logp_ratio, torch.clamp(logp_ratio,1 - self.config["clip_param"],1 + self.config["clip_param"]), torch.full_like(logp_ratio, self.theta1)),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        return total_loss
    
    # def _get_default_view_requirements(self):
    #     """Returns a default ViewRequirements dict.

    #     Note: This is the base/maximum requirement dict, from which later
    #     some requirements will be subtracted again automatically to streamline
    #     data collection, batch creation, and data transfer.

    #     Returns:
    #         ViewReqDict: The default view requirements dict.
    #     """

    #     # Default view requirements (equal to those that we would use before
    #     # the trajectory view API was introduced).
    #     return {
    #         SampleBatch.OBS: ViewRequirement(space=self.observation_space),
    #         SampleBatch.NEXT_OBS: ViewRequirement(
    #             data_col=SampleBatch.OBS,
    #             shift=1,
    #             space=self.observation_space,
    #             used_for_compute_actions=False,
    #         ),
    #         SampleBatch.ACTIONS: ViewRequirement(
    #             space=self.action_space, used_for_compute_actions=False
    #         ),
    #         # For backward compatibility with custom Models that don't specify
    #         # these explicitly (will be removed by Policy if not used).
    #         SampleBatch.PREV_ACTIONS: ViewRequirement(
    #             data_col=SampleBatch.ACTIONS, shift=-1, space=self.action_space
    #         ),
    #         SampleBatch.REWARDS: ViewRequirement(),
    #         # For backward compatibility with custom Models that don't specify
    #         # these explicitly (will be removed by Policy if not used).
    #         SampleBatch.PREV_REWARDS: ViewRequirement(
    #             data_col=SampleBatch.REWARDS, shift=-1
    #         ),
    #         SampleBatch.TERMINATEDS: ViewRequirement(),
    #         SampleBatch.TRUNCATEDS: ViewRequirement(),
    #         SampleBatch.INFOS: ViewRequirement(used_for_compute_actions=False),
    #         SampleBatch.EPS_ID: ViewRequirement(),
    #         SampleBatch.UNROLL_ID: ViewRequirement(),
    #         SampleBatch.AGENT_INDEX: ViewRequirement(),
    #         SampleBatch.T: ViewRequirement(),
    #         SampleBatch.VF_PREDS: ViewRequirement(),
    #         SampleBatch.ACTION_LOGP: ViewRequirement(),
    #     }