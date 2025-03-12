# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown, kk
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "kk" in data_source:
        return kk.compute_score
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra

def debug_setup():
    """Set up environment for debugging without using the command line"""
    import os
    import sys
    
    # Set required environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    
    # Create Hydra compatible command-line arguments
    debug_args = [
        "algorithm.adv_estimator=grpo",
        "data.train_files=/usr/project/xtmp/rx55/projects/long_cot/src/Logic-RL/data/kk/inter/5ppl/train.parquet",
        "data.val_files=/usr/project/xtmp/rx55/projects/long_cot/src/Logic-RL/data/kk/inter/5ppl/test.parquet",
        "data.train_batch_size=4",
        "data.val_batch_size=4",
        "data.max_prompt_length=400",
        "data.max_response_length=2500",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct",
        "actor_rollout_ref.actor.optim.lr=4e-7",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.grad_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size=2",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size=2",
        "actor_rollout_ref.ref.log_prob_micro_batch_size=2",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "trainer.critic_warmup=0",
        "trainer.logger=[wandb]",
        "trainer.project_name=GRPO_logic_KK",
        "trainer.experiment_name=Qwen-7B",
        "trainer.default_local_dir=/usr/project/xtmp/rx55/projects/long_cot/results/logic-rl_5ppl_test",
        "trainer.n_gpus_per_node=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "trainer.nnodes=1",
        "trainer.default_hdfs_dir=null",
        "trainer.save_freq=20",
        "trainer.test_freq=20",
        "trainer.total_epochs=120",
    ]
    
    # Set these as command line arguments for Hydra
    sys.argv = [sys.argv[0]] + debug_args


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    # For debugging - direct call to main_task
    if os.environ.get("VS_CODE_DEBUG", "False") == "True":
        main_task(config)
    else:
        # Normal execution with Ray
        ray.get(main_task.remote(config))


# Comment out the @ray.remote decorator temporarily
@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()

if __name__ == '__main__':
    import os
    
    # Set this environment variable to True when debugging
    os.environ["VS_CODE_DEBUG"] = "False"
    
    # Call setup function to prepare debugging environment
    if os.environ.get("VS_CODE_DEBUG", "False") == "True":
        debug_setup()
    
    # Then call the main function as usual
    main()
