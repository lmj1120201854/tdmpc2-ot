from time import time

import numpy as np
import torch
import pickle
import sys
from termcolor import colored
from tensordict.tensordict import TensorDict
from copy import deepcopy
from common.resnet import ResNet
from trainer.base import Trainer


class OTTrainer(Trainer):
    """Trainer class for Modem training. Assumes semi-sparse reward environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._step = 0
        self._pretrain_step = 0
        self._ep_idx = 0
        self._start_time = time()
        self.device = torch.device("cuda:0")

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self, pretrain=False):
        """Evaluate agent."""
        ep_rewards, ep_max_rewards, ep_successes, ep_seeds = [], [], [], []
        for i in range(max(1, self.cfg.eval_episodes // self.cfg.num_envs)):
            seed = np.random.randint(2**31)
            obs, done, ep_reward, ep_max_reward, t = (
                self.env.reset(seed=seed),
                torch.tensor(False),
                0,
                None,
                0,
            )
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=True)
            while not done.any():
                torch.compiler.cudagraph_mark_step_begin()
                action = (
                    self.agent.policy_action(obs, eval_mode=True)
                    if pretrain
                    else self.agent.act(obs, t0=t == 0, eval_mode=True)
                )
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_max_reward = (
                    torch.maximum(ep_max_reward, reward)
                    if ep_max_reward is not None
                    else reward
                )
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            assert (
                done.all()
            ), "Vectorized environments must reset all environments at once."
            ep_rewards.append(ep_reward)
            ep_max_rewards.append(ep_max_reward)
            ep_successes.append(info["success"].float().mean())
            ep_seeds.append(seed)

        if self.cfg.save_video:
            if pretrain:
                self.logger.video.save(
                    "pretrain/iteration",
                    self._pretrain_step,
                    key="videos/pretrain_video",
                )
            else:
                self.logger.video.save("eval/step", self._step)

        return dict(
            episode_reward=torch.cat(ep_rewards).mean(),
            episode_max_reward=torch.cat(ep_max_rewards).max(),
            episode_success=torch.stack(ep_successes).mean(),
            best_seed=(
                ep_seeds[torch.argmax(torch.stack(ep_rewards).mean(dim=1)).item()]
                if pretrain
                else None
            ),
        )

    def to_td(self, obs, action=None, reward=None, device="cpu"):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=())
        else:
            obs = obs.unsqueeze(0)
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan")).repeat(self.cfg.num_envs)
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(
                1,
                self.cfg.num_envs,
            ),
        )
        return td.to(torch.device(device))

    def pretrain(self):
        """Pretrains agent policy with demonstration data"""
        demo_buffer = self.buffer._offline_buffer
        n_iterations = self.cfg.pretrain.n_epochs
        self.cfg.pretrain.eval_freq = n_iterations // 25
        start_time = time()
        best_model, best_score = deepcopy(self.agent.bc_model.state_dict()), 0

        print(
            colored(
                f"Policy pretraining: {n_iterations} iterations", "red", attrs=["bold"]
            )
        )

        self.agent.bc_model.train()
        for self._pretrain_step in range(n_iterations):
            metrics = self.agent.init_bc(demo_buffer)

            if self._pretrain_step % self.cfg.pretrain.eval_freq == 0:
                eval_metrics = self.eval(pretrain=True)
                eval_metrics.update({"iteration": self._pretrain_step})
                self.logger.log(eval_metrics, category="pretrain")

                if eval_metrics["episode_reward"] > best_score:
                    best_model = deepcopy(self.agent.bc_model.state_dict())
                    best_score = eval_metrics["episode_reward"]
                    best_seed = eval_metrics["best_seed"]

            if self._pretrain_step % self.cfg.pretrain.log_freq == 0:
                metrics.update(
                    {
                        "iteration": self._pretrain_step,
                        "total_time": time() - start_time,
                    }
                )
                self.logger.log(metrics, category="pretrain")

        if best_score == 0:
            best_model = deepcopy(self.agent.bc_model.state_dict())
            best_seed = eval_metrics["best_seed"]

        self.agent.model.eval()
        self.agent.model.load_state_dict(best_model)
        self.agent.bc_model.load_state_dict(best_model)
        self.seed_scheduler.start(init_seed=best_seed, max_seeds=1e4)


    def ot_init_demos(self):
        with open(f"/home/burson/data/tdmpc_ot/expert_demos/{self.cfg.task}.pkl", "rb") as f:
            data = pickle.load(f)
            next_observations = data[0]['next_observations'][:-1]
            rgb_base_tensors = [td['rgb_base'] for td in next_observations]
            # rgb_hand_tensors = [td['rgb_hand'] for td in next_observations]
            # torch.Size([100, 3, 128, 128])
            final_rgb_base_tensor = torch.stack(rgb_base_tensors)
            # final_rgb_hand_tensor = torch.stack(rgb_hand_tensors)

            # rgb_obs = torch.cat((final_rgb_base_tensor, final_rgb_hand_tensor), dim=1)
            # print(f'rgb_obs: shape:{rgb_obs.shape}')

            cost_encoder = ResNet().to(self.device)
            _ = cost_encoder.eval()
            
            input_tensor = torch.as_tensor(final_rgb_base_tensor).float().to(self.device)
            # input_tensor: shape:torch.Size([100, 3, 128, 128]) ----> demos: torch.Size([100, 32768])
            with torch.no_grad():
                demos = cost_encoder(input_tensor)  

            # print(f'demos: shape:{demos.shape}')
            # sys.exit()

            self.agent.init_demos(cost_encoder, demos)  



    def train(self):
        """Train agent"""

        # ot_reward demos init
        if self.cfg.ot_reward_shaping:
            self.ot_init_demos()

        # Policy pretraining
        # if self.cfg.get("policy_pretraining", False):
        #     self.pretrain()

        # Start interactive training
        print(colored("\nReplay buffer seeding", "yellow", attrs=["bold"]))
        log_metrics, train_metrics, done, eval_next = {}, {}, torch.tensor(True), False
        pixel_obs, global_episode = [], 0

        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Save Agent periodically
            if self._step % self.cfg.save_freq == 0 and self._step > 0:
                print("Saving agent checkpoint...")
                self.logger.save_agent(self.agent, identifier=f"agent_{self._step}")

            # Reset environment
            if done.any():
                assert (
                    done.all()
                ), "Vectorized environments must reset all environments at once."
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                ''''
                            TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([101, 16, 7]), device=cpu, dtype=torch.float32, is_shared=False),
                    obs: TensorDict(
                        fields={
                            rgb_base: Tensor(shape=torch.Size([101, 16, 3, 128, 128]), device=cpu, dtype=torch.uint8, is_shared=False),
                            rgb_hand: Tensor(shape=torch.Size([101, 16, 3, 128, 128]), device=cpu, dtype=torch.uint8, is_shared=False),
                            state: Tensor(shape=torch.Size([101, 16, 25]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([101, 16]),
                        device=cpu,
                        is_shared=False),
                    reward: Tensor(shape=torch.Size([101, 16]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([101, 16]),
                device=cpu,
                is_shared=False)
                            
                '''

                # ot reward shaping
                if self.cfg.ot_reward_shaping and global_episode > 0:

                    tds = torch.cat(self._tds)
                    
                    pixel_obs = torch.stack(pixel_obs, dim=1) # pixel_obs: torch.Size([16, 100, 6, 128, 128])
                    
                    ot_rewards, cost_min, cost_max = self.agent.ot_rewarder(pixel_obs)
                    
                    assert cost_min >= 0
                    

                    # use first episode to normalize rewards
                    if global_episode == self.cfg.num_envs:
                        
                        # [16,100]-->[100,16]
                        ot_rewards_tensor = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in ot_rewards]) # [16,100]

                        ot_rewards_tensor_sum = ot_rewards_tensor.sum(dim=1)
                        ot_rewards_mean_sum = abs(ot_rewards_tensor_sum.mean())

                        if ot_rewards_mean_sum < 1e-6:
                            ot_rewards_mean_sum = 1.0

                        self.cfg.sinkhorn_rew_scale = self.cfg.auto_rew_scale_factor / ot_rewards_mean_sum
                        print(f"OT reward scaling factor set to: {self.cfg.sinkhorn_rew_scale:.3f}")
                        ot_rewards, cost_min, cost_max = self.agent.ot_rewarder(pixel_obs)


                    ot_rewards_tensor = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in ot_rewards])
                    ot_rewards_reshaped = ot_rewards_tensor.transpose(0, 1).to(tds.device) # [100,16]
                    tds["reward"][1:] += ot_rewards_reshaped

                    # add to replay buffer with ot rewards
                    self._ep_idx = self.buffer.add(tds)
                    train_metrics.update(
                        episode_reward=np.nansum(tds["reward"], axis=0).mean(),
                        episode_max_reward=np.nanmax(tds["reward"], axis=0).max(),
                        episode_success=info["success"].float().nanmean(),
                    )

                    train_metrics.update(self.common_metrics())
                    # self.logger.log(train_metrics, "train")
                    self.seed_scheduler.step(train_metrics["episode_success"].item())

                # if self._step > 0:
                #     tds = torch.cat(self._tds)
                #     self._ep_idx = self.buffer.add(tds)

                #     train_metrics.update(
                #         episode_reward=np.nansum(tds["reward"], axis=0).mean(),
                #         episode_max_reward=np.nanmax(tds["reward"], axis=0).max(),
                #         episode_success=info["success"].float().nanmean(),
                #     )

                #     train_metrics.update(self.common_metrics())
                #     # self.logger.log(train_metrics, "train")
                #     self.seed_scheduler.step(train_metrics["episode_success"].item())

                obs = self.env.reset(seed=self.seed_scheduler.sample())
                self._tds = [self.to_td(obs, device="cpu")]

                pixel_obs = []
                global_episode += self.cfg.num_envs

            # Collect experience
            if self._step > self.cfg.seed_steps:
                self._alpha = (
                    max(0, self.cfg.max_bc_steps - self._step) / self.cfg.max_bc_steps
                )
                if np.random.random() < self._alpha and self.cfg.get(
                    "policy_pretraining", False
                ):
                    action = self.agent.policy_action(obs, eval_mode=True)
                else:
                    action = self.agent.act(obs, t0=len(self._tds) == 1)
            elif self.cfg.get("policy_pretraining", False):
                action = self.agent.policy_action(obs, eval_mode=True)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)

            # concat_obs = torch.cat((obs['rgb_base'], obs['rgb_hand']), dim=1)
            # pixel_obs.append(concat_obs)

            pixel_obs.append(obs['rgb_base'])
            
            self._tds.append(self.to_td(obs, action, reward, device="cpu"))


            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = max(
                        1, int(self.cfg.seed_steps / self.cfg.steps_per_update)
                    )
                    print(colored("\nTraining MoDem Agent", "green", attrs=["bold"]))
                    print(
                        f"Pretraining agent with {num_updates} update steps on seed data..."
                    )
                else:
                    num_updates = max(
                        1, int(self.cfg.num_envs / self.cfg.steps_per_update)
                    )
                for _ in range(num_updates):
                    agent_train_metrics = self.agent.update(
                        self.buffer, action_penalty=self.cfg.action_penalty
                    )
                # train_metrics.update(agent_train_metrics)
                log_metrics.update(self.common_metrics())
                log_metrics.update(agent_train_metrics)
                if self._step % self.cfg.train_log_freq == 0:
                    self.logger.log(log_metrics, "train")
                    log_metrics = {}

            self._step += self.cfg.num_envs

        self.logger.finish(self.agent)
