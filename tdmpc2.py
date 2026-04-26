import torch
import torch.nn.functional as F
import numpy as np
import sys
from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.resnet import ResNet
from common.math import cosine_distance, mask_optimal_transport_plan
from tensordict import TensorDict

class RNDNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        # Target Network
        self.target = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        # Predictor Network
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
      
        for param in self.target.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # x: latent state z
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        return pred_feat, target_feat

class TDMPC2(torch.nn.Module):
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        self.model = WorldModel(cfg).to(self.device)
        self.bc_model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._ot_reward.parameters()},
                {"params": self.model._Qs.parameters()},
                {
                    "params": (
                        self.model._task_emb.parameters() if self.cfg.multitask else []
                    )
                },
            ],
            lr=self.cfg.lr,
            capturable=True,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True
        )
        self.bc_optim = torch.optim.Adam(self.bc_model.parameters(), lr=self.cfg.lr)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2 * int(
            cfg.action_dim >= 20
        )  # Heuristic for large action spaces
        self.discount = (
            torch.tensor(
                [self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
                device="cuda:0",
            )
            if self.cfg.multitask
            else self._get_discount(cfg.episode_length)
        )
        self._prev_mean = torch.nn.Buffer(
            torch.zeros(
                self.cfg.num_envs,
                self.cfg.horizon,
                self.cfg.action_dim,
                device=self.device,
            )
        )

        # OT Sinkhorn 的时间对角 mask —— 使用通用 `episode_length`，而不是硬编码
        # 某个具体后端的 max_episode_steps，避免不同 env 时字段缺失。
        _mask_len = int(
            getattr(
                self.cfg,
                "episode_length",
                self.cfg.maniskill.max_episode_steps,
            )
        )
        self.mask = np.triu(
            np.tril(np.ones((_mask_len, _mask_len)), k=self.cfg.mask_k),
            k=-self.cfg.mask_k,
        )

        # === RND 模块 ===
        self.use_rnd = bool(getattr(self.cfg, "use_rnd", True))
        self.rnd_model = RNDNetwork(cfg.latent_dim).to(self.device)
        self.rnd_optim = torch.optim.Adam(
            self.rnd_model.predictor.parameters(),
            lr=self.cfg.lr,
        )

        # RND intrinsic-reward scale: 纯函数式 cosine 退火
        self._rnd_scale_start = float(getattr(self.cfg, "rnd_scale_start", 0.1))
        self._rnd_scale_end = float(getattr(self.cfg, "rnd_scale_end", 0.01))
        self._rnd_decay_steps = max(
            1,
            int(getattr(
                self.cfg,
                "rnd_decay_steps",
                int(self.cfg.steps / max(1, self.cfg.steps_per_update)),
            )),
        )
        self._rnd_update_step = 0
        self.rnd_scale = self._rnd_scale_start

        # 统一的各信号权重（在 planning / value 中使用）
        self.ot_planning_lambda = float(getattr(self.cfg, "ot_planning_lambda", 1.0))
        self.rnd_coef = float(getattr(self.cfg, "rnd_coef", 1.0))

        if cfg.compile:
            print("compiling - tdmpc update")
            self._update = torch.compile(self._update, mode="reduce-overhead")
            print("compiling - bc update")
            self._init_bc = torch.compile(self._init_bc, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if False:  # self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
                episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
                float: Discount factor for the task.
        """
        if self.cfg.discount_hardcoded != 0:
            return self.cfg.discount_hardcoded
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
        )

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
                fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
                fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    def init_bc(self, buffer):
        """
        Initialize policy using a behavior cloning objective.
        """
        obs, action, reward, task = buffer.sample(return_td=False)
        torch.compiler.cudagraph_mark_step_begin()
        return self._init_bc(obs, action, reward, task)

    def _init_bc(self, obs, action, rew, task):
        self.bc_optim.zero_grad(set_to_none=True)
        a = self.bc_model.pi(self.bc_model.encode(obs[:-1], task), task)[0]
        loss = F.mse_loss(a, action, reduce=True)
        loss.backward()
        self.bc_optim.step()
        self.model.load_state_dict(self.bc_model.state_dict())

        metrics = (
            TensorDict(
                {
                    "bc_loss": loss,
                }
            )
            .detach()
            .mean()
        )
        return metrics


    # ------------ OT reward --------------#
    def get_context_observations(self, observations):
        L = len(observations)
        idx0 = np.arange(L)

        context_observations = [observations[idx0]]
        for i in range(1, self.cfg.context_num):
            idx_i = (idx0 + i).clip(0, L-1)
            context_observations.append(observations[idx_i])

        context_observations = torch.stack(context_observations)
        return context_observations
    
    def init_demos(self, cost_encoder, demos):
        # [100, 32768] ----> [3, 100, 32768]
        self.cost_encoder = cost_encoder
        self.demos = self.get_context_observations(demos)

    def ot_rewarder(self, observations):
        """
        Compute OT-based shaping reward for a batch of online rollouts against a
        (set of) expert demonstration(s).

        Args:
            observations: Tensor of shape [num_envs, T, C, H, W] (uint8 or float)
        Returns:
            ot_rewards_list: list of np.ndarray[T], length = num_envs
            cost_min, cost_max: min/max entries of the last cost matrix (for logging)
        """
        obs = torch.as_tensor(observations).float().to(self.device)
        num_envs = obs.shape[0]
        T = obs.shape[1]

        # Sanity check: OT 需要在线轨迹长度与 demo 长度、mask 尺寸一致
        demo_T = self.demos.shape[1]
        mask_T = self.mask.shape[0]
        if not (T == demo_T == mask_T):
            raise ValueError(
                f"OT rewarder length mismatch: online T={T}, demo T={demo_T}, "
                f"mask T={mask_T}. Check `episode_length`, demo trajectory length "
                f"and `mask` construction."
            )

        ot_rewards_list = []
        cost_min, cost_max = float("inf"), float("-inf")

        with torch.no_grad():
            # 将所有 env 的轨迹拼在一起做一次 ResNet 前向，减少 Python 开销
            # obs: [N, T, C, H, W] -> [N*T, C, H, W]
            flat_obs = obs.reshape(num_envs * T, *obs.shape[2:])
            flat_feat = self.cost_encoder(flat_obs)  # [N*T, D]

            # Sanity check: online feat dim 需要和 demo feat dim 一致
            demo_D = self.demos.shape[-1]
            if flat_feat.shape[-1] != demo_D:
                raise ValueError(
                    f"OT rewarder feature-dim mismatch: online={flat_feat.shape[-1]} vs "
                    f"demo={demo_D}. 常见原因是在线 rgb 图像分辨率与 demo 不一致，"
                    f"导致 ResNet flatten 后的维度不同。请确认 env `image_size`。"
                )

            feat = flat_feat.reshape(num_envs, T, -1)  # [N, T, D]

            for env_idx in range(num_envs):
                encoded_trajectory = feat[env_idx]   # [T, D]
                context_obs = self.get_context_observations(encoded_trajectory)  # [ctx, T, D]

                cost_matrix = 0
                for i in range(self.cfg.context_num):
                    cost_matrix = cost_matrix + cosine_distance(context_obs[i], self.demos[i])
                cost_matrix = cost_matrix / self.cfg.context_num

                transport_plan = mask_optimal_transport_plan(
                    context_obs[0],
                    self.demos[0],
                    cost_matrix,
                    self.mask,
                    niter=self.cfg.niter,
                    epsilon=self.cfg.ot_epsilon,
                )

                ot_rewards = -self.cfg.sinkhorn_rew_scale * torch.diag(
                    torch.mm(transport_plan, cost_matrix.T)
                ).detach().cpu().numpy()

                ot_rewards_list.append(ot_rewards)
                cost_min = min(cost_min, cost_matrix.min().item())
                cost_max = max(cost_max, cost_matrix.max().item())

        return ot_rewards_list, cost_min, cost_max
            
    # ------------ OT reward --------------#


    @torch.no_grad()
    def policy_action(self, obs, eval_mode=False, task=None):
        """
        Select an action by only sampling from policy.

        Args:
                obs (torch.Tensor): Observation from the environment.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (int): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.bc_model.encode(obs, task)
        a = self.bc_model.pi(z, task)[int(not eval_mode)]
        return a.cpu()

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.

        Args:
                obs (torch.Tensor): Observation from the environment.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (int): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            a = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
        else:
            z = self.model.encode(obs, task)
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(
                self.model.reward(z, actions[:, t], task), self.cfg
            )

            ot_reward = math.two_hot_inv(
                self.model.ot_reward(z, task), self.cfg
            )

            # 一次 dynamics step，避免重复计算
            next_z = self.model.next(z, actions[:, t], task)

            # Intrinsic reward (RND error) —— 对 rollout 的 next_z 使用
            if self.use_rnd and self.rnd_scale > 0:
                pred_feat, target_feat = self.rnd_model(next_z)
                rnd_error = (pred_feat - target_feat).pow(2).sum(dim=-1, keepdim=True)
                intrinsic_reward = self.rnd_coef * self.rnd_scale * rnd_error
            else:
                intrinsic_reward = 0.0

            # 统一加权叠加：任务奖励 + OT 奖励 + RND 探索奖励
            step_reward = (
                reward
                + self.ot_planning_lambda * ot_reward
                + intrinsic_reward
            )
            G = G + discount * step_reward

            z = next_z
            if self.cfg.multitask:
                _task_tensor = task if isinstance(task, torch.Tensor) else torch.tensor(task)
                discount_update = self.discount[_task_tensor]
            else:
                discount_update = self.discount
            discount = discount * discount_update
        return G + discount * self.model.Q(
            z, self.model.pi(z, task)[1], task, return_type="avg"
        )

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Args:
                z (torch.Tensor): Latent state from which to plan. Shape (b_size, z_dim)
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        b_size = obs.shape[0]
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                b_size,
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )
            _z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[:, t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[:, t], task)
            pi_actions[:, -1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
        mean = torch.zeros(
            b_size, self.cfg.horizon, self.cfg.action_dim, device=self.device
        )
        std = torch.full(
            (b_size, self.cfg.horizon, self.cfg.action_dim),
            self.cfg.max_std,
            dtype=torch.float,
            device=self.device,
        )
        if not t0:
            mean[:, :-1] = self._prev_mean[:, 1:]
        actions = torch.empty(
            b_size,
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        if self.cfg.num_pi_trajs > 0:
            actions[:, :, : self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            r = torch.randn(
                b_size,
                self.cfg.horizon,
                self.cfg.num_samples - self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=std.device,
            )
            actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, :, self.cfg.num_pi_trajs :] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(
                value.squeeze(2), self.cfg.num_elites, dim=1
            ).indices
            elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(2))
            elite_actions = torch.gather(
                actions,
                2,
                elite_idxs.unsqueeze(1)
                .unsqueeze(3)
                .expand(-1, self.cfg.horizon, -1, self.cfg.action_dim),
            )

            # Update parameters
            max_value = elite_value.max(1).values
            score = torch.exp(
                self.cfg.temperature * (elite_value - max_value.unsqueeze(1))
            )
            score = score / score.sum(1, keepdim=True)
            mean = (score.unsqueeze(1) * elite_actions).sum(dim=2) / (
                score.sum(1, keepdim=True) + 1e-9
            )
            std = (
                (score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2).sum(
                    dim=2
                )
                / (score.sum(1, keepdim=True) + 1e-9)
            ).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action sequence with probability `score`
        rand_idx = math.gumbel_softmax_sample(
            score.squeeze(-1), dim=1
        )  # gumbel_softmax_sample is compatible with cuda graphs
        actions = torch.stack(
            [elite_actions[i, :, rand_idx[i]] for i in range(rand_idx.shape[0])], dim=0
        )
        a, std = actions[:, 0], std[:, 0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
                zs (torch.Tensor): Sequence of latent states.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                float: Loss of the policy update.
        """
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type="avg", detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        return pi_loss.detach(), pi_grad_norm

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
                next_z (torch.Tensor): Latent state at the following time step.
                reward (torch.Tensor): Reward at the current time step.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, task)[1]
        discount = (
            self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        )
        return reward + discount * self.model.Q(
            next_z, pi, task, return_type="min", target=True
        )

    def _update(
        self, obs, action, reward, ot_reward, task=None, modify_reward=None, action_penalty=False
    ):
        # Compute targets
        with torch.no_grad():
            # Get latent states
            next_z = self.model.encode(obs[1:], task)

            # Modify reward if necessary
            if modify_reward:
                reward = modify_reward(next_z, reward)

            if action_penalty:
                reward -= torch.linalg.norm(action, ord=2, dim=-1, keepdim=True).pow(
                    2
                ) / (action.shape[-1] * 5)

            # Compute td_targets
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.model.train()
        self.rnd_model.train()

        # Latent rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action, task)
            consistency_loss = (
                consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
            )
            zs[t + 1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type="all")
        reward_preds = self.model.reward(_zs, action, task)
        # NOTE: 修复 bug —— 原实现 `_zs[:, -128:, :]` 会按 batch 维度截取最后 128 个样本,
        #   导致与 ot_reward target 的 shape 不一致。OT reward 只依赖 latent z，
        #   因此直接在完整 rollout 上预测即可。
        ot_reward_preds = self.model.ot_reward(_zs, task)

        # OT reward 可能在 EnsembleBuffer 的 offline 段为 NaN (demo 没有 ot_reward),
        # 构造 per-sample mask, 只对 online 段样本计算 ot_reward_loss.
        # ot_reward shape: [horizon, batch, 1]
        ot_valid_mask = (~torch.isnan(ot_reward)).float()              # [T, B, 1]
        ot_reward_clean = torch.where(
            ot_valid_mask.bool(), ot_reward, torch.zeros_like(ot_reward)
        )

        # Compute losses
        reward_loss, ot_reward_loss, value_loss = 0, 0, 0
        for t, (
            rew_pred_unbind,
            rew_unbind,
            ot_rew_pred_unbind,
            ot_rew_unbind,
            ot_mask_unbind,
            td_targets_unbind,
            qs_unbind,
        ) in enumerate(
            zip(
                reward_preds.unbind(0),
                reward.unbind(0),
                ot_reward_preds.unbind(0),
                ot_reward_clean.unbind(0),
                ot_valid_mask.unbind(0),
                td_targets.unbind(0),
                qs.unbind(1),
            )
        ):
            reward_loss = (
                reward_loss
                + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean()
                * self.cfg.rho**t
            )

            # masked mean: sum(loss * mask) / max(sum(mask), 1)
            ot_ce = math.soft_ce(ot_rew_pred_unbind, ot_rew_unbind, self.cfg)
            denom = ot_mask_unbind.sum().clamp_min(1.0)
            ot_reward_loss = (
                ot_reward_loss
                + (ot_ce * ot_mask_unbind).sum() / denom * self.cfg.rho**t
            )

            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = (
                    value_loss
                    + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean()
                    * self.cfg.rho**t
                )

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        ot_reward_loss = ot_reward_loss / self.cfg.horizon
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)

        # === RND Loss ===
        # 为保证 "训练分布 == 使用分布"：训练 RND 时同时覆盖
        #   (a) encoder 真实 latent (next_z)，防止 predictor 过拟合 dynamics 误差
        #   (b) dynamics 展开得到的 zs[1:]（即 planning 中实际评估的对象）
        # 全部 detach，只更新 predictor，不回传到 encoder / dynamics。
        rnd_real = next_z.detach().reshape(-1, self.cfg.latent_dim)
        rnd_rollout = zs[1:].detach().reshape(-1, self.cfg.latent_dim)
        rnd_input = torch.cat([rnd_real, rnd_rollout], dim=0)
        pred_feat, target_feat = self.rnd_model(rnd_input)
        rnd_loss = F.mse_loss(pred_feat, target_feat)

        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.ot_reward_coef * ot_reward_loss
            + self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update RND Predictor
        self.rnd_optim.zero_grad(set_to_none=True)
        rnd_loss.backward()
        self.rnd_optim.step()

        # Update policy
        pi_loss, pi_grad_norm = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        with torch.no_grad():
            intr_reward_mean = (pred_feat - target_feat).pow(2).sum(dim=-1).mean()
        return (
            TensorDict(
                {
                    "consistency_loss": consistency_loss,
                    "reward_loss": reward_loss,
                    "ot_reward_loss": ot_reward_loss,
                    "value_loss": value_loss,
                    "pi_loss": pi_loss,
                    "total_loss": total_loss,
                    "grad_norm": grad_norm,
                    "pi_grad_norm": pi_grad_norm,
                    "pi_scale": self.scale.value,
                    "rnd_loss": rnd_loss,
                    "rnd_scale": torch.tensor(self.rnd_scale, device=self.device),
                    "intr_reward_mean": intr_reward_mean,
                }
            )
            .detach()
            .mean()
        )

    def _cosine_rnd_scale(self, step):
        """Pure-Python cosine annealing for RND intrinsic-reward scale."""
        import math as _math
        step = min(step, self._rnd_decay_steps)
        cos_out = 0.5 * (1 + _math.cos(_math.pi * step / self._rnd_decay_steps))
        return self._rnd_scale_end + (self._rnd_scale_start - self._rnd_scale_end) * cos_out

    def update(self, buffer, **kwargs):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
                buffer (common.buffer.Buffer): Replay buffer.

        Returns:
                dict: Dictionary of training statistics.
        """
        sampled = buffer.sample()
        # 兼容两种 buffer 协议：
        #   4 元组 (obs, action, reward, task) —— 原始 TD-MPC2 Buffer / Demo3Buffer
        #   5 元组 (obs, action, reward, ot_reward, task) —— EnsembleBuffer (OT 模式)
        if len(sampled) == 5:
            obs, action, reward, ot_reward, task = sampled
        elif len(sampled) == 4:
            obs, action, reward, task = sampled
            # 无 OT 奖励通道的 buffer：用 float32 NaN 占位，_update 里会 mask 掉.
            # 不用 full_like(reward) 是因为 reward 可能是 int32 dtype.
            ot_reward = torch.full(
                reward.shape, float("nan"), dtype=torch.float32, device=reward.device
            )
        else:
            raise ValueError(f"Unexpected buffer.sample() return length: {len(sampled)}")

        if task is not None:
            kwargs["task"] = task

        # 更新 RND 调度
        self._rnd_update_step += 1
        self.rnd_scale = self._cosine_rnd_scale(self._rnd_update_step)

        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, ot_reward, **kwargs)