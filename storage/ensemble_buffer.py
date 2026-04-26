import torch
from copy import deepcopy

from storage.buffer import Buffer
from termcolor import colored


class EnsembleBuffer(Buffer):
    """
    Ensemble of an offline dataloader and an online replay buffer.
    """

    def __init__(self, cfg):
        _cfg1, _cfg2 = deepcopy(cfg), deepcopy(cfg)
        _cfg1.batch_size = int(cfg.batch_size * (1 - _cfg1.demo_sampling_ratio))
        _cfg2.batch_size = int(cfg.batch_size - _cfg1.batch_size)
        super().__init__(_cfg1)

        from storage.data_utils import load_dataset_as_td

        demo_dataset = load_dataset_as_td(
            _cfg2.demo_path,
            num_traj=_cfg2.n_demos,
            success_only=_cfg2.demo_success_only,
        )
        cfg.n_demos = len(demo_dataset)
        _cfg2.buffer_size = (len(demo_dataset) + int(len(demo_dataset) == 1)) * len(demo_dataset[0]) + 100000  # Offline buffer is not dynamically alocated

        cfg.batch_size = _cfg1.batch_size + _cfg2.batch_size
        # NOTE: Make sure demonstrations contain same type of rewards as online environment!
        self._offline_buffer = Buffer(_cfg2)
        if len(demo_dataset) == 1:
            self._offline_buffer.add(
                demo_dataset[0].unsqueeze(1)
            )  # TODO: This is a patch to solve bug in Replay Buffer when only having 1 trajectory (we duplicate first demo)
        for _td in demo_dataset:
            self._offline_buffer.add(_td.unsqueeze(1))
        print(
            colored(
                f"Filled demo buffer with {self._offline_buffer.num_eps} trajectories",
                "green",
            )
        )


    def add_success_data(self, td, success_mask):
        assert td.shape[1] == success_mask.shape[0], "Mismatch between td batch size and success_mask"

        b_size = td.shape[1]
        td = td.permute(1, 0)
        
        # b_size: 16, success_mask: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float64)

        for i in range(b_size):
            if success_mask[i]:
                ep_td = td[i]
                ep_td["episode"] = torch.ones_like(ep_td["reward"], dtype=torch.int64) * torch.arange(
                    self._offline_buffer._num_eps, self._offline_buffer._num_eps + 1
                )
                ep_td["reward"] = torch.round(ep_td["reward"]).to(torch.int32)
                self._offline_buffer._buffer.extend(ep_td)
                if ep_td.shape[0] > self._offline_buffer._max_length:
                    self._offline_buffer._max_length = ep_td.shape[0]
                self._offline_buffer._num_eps += 1


    def sample(self, return_td=False):
        """Sample a batch of subsequences from the two buffers.

        Returns a 5-tuple (obs, action, reward, ot_reward, task) with a unified
        batch dimension `offline_bs + online_bs`.

        NOTE: offline demos don't carry OT rewards; we pad those positions with
        NaN so the agent update can build a mask and exclude them from
        ot_reward loss while still keeping tensor shapes aligned.
        """
        if return_td:
            raise NotImplementedError(
                f"TensorDict return not implemented for EnsembleBuffer"
            )
        if self._offline_buffer.batch_size <= 0:
            obs, action, reward, task = super().sample()
            ot_reward = torch.full_like(reward, float("nan"))
            return obs, action, reward, ot_reward, task

        obs0, action0, reward0, task0 = self._offline_buffer.sample()
        obs1, action1, reward1, ot_reward1, task1 = super().ot_sample()

        # Pad offline segment with NaN to keep batch dims aligned.
        ot_reward0 = torch.full_like(reward0, float("nan"))

        return (
            torch.cat([obs0, obs1], dim=1),
            torch.cat([action0, action1], dim=1),
            torch.cat([reward0, reward1], dim=1),
            torch.cat([ot_reward0, ot_reward1], dim=1),
            torch.cat([task0, task1], dim=0) if task0 and task1 else None,
        )

    def sample_for_disc(self, batch_size: int):
        if self._offline_buffer.batch_size > 0:
            td0 = self._offline_buffer.sample_single(return_td=True)
            td1 = super().sample_single(return_td=True)

            # Concat + Shuffle
            tds = torch.cat([td0, td1], dim=0)
        else:
            tds = super().sample_single(return_td=True)

        obs_return = []
        for stage_index in range(self.cfg.n_stages):
            # Success indices
            success_indices = (
                (tds["reward"] == stage_index) * (tds["stage"] > stage_index)
            ).nonzero(as_tuple=True)[0]
            fail_indices = (
                (tds["reward"] == stage_index) * (tds["stage"] <= stage_index)
            ).nonzero(as_tuple=True)[0]

            # Cut at minimum to avoid data imbalance
            success_indices = success_indices[
                : min(len(success_indices), len(fail_indices))
            ]
            fail_indices = fail_indices[: min(len(success_indices), len(fail_indices))]

            # Extract observations for those indices
            obs_return.append(
                {
                    "success_data": tds["obs"][success_indices],
                    "fail_data": tds["obs"][fail_indices],
                }
            )

        return obs_return