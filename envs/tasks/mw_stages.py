import gymnasium as gym
import numpy as np
import numpy.typing as npt

SUPPORTED_REWARD_MODES = ("dense", "sparse", "semi_sparse")


def getRewardWrapper(task: str):
    #### medium task: basketball, bin-picking, box-close, coffee-pull, coffee-push, hammer, peg-insert-side, push-wall, soccer, sweep, sweep-into ####
    if task.startswith("basketball"):
        return Basketball_DEMO3
    if task.startswith("bin-picking"):
        return Bin_Picking_DEMO3
    if task.startswith("box-close"):
        return Box_Close_DEMO3
    if task.startswith("coffee-pull"):
        return Coffee_Pull_DEMO3
    if task.startswith("coffee-push"):
        return Coffee_Push_DEMO3
    if task.startswith("hammer"):
        return Hammer_DEMO3
    if task.startswith("peg-insert-side"):
        return PegInsertSide_DEMO3
    if task.startswith("push-wall"):
        return Push_Wall_DEMO3
    if task.startswith("soccer"):
        return Soccer_DEMO3
    if task.startswith("sweep"):
        return Sweep_DEMO3
    if task.startswith("sweep-into"):
        return Sweep_Into_DEMO3

    if task.startswith("window-close"):
        return Window_Close_DEMO3
    if task.startswith("window-open"):
        return Window_Open_DEMO3
    if task.startswith("assembly"):
        return Assembly_DEMO3
    if task.startswith("pick-place"):
        return PickAndPlace_DEMO3
    
    if task.startswith("pick-place-wall"):
        return PickAndPlaceWall_DEMO3
    if task.startswith("stick-push"):
        return StickPush_DEMO3
    if task.startswith("stick-pull"):
        return StickPull_DEMO3
    if task.startswith("hand-insert"):
        return HandAndInsert_DEMO3
    if task.startswith("pick-out-of-hole"):
        return PickOutOfHole_DEMO3
    if task.startswith("shelf-place"):
        return ShelfAndPlace_DEMO3
    if task.startswith("pick-place-wall"):
        return PickAndPlaceWall_DEMO3
    if task.startswith("push"):
        return Push_DEMO3
    if task.startswith("push-back"):
        return PushBack_DEMO3
    if task.startswith("disassemble"):
        return Disassemble_DEMO3
    raise NotImplementedError(f"Task {task} is not supported yet.")


class MetaWorldRewardWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, cfg):
        super().__init__(env)
        if cfg.reward_mode not in SUPPORTED_REWARD_MODES:
            self.reward_mode = SUPPORTED_REWARD_MODES[0]
        else:
            self.reward_mode = cfg.reward_mode
        self._info = {}

    def step(self, action: npt.NDArray[np.float32]):
        obs, rew, termindated, truncated, info = self.env.step(action)
        self._info = info
        if self.reward_mode == "sparse":
            rew = float(info["success"])
        elif self.reward_mode == "dense":
            rew = rew
        elif self.reward_mode == "semi_sparse":
            rew = self.compute_semi_sparse_reward(info)
        else:
            raise NotImplementedError(self.reward_mode)
        return obs, rew, termindated, truncated, info

    def compute_stage_indicator(self):
        raise NotImplementedError()

    def reward(self, *args, **kwargs):
        return self.compute_semi_sparse_reward(self._info)

    def compute_semi_sparse_reward(self, info):
        stage_indicators = self.compute_stage_indicator(info)
        assert len(stage_indicators.keys()) <= self.n_stages
        reward = sum(stage_indicators.values())
        assert reward.is_integer(), "Semi-sparse reward is not an integer"
        return reward


############################################
# Assembly (Hard)
############################################
class Assembly_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }


############################################
# Pick And Place (Hard)
############################################
class PickAndPlace_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Stick Pull (Very Hard)
############################################
class StickPull_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }


############################################
# Stick Push (Very Hard)
############################################
class StickPush_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }


############################################
# Pick And Place (Hard)
############################################
class PickAndPlaceWall_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }


############################################
# hand-insert (Hard)
############################################
class HandAndInsert_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }
    

############################################
# pick-out-of-hole (Hard)
############################################
class PickOutOfHole_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }
    

############################################
# shelf-place (Very Hard)
############################################
class ShelfAndPlace_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }
    
############################################
# pick-place-wall (VeryHard)
############################################
class PickPlaceWall_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }


############################################
# push (Hard)
############################################
class Push_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }


############################################
# push_back (Hard)
############################################
class PushBack_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }


############################################
# disassemble (Very Hard)
############################################
class Disassemble_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }


############################################
# Basketball (Medium)
############################################
class Basketball_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Bin Picking (Medium)
############################################
class Bin_Picking_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2 

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Box Close (Medium)
############################################
class Box_Close_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2 

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Coffee Pull (Medium)
############################################
class Coffee_Pull_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2 

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Coffee Push (Medium)
############################################
class Coffee_Push_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2 

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Hammer (Medium)
############################################
class Hammer_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Peg Insert Side (Medium)
############################################
class PegInsertSide_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2 

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Push Wall (Medium)
############################################
class Push_Wall_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2 

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Soccer (Medium)
############################################
class Soccer_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2 

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Sweep (Medium)
############################################
class Sweep_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2 

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }



############################################
# Sweep Into (Medium)
############################################
class Sweep_Into_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 3 # Grasp broom, Push dirt, Success (Into bin)

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }






############################################
# window close (easy)
############################################
class Window_Close_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }

############################################
# window open (easy)
############################################
class Window_Open_DEMO3(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2

    def compute_stage_indicator(self, eval_info):
        return {
            "is_grasped": float(eval_info["grasp_success"] or eval_info["success"]),
            "success": float(eval_info["success"]),
        }