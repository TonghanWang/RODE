REGISTRY = {}

from .obs_reward_encoder import ObsRewardEncoder

REGISTRY["obs_reward"] = ObsRewardEncoder
