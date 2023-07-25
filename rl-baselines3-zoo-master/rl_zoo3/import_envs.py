from typing import Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs  # pytype: disable=import-error
except ImportError:
    pybullet_envs = None

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    highway_env = None
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

try:
    import neck_rl  # pytype: disable=import-error
except ImportError:
    neck_rl = None

try:
    import mocca_envs  # pytype: disable=import-error
except ImportError:
    mocca_envs = None

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    custom_envs = None

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    gym_donkeycar = None

try:
    import panda_gym  # pytype: disable=import-error
except ImportError:
    panda_gym = None

try:
    import rocket_lander_gym  # pytype: disable=import-error
except ImportError:
    rocket_lander_gym = None

try:
    import minigrid  # pytype: disable=import-error
except ImportError:
    minigrid = None


# Register no vel envs
def create_no_vel_env(env_id: str):
    def make_env(render_mode: Optional[str] = None):
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),
    )

class SwimmerRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SwimmerRewardWrapper, self).__init__(env)
        # Swimmer-v3 환경의 행동 공간과 상태 공간 정보를 가져옵니다.
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        # 원래 step 메서드 호출하여 다음 상태와 보상, 종료 여부 등을 얻습니다.
        next_state, original_reward, done, info = self.env.step(action)

        # 오른쪽으로 돌면 추가 보상을 줍니다.
        right_reward = 0.0
        if action == 2:  # 예시로, Swimmer-v3에서 'right' 액션의 인덱스가 2일 경우
            right_reward = 0.5  # 오른쪽으로 돌 때 추가 보상

        # 새로운 보상을 계산하여 원래 보상과 더합니다.
        reward = original_reward + right_reward

        return next_state, reward, done, info